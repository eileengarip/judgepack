"""Model providers plus the plumbing around them: caching, retries, rate limiting, usage.

Every response is validated against a pydantic schema and cached on disk, keyed by the
exact prompt. Re-running an evaluation is free and gives identical results.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .config import DEFAULT_MODELS, ModelConfig
from .util import short

T = TypeVar("T", bound=BaseModel)
CACHE_VERSION = "1"


class LLMError(RuntimeError):
    """A model call failed in a way the user needs to know about."""


class QuotaExhausted(LLMError):
    """The provider's quota is used up for the day (or is zero). Retrying won't help, so the whole run stops."""


def quota_details(exc: BaseException) -> dict[str, Any]:
    """What a 429 error says about the quota it hit: which quota, its limit, and how long to wait.

    Gemini puts this in the error body (QuotaFailure and RetryInfo). Returns {} if it isn't there.
    """
    text = str(exc)
    details = getattr(exc, "details", None)
    if details is not None:
        try:
            text += " " + json.dumps(details, default=str)
        except (TypeError, ValueError):
            text += " " + str(details)
    ids = re.findall(r"""["']quotaId["']\s*:\s*["']([^"']+)""", text)
    values = [int(v) for v in re.findall(r"""["']quotaValue["']\s*:\s*["']?(\d+)""", text)]
    info: dict[str, Any] = {}
    if ids:
        daily = [i for i, q in enumerate(ids) if "perday" in q.lower()]
        k = daily[0] if daily else 0
        info["quota_id"] = ids[k]
        info["daily"] = bool(daily)
        if k < len(values):
            info["limit"] = values[k]
    delay = re.search(r"""["']retryDelay["']\s*:\s*["']?(\d+(?:\.\d+)?)s""", text)
    if delay:
        info["retry_after"] = float(delay.group(1))
    return info


def quota_message(provider: str, model: str, info: dict[str, Any]) -> str | None:
    """A plain explanation if this quota can't recover by waiting a minute; None if a short wait will do."""
    limit = info.get("limit")
    saved = "Finished calls are saved, so nothing is lost: running the same command again carries on from there."
    if limit == 0:
        return (f"{provider} gives '{model}' no free quota on this project, so every call is refused. "
                f"Check your limits at aistudio.google.com/rate-limit, or link a billing account in AI Studio. {saved}")
    if info.get("daily"):
        per_day = f" ({limit:,} requests a day)" if limit else ""
        return (f"{provider}'s daily quota for '{model}' is used up{per_day}. It resets at midnight Pacific time "
                f"(around 8am UK time); a billing account in AI Studio lifts it. {saved}")
    return None


@dataclass
class RawResponse:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class TaskUsage:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Usage:
    """Thread-safe token and call accounting, per task."""

    def __init__(self) -> None:
        self.by_task: dict[str, TaskUsage] = {}
        self._lock = threading.Lock()

    def record(self, task: str, *, call: bool = False, cache_hit: bool = False, failure: bool = False,
               input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            u = self.by_task.setdefault(task, TaskUsage())
            u.calls += int(call)
            u.cache_hits += int(cache_hit)
            u.failures += int(failure)
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens

    def total(self) -> TaskUsage:
        t = TaskUsage()
        for u in self.by_task.values():
            t.calls += u.calls
            t.cache_hits += u.cache_hits
            t.failures += u.failures
            t.input_tokens += u.input_tokens
            t.output_tokens += u.output_tokens
        return t

    def cost(self, price_in: float | None, price_out: float | None) -> float | None:
        if price_in is None or price_out is None:
            return None
        t = self.total()
        return t.input_tokens / 1e6 * price_in + t.output_tokens / 1e6 * price_out

    def to_dict(self) -> dict[str, Any]:
        return {"by_task": {k: asdict(v) for k, v in self.by_task.items()}, "total": asdict(self.total())}


class Provider:
    name = "provider"
    model = ""
    is_mock = False

    def generate(self, *, system: str, prompt: str, schema: type[BaseModel], temperature: float,
                 task: str, context: dict[str, Any] | None) -> RawResponse:
        raise NotImplementedError


def gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic model to Gemini's schema format, with an explicit property order.

    Gemini may otherwise emit fields alphabetically, which would put a verdict ("better")
    before its reason. propertyOrdering keeps reasons first, as declared in schemas.py.
    """
    root = model.model_json_schema()
    defs = root.get("$defs", {})

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in node:
            return convert(defs[node["$ref"].rsplit("/", 1)[-1]])
        out: dict[str, Any] = {}
        if node.get("description"):
            out["description"] = node["description"]
        if "enum" in node:
            out.update(type="STRING", enum=[str(v) for v in node["enum"]])
            return out
        kind = node.get("type")
        if kind == "object":
            props = node.get("properties", {})
            out.update(
                type="OBJECT",
                properties={k: convert(v) for k, v in props.items()},
                required=list(node.get("required", props)),
                property_ordering=list(props),
            )
        elif kind == "array":
            out.update(type="ARRAY", items=convert(node.get("items", {"type": "string"})))
        else:
            out["type"] = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}.get(
                kind, "STRING"
            )
        return out

    return convert(root)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError('The Gemini provider needs the google-genai package: pip install -e ".[gemini]"') from exc
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("Set the GEMINI_API_KEY environment variable. You can create a key in Google AI Studio (aistudio.google.com).")
        self.model = model
        self._types = types
        self._client = genai.Client(api_key=key)
        self._schema_ok = True

    def generate(self, *, system, prompt, schema, temperature, task, context):
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if self._schema_ok:
            kwargs["response_schema"] = self._types.Schema.model_validate(gemini_schema(schema))
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=self._types.GenerateContentConfig(**kwargs)
            )
        except Exception as exc:
            # If the API ever rejects the schema itself, fall back to JSON mode; the prompt
            # spells out the fields and pydantic still validates every reply.
            if self._schema_ok and status_of(exc) == 400 and "schema" in str(exc).lower():
                self._schema_ok = False
                return self.generate(system=system, prompt=prompt, schema=schema, temperature=temperature,
                                     task=task, context=context)
            raise
        usage = getattr(resp, "usage_metadata", None)
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        return RawResponse(text=resp.text or "", input_tokens=inp, output_tokens=out)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError('The Anthropic provider needs the anthropic package: pip install -e ".[anthropic]"') from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("Set the ANTHROPIC_API_KEY environment variable.")
        self.model = model
        self._client = anthropic.Anthropic()

    def generate(self, *, system, prompt, schema, temperature, task, context):
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in msg.content)
        return RawResponse(text=text, input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)


class MockProvider(Provider):
    """Rule-based stand-in for tests and offline previews. Not an AI model."""

    name = "mock"
    is_mock = True

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def generate(self, *, system, prompt, schema, temperature, task, context):
        from . import mock

        ctx = context or {}
        if task == "brief":
            data = mock.mock_brief(ctx["submission"], ctx["config"])
        elif task == "score":
            data = mock.mock_score(ctx["submission"], ctx["criteria"], ctx.get("sample", 0))
        elif task == "pairwise":
            data = mock.mock_pairwise(ctx["a"], ctx["b"])
        else:
            raise LLMError(f"The mock provider has no rule for task '{task}'.")
        text = json.dumps(data)
        return RawResponse(text=text, input_tokens=len(system + prompt) // 4, output_tokens=len(text) // 4)


def make_provider(model_cfg: ModelConfig) -> Provider:
    name = model_cfg.provider
    if name == "gemini":
        return GeminiProvider(model_cfg.name)
    if name == "anthropic":
        return AnthropicProvider(model_cfg.name)
    if name == "mock":
        return MockProvider(model_cfg.name or DEFAULT_MODELS["mock"])
    raise LLMError(f"Unknown provider '{name}'")


def status_of(exc: BaseException) -> int | None:
    """HTTP status from a provider exception, if it carries one."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def extract_json(text: str) -> Any:
    """Parse a JSON object from a reply, tolerating code fences and stray prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("the reply contained no JSON object") from None
        try:
            return json.loads(t[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"the reply was not valid JSON ({exc.msg})") from None


class RateLimiter:
    """Spaces calls evenly so a free-tier key doesn't hit its per-minute limit."""

    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute and requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class LLM:
    def __init__(
        self,
        provider: Provider,
        *,
        cache_dir: str | Path | None = ".judgepack_cache",
        requests_per_minute: float = 10.0,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max(1, max_retries)
        self.usage = Usage()
        self._limiter = RateLimiter(0 if provider.is_mock else requests_per_minute)
        self._sleep = sleep
        self._quota_gone: str | None = None  # set once a daily quota runs out, so no other call tries

    def _key(self, task: str, system: str, prompt: str, schema: type[BaseModel], temperature: float, sample: int) -> str:
        blob = json.dumps(
            [CACHE_VERSION, self.provider.name, self.provider.model, task, system, prompt,
             schema.__name__, schema.model_json_schema(), round(temperature, 4), sample],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not path or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key: str, data: Any) -> None:
        path = self._cache_path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def run(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float,
        sample: int = 0,
        context: dict[str, Any] | None = None,
        check: Callable[[T], None] | None = None,
    ) -> T:
        """One validated model call. Raises LLMError if no usable reply arrives."""
        key = self._key(task, system, prompt, schema, temperature, sample)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                obj = schema.model_validate(cached)
                if check:
                    check(obj)
                self.usage.record(task, cache_hit=True)
                return obj
            except (ValidationError, ValueError):
                pass  # stale entry from an older schema: regenerate

        repair = ""
        last_error: BaseException | None = None
        for attempt in range(self.max_retries):
            self._limiter.wait()
            if self._quota_gone:
                raise QuotaExhausted(self._quota_gone)
            try:
                raw = self.provider.generate(system=system, prompt=prompt + repair, schema=schema,
                                             temperature=temperature, task=task, context=context)
            except LLMError:
                raise
            except Exception as exc:  # provider SDK errors
                status = status_of(exc)
                if status in (401, 403):
                    raise LLMError(f"{self.provider.name} rejected the API key ({status}). Check it and try again.") from exc
                if status == 404:
                    raise LLMError(f"{self.provider.name} doesn't recognise model '{self.provider.model}'. Check the name in your config.") from exc
                if status is not None and 400 <= status < 500 and status != 429:
                    raise LLMError(f"{self.provider.name} refused the request ({status}): {short(exc)}") from exc
                info = quota_details(exc) if status == 429 else {}
                fatal = quota_message(self.provider.name.capitalize(), self.provider.model, info)
                if fatal:
                    # A daily (or zero) quota won't recover by waiting, so don't spend minutes retrying every call.
                    self._quota_gone = fatal
                    raise QuotaExhausted(fatal) from exc
                last_error = exc
                self.usage.record(task, failure=True)
                if "retry_after" in info:
                    backoff = min(120.0, info["retry_after"]) + 1 + random.random()  # the server says how long
                else:
                    backoff = min(90.0, (2**attempt) * (6.0 if status == 429 else 1.5)) + random.random()
                self._sleep(backoff)
                continue

            self.usage.record(task, call=True, input_tokens=raw.input_tokens, output_tokens=raw.output_tokens)
            try:
                obj = schema.model_validate(extract_json(raw.text))
                if check:
                    check(obj)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                self.usage.record(task, failure=True)
                repair = (
                    "\n\nYour previous reply could not be used: "
                    f"{short(exc, 300)}\nReply again with only the JSON object, following every rule above."
                )
                continue
            self._cache_put(key, obj.model_dump(mode="json"))
            return obj
        raise LLMError(f"{task}: no usable reply after {self.max_retries} attempts ({short(last_error)})")
