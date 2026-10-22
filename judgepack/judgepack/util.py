"""Small text and progress helpers shared across modules."""

from __future__ import annotations

import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_URL = re.compile(r"https?://[^\s,;\"'<>()\[\]]+", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")

T = TypeVar("T")
R = TypeVar("R")


def norm(text: object) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces. Used for matching names."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return _NON_ALNUM.sub(" ", t).strip()


def slug(text: str) -> str:
    return norm(text).replace(" ", "-") or "untitled"


def find_urls(text: str | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in _URL.findall(text or ""):
        url = url.rstrip(".,)")
        if url.lower() not in seen:
            seen.add(url.lower())
            out.append(url)
    return out


def split_list(text: str | None) -> list[str]:
    """Split a 'built with' style cell on commas, semicolons, pipes or newlines."""
    seen: set[str] = set()
    out: list[str] = []
    for part in re.split(r"[,;|\n]+", text or ""):
        part = part.strip(" \t\"'")
        if part and norm(part) not in seen:
            seen.add(norm(part))
            out.append(part)
    return out


def word_count(text: str | None) -> int:
    return len(_WORD.findall(text or ""))


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    cut = text.rfind(" ", 0, max_chars)
    cut = cut if cut > max_chars * 0.8 else max_chars
    return text[:cut].rstrip() + " […]", True


def short(text: object, limit: int = 180) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def progress(label: str, done: int, total: int) -> None:
    """Minimal progress line on stderr."""
    if total <= 0:
        return
    if sys.stderr.isatty():
        end = "\n" if done >= total else ""
        sys.stderr.write(f"\r  {label}: {done}/{total}{end}")
        sys.stderr.flush()
    elif done == total or done % max(1, total // 5) == 0:
        sys.stderr.write(f"  {label}: {done}/{total}\n")


def run_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int,
    label: str,
    catch: tuple[type[BaseException], ...] = (),
    stop: tuple[type[BaseException], ...] = (),
) -> tuple[list[R | None], list[str]]:
    """Run fn over items with a thread pool, keeping input order.

    Exceptions listed in `catch` are recorded as errors (that item's result is None).
    Exceptions listed in `stop` (checked first) cancel the items not yet started and propagate,
    e.g. a quota that is used up for the day. Anything else is a bug and propagates.
    """
    results: list[R | None] = [None] * len(items)
    errors: list[str] = []
    if not items:
        return results, errors
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except stop:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except catch as exc:  # noqa: PERF203 - one failure must not sink the batch
                errors.append(str(exc))
            done += 1
            progress(label, done, len(items))
    return results, errors
