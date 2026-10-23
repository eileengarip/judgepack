"""Event configuration: the judging rubric, each year's theme and sponsor challenges, the
paper judging sheet, and model settings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .util import norm

DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-haiku-5-5",
    "mock": "mock",
}


class ConfigError(ValueError):
    """The YAML config is missing something or contradicts itself."""


@dataclass
class Criterion:
    id: str
    name: str
    description: str
    weight: float = 1.0
    anchors: dict[int, str] = field(default_factory=dict)

    @property
    def needs_theme(self) -> bool:
        return "{theme}" in self.description or any("{theme}" in a for a in self.anchors.values())

    def with_theme(self, theme: str) -> "Criterion":
        return replace(
            self,
            description=self.description.replace("{theme}", theme),
            anchors={k: v.replace("{theme}", theme) for k, v in self.anchors.items()},
        )


@dataclass
class Challenge:
    id: str
    name: str
    brief: str = ""  # empty when the brief couldn't be found: no fit is assessed for it
    sponsor: str = ""
    prize_names: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [n for n in (self.id, self.name, *self.prize_names) if norm(n)]

    def matches(self, text: str) -> bool:
        """True if a prize name, an opt-in or a judging-sheet label refers to this challenge."""
        hay = norm(text)
        if not hay:
            return False
        return any(norm(n) in hay for n in (self.name, *self.prize_names) if norm(n))

    def is_label(self, label: str) -> bool:
        """Exact match against the short names used on judging sheets (e.g. 'PPL')."""
        key = norm(label)
        return bool(key) and any(norm(n) == key for n in self.labels)


@dataclass
class EventSpec:
    """What's specific to one year: its theme, context, challenges and judging sheet."""

    name: str
    theme: str = ""
    context: str = ""
    challenges: list[Challenge] | None = None  # None: use the top-level challenges
    judging_sheet: "list[SheetColumn] | None" = None  # None: use the top-level sheet


@dataclass
class SheetColumn:
    """One marking point on the paper judging sheet."""

    column: str
    max: float
    criterion: str | None = None  # the AI criterion it corresponds to; None if it can't be judged from a write-up


@dataclass
class ModelConfig:
    provider: str = "gemini"
    name: str = DEFAULT_MODELS["gemini"]
    temperature: float = 0.4
    samples: int = 3
    requests_per_minute: float = 10.0
    workers: int = 4
    max_retries: int = 5
    price_input_per_million: float | None = None
    price_output_per_million: float | None = None


@dataclass
class PairwiseConfig:
    opponents_per_project: int = 4
    max_chars_each: int = 6000
    seed: int = 7


@dataclass
class EvaluationConfig:
    # "any" means any prize counts as a win; otherwise a list of name fragments, e.g. ["overall", "1st"].
    win_prizes: str | list[str] = "any"
    # Prizes about who the team is rather than the project (e.g. beginners) are left out of the win label.
    exclude_prizes: list[str] = field(default_factory=list)
    bootstrap: int = 2000
    seed: int = 0


@dataclass
class Config:
    event_name: str
    event_context: str
    criteria: list[Criterion]
    challenges: list[Challenge] = field(default_factory=list)
    events: list[EventSpec] = field(default_factory=list)
    judging_sheet: list[SheetColumn] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    pairwise: PairwiseConfig = field(default_factory=PairwiseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    max_description_chars: int = 12000

    # ----------------------------------------------------------------- per-event views

    def event_spec(self, event: str) -> EventSpec:
        key = norm(event)
        for spec in self.events:
            if norm(spec.name) == key:
                return spec
        return EventSpec(name=event)

    def canonical_event(self, event: str) -> str:
        """The config's spelling of an event name, e.g. 'Adahack 2026' -> 'AdaHack 2026'."""
        key = norm(event)
        return next((s.name for s in self.events if norm(s.name) == key), event)

    def theme_for(self, event: str) -> str:
        return self.event_spec(event).theme

    def context_for(self, event: str) -> str:
        spec = self.event_spec(event)
        parts = [p for p in (self.event_context, spec.context) if p]
        if spec.theme:
            parts.append(f"This year's theme was {spec.theme}.")
        return " ".join(parts)

    def criteria_for(self, event: str) -> list[Criterion]:
        """The rubric for one event. A criterion about the theme is dropped if that year's theme is unknown."""
        theme = self.theme_for(event)
        out = []
        for c in self.criteria:
            if c.needs_theme:
                if theme:
                    out.append(c.with_theme(theme))
            else:
                out.append(c)
        return out

    def challenges_for(self, event: str) -> list[Challenge]:
        spec = self.event_spec(event)
        return self.challenges if spec.challenges is None else spec.challenges

    def briefed_challenges(self, event: str) -> list[Challenge]:
        """Challenges the AI can assess fit for: only those whose brief is known."""
        return [c for c in self.challenges_for(event) if c.brief]

    def challenge_for_label(self, event: str, label: str) -> Challenge | None:
        """Map a judging-sheet label such as 'JANE STREET' to a challenge."""
        challenges = self.challenges_for(event)
        exact = [c for c in challenges if c.is_label(label)]
        if len(exact) == 1:
            return exact[0]
        loose = [c for c in challenges if c.matches(label)]
        return loose[0] if len(loose) == 1 else None

    def sheet_for(self, event: str) -> list[SheetColumn]:
        """The paper judging sheet used that year (sheets changed between years)."""
        spec = self.event_spec(event)
        return self.judging_sheet if spec.judging_sheet is None else spec.judging_sheet

    def canonical_criterion(self, raw_id: str) -> str | None:
        key = norm(raw_id)
        for c in self.criteria:
            if norm(c.id) == key or norm(c.name) == key:
                return c.id
        return None


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(d, dict) or d.get(key) in (None, ""):
        raise ConfigError(f"{where}: missing '{key}'")
    return d[key]


def _opt_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _unique(ids: list[str], where: str) -> None:
    seen: set[str] = set()
    for i in ids:
        if norm(i) in seen:
            raise ConfigError(f"{where}: '{i}' is used twice")
        seen.add(norm(i))


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _challenges(items: Any, where: str) -> list[Challenge]:
    out: list[Challenge] = []
    for i, c in enumerate(items or []):
        w = f"{where}[{i}]"
        brief = c.get("brief") if isinstance(c, dict) else None
        out.append(
            Challenge(
                id=str(_require(c, "id", w)),
                name=str(_require(c, "name", w)),
                brief="" if brief in (None, "") else str(brief).strip(),
                sponsor=str(c.get("sponsor") or ""),
                prize_names=_str_list(c.get("prize_names")),
            )
        )
    _unique([c.id for c in out], where)
    return out


def load_config(path: str | Path) -> Config:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {p}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML ({exc})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a YAML mapping at the top level")
    return config_from_dict(data, where=p.name)


def _sheet(raw: Any, criterion_ids: set[str], where: str) -> list[SheetColumn]:
    sheet: list[SheetColumn] = []
    for i, col in enumerate(raw or []):
        w = f"{where}[{i}]"
        crit = col.get("criterion")
        if crit not in (None, "") and str(crit) not in criterion_ids:
            raise ConfigError(f"{w}: criterion '{crit}' isn't one of the criteria ids ({', '.join(sorted(criterion_ids))})")
        mx = float(_require(col, "max", w))
        if mx <= 0:
            raise ConfigError(f"{w}: max must be positive")
        sheet.append(SheetColumn(column=str(_require(col, "column", w)), max=mx,
                                 criterion=None if crit in (None, "") else str(crit)))
    _unique([s.column for s in sheet], where)
    return sheet


def config_from_dict(data: dict[str, Any], where: str = "config") -> Config:
    event = data.get("event") or {}
    event_name = str(_require(event, "name", f"{where} > event"))

    criteria: list[Criterion] = []
    for i, c in enumerate(data.get("criteria") or []):
        w = f"{where} > criteria[{i}]"
        anchors: dict[int, str] = {}
        for k, v in (c.get("anchors") or {}).items():
            try:
                level = int(k)
            except (TypeError, ValueError):
                raise ConfigError(f"{w}: anchor keys must be numbers 1-5, got {k!r}") from None
            if not 1 <= level <= 5:
                raise ConfigError(f"{w}: anchor keys must be 1-5, got {level}")
            anchors[level] = str(v).strip()
        weight = float(c.get("weight", 1.0))
        if weight <= 0:
            raise ConfigError(f"{w}: weight must be positive")
        criteria.append(
            Criterion(
                id=str(_require(c, "id", w)),
                name=str(_require(c, "name", w)),
                description=str(_require(c, "description", w)).strip(),
                weight=weight,
                anchors=dict(sorted(anchors.items())),
            )
        )
    if not criteria:
        raise ConfigError(f"{where}: add at least one judging criterion under 'criteria'")
    _unique([c.id for c in criteria], f"{where} > criteria")

    challenges = _challenges(data.get("challenges"), f"{where} > challenges")

    criterion_ids = {c.id for c in criteria}
    events: list[EventSpec] = []
    raw_events = data.get("events") or {}
    if not isinstance(raw_events, dict):
        raise ConfigError(f"{where} > events: expected one entry per year, e.g. 'AdaHack 2026:'")
    for name, spec in raw_events.items():
        spec = spec or {}
        w = f"{where} > events > {name}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{w}: expected theme / context / challenges underneath")
        events.append(
            EventSpec(
                name=str(name),
                theme=str(spec.get("theme") or "").strip(),
                context=str(spec.get("context") or "").strip(),
                challenges=_challenges(spec["challenges"], f"{w} > challenges") if "challenges" in spec else None,
                judging_sheet=(_sheet(spec["judging_sheet"], criterion_ids, f"{w} > judging_sheet")
                               if "judging_sheet" in spec else None),
            )
        )
    _unique([e.name for e in events], f"{where} > events")

    sheet = _sheet(data.get("judging_sheet"), criterion_ids, f"{where} > judging_sheet")

    m = data.get("model") or {}
    provider = str(m.get("provider", "gemini")).lower()
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"{where} > model: provider must be one of {', '.join(DEFAULT_MODELS)}")
    model = ModelConfig(
        provider=provider,
        name=str(m.get("name") or DEFAULT_MODELS[provider]),
        temperature=float(m.get("temperature", 0.4)),
        samples=max(1, int(m.get("samples", 3))),
        requests_per_minute=float(m.get("requests_per_minute", 10)),
        workers=max(1, int(m.get("workers", 4))),
        max_retries=max(1, int(m.get("max_retries", 5))),
        price_input_per_million=_opt_float(m.get("price_input_per_million")),
        price_output_per_million=_opt_float(m.get("price_output_per_million")),
    )

    pw = data.get("pairwise") or {}
    pairwise = PairwiseConfig(
        opponents_per_project=max(1, int(pw.get("opponents_per_project", 4))),
        max_chars_each=int(pw.get("max_chars_each", 6000)),
        seed=int(pw.get("seed", 7)),
    )

    ev = data.get("evaluation") or {}
    win = ev.get("win_prizes", "any")
    if isinstance(win, str) and win.strip().lower() != "any":
        win = [win]
    if isinstance(win, str):
        win = "any"
    evaluation = EvaluationConfig(
        win_prizes=win if win == "any" else [str(x) for x in win],
        exclude_prizes=_str_list(ev.get("exclude_prizes")),
        bootstrap=max(100, int(ev.get("bootstrap", 2000))),
        seed=int(ev.get("seed", 0)),
    )

    return Config(
        event_name=event_name,
        event_context=str(event.get("context") or "").strip(),
        criteria=criteria,
        challenges=challenges,
        events=events,
        judging_sheet=sheet,
        model=model,
        pairwise=pairwise,
        evaluation=evaluation,
        max_description_chars=int(data.get("max_description_chars", 12000)),
    )
