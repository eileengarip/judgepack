"""Load hackathon submissions (e.g. a Devpost organiser export) and the list of winners.

Privacy by construction: only a whitelist of project fields is kept. Every other
column, including names and email addresses, is dropped at load time and never
reaches a model, a cache file or a report.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .util import find_urls, norm, slug, split_list, truncate, word_count

# Field -> header aliases, already normalised (lowercase, punctuation stripped).
# Devpost has used both "Project Title"/"Opt-In Prizes" and "Submission Title"/"Sponsor Prizes".
FIELD_ALIASES: dict[str, list[str]] = {
    "title": ["submission title", "project title", "project name", "title", "project", "name"],
    "tagline": ["tagline", "elevator pitch", "short description", "subtitle", "summary"],
    "description": ["about the project", "about", "description", "project description", "story", "write up", "writeup"],
    "built_with": ["built with", "technologies", "tech stack", "stack", "tools"],
    "links": ["try it out links", "try it out", "links", "github", "repository", "repo", "source code"],
    "video": ["video demo link", "demo video", "video link", "video", "youtube"],
    "url": ["submission url", "project url", "devpost url", "url"],
    "opted_in": ["sponsor prizes", "opt in prizes", "prizes", "challenges", "categories"],
    "table": ["table number", "table"],
    "event": ["event", "hackathon"],
    "status": ["project status", "submission status", "status"],
}
REQUIRED_FIELDS = ("title", "description")

# Headers that look like personal data are never mapped, even by accident.
PERSONAL_HINTS = (
    "email", "first name", "last name", "full name", "screen name", "phone", "submitter",
    "team member", "teammate", "colleague", "address", "date of birth", "age", "gender",
    "pronoun", "pronouns", "dietary", "school", "university", "universities", "college", "colleges",
    "student id", "linkedin",
)

# Organisers hide spam, tests and duplicate entries on Devpost; drafts were never submitted.
SKIP_STATUSES = ("hidden", "draft")
DUPLICATE_SIMILARITY = 0.9

# Sentences in which a team reports its own result ("Update: we won 2nd place!") are
# removed, otherwise the model could read the answer key. Deliberately narrow so that
# ordinary uses ("the winner of each round", "in the first place") survive.
_OUTCOME_PATTERNS = [
    # Trophy emoji only count beside a result word: "Score System 🏆" is decoration, "🏆 We won" is not.
    r"(?:🏆|🥇|🥈|🥉)[^.\n]{0,60}\b(?:won|wins?|winners?|place|prize)\b",
    r"\b(?:won|wins?|winners?|place|prize)\b[^.\n]{0,60}(?:🏆|🥇|🥈|🥉)",
    r"\bwe\s+(?:won|placed|came\s+(?:first|second|third|1st|2nd|3rd)|took\s+home)\b",
    r"\bwon\s+(?:the\s+|a\s+|an\s+)?(?:best|first|second|third|1st|2nd|3rd|overall|prize|award|category|challenge|track)\b",
    r"\b(?:1st|2nd|3rd|second|third)\s+place\b",
    r"(?<!\bin\sthe\s)\bfirst\s+place\b",
    r"\b(?:winners?|runners?[- ]up|finalists?)\s+(?:of|in|at|for)\s+(?:the\s+)?(?:best|overall|prize|award|category|challenge|track|hackathon)\b",
    r"\bjudges['’]?\s+(?:choice|favou?rite)\b",
    r"\bupdate\b[^.\n]*\b(?:won|winner|prize|placed)\b",
]
_OUTCOME = re.compile("|".join(f"(?:{p})" for p in _OUTCOME_PATTERNS), re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class IngestError(ValueError):
    """A submissions or winners file can't be read as expected."""


@dataclass
class Submission:
    id: str
    event: str
    title: str
    description: str
    tagline: str = ""
    built_with: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    video: str = ""
    url: str = ""
    opted_in_raw: str = ""
    table: str = ""
    word_count: int = 0
    redactions: int = 0
    truncated: bool = False
    order: int = 0
    aliases: list[str] = field(default_factory=list)  # titles of duplicate submissions merged into this one
    # Evaluation labels only. Never shown to a model.
    prizes: list[str] = field(default_factory=list)

    @property
    def table_no(self) -> int | None:
        """The table number as an integer: '23 Travel' -> 23, 'Used to be table 6, now 23' -> 23."""
        numbers = re.findall(r"\d+", self.table or "")
        return int(numbers[-1]) if numbers else None

    @property
    def titles(self) -> list[str]:
        return [self.title, *self.aliases]

    @property
    def has_repo(self) -> bool:
        return any(h in link.lower() for link in self.links for h in ("github.com", "gitlab.com", "bitbucket.org"))

    @property
    def has_video(self) -> bool:
        hosts = ("youtube.com", "youtu.be", "vimeo.com", "loom.com")
        return bool(self.video) or any(h in link.lower() for link in self.links for h in hosts)


@dataclass
class FileReport:
    path: str
    event_default: str
    rows: int
    kept: int
    mapping: dict[str, str]
    ignored_columns: list[str]
    personal_columns: list[str]
    skipped: list[str]
    hidden: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    submissions: list[Submission]
    files: list[FileReport]

    @property
    def redacted(self) -> list[Submission]:
        return [s for s in self.submissions if s.redactions]

    @property
    def truncated(self) -> list[Submission]:
        return [s for s in self.submissions if s.truncated]

    @property
    def events(self) -> list[str]:
        seen: list[str] = []
        for s in self.submissions:
            if s.event not in seen:
                seen.append(s.event)
        return seen


@dataclass
class WinnerReport:
    rows: int
    matched: int
    fuzzy: list[str]
    unmatched: list[str]
    winning_projects: int


def redact_outcomes(text: str) -> tuple[str, int]:
    """Drop sentences in which a team reports its own hackathon result."""
    if not text:
        return text, 0
    removed = 0
    lines_out: list[str] = []
    for line in text.splitlines():
        kept: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(line):
            if sentence.strip() and _OUTCOME.search(sentence):
                removed += 1
            else:
                kept.append(sentence)
        rebuilt = " ".join(s for s in kept if s.strip()).strip()
        if rebuilt or not line.strip():
            lines_out.append(rebuilt)
    out = "\n".join(lines_out).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out, removed


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise IngestError(f"File not found: {path}") from None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = [h for h in (reader.fieldnames or []) if h is not None]
    if not headers:
        raise IngestError(f"{path.name}: no header row found")
    rows = [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]
    return headers, rows


def _is_personal(normalised_header: str) -> bool:
    padded = f" {normalised_header} "
    return any(f" {hint} " in padded for hint in PERSONAL_HINTS)


def map_columns(
    headers: list[str], overrides: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str], list[str]]:
    """Work out which column holds which field. Returns (mapping, ignored, personal)."""
    overrides = overrides or {}
    by_norm: dict[str, str] = {}
    for h in headers:
        by_norm.setdefault(norm(h), h)

    mapping: dict[str, str] = {}
    claimed: set[str] = set()
    for fld, col in overrides.items():
        if fld not in FIELD_ALIASES:
            raise IngestError(f"--map: unknown field '{fld}'. Fields are: {', '.join(FIELD_ALIASES)}")
        match = col if col in headers else by_norm.get(norm(col))
        if not match:
            raise IngestError(f"--map {fld}: no column called '{col}'. Columns are: {', '.join(headers)}")
        mapping[fld] = match
        claimed.add(match)

    personal = [h for h in headers if _is_personal(norm(h)) and h not in claimed]
    for fld, aliases in FIELD_ALIASES.items():
        if fld in mapping:
            continue
        for alias in aliases:
            col = by_norm.get(alias)
            if col and col not in claimed and col not in personal:
                mapping[fld] = col
                claimed.add(col)
                break
        else:
            # Multi-word aliases may sit inside a longer header, e.g. "About the project (required)".
            for alias in (a for a in aliases if " " in a):
                cands = [h for n, h in by_norm.items() if alias in n and h not in claimed and h not in personal]
                if cands:
                    mapping[fld] = cands[0]
                    claimed.add(cands[0])
                    break
    ignored = [h for h in headers if h not in claimed and h not in personal]
    return mapping, ignored, personal


def _event_from_filename(path: Path) -> str:
    name = re.sub(r"[_\-]+", " ", path.stem).strip()
    return name.title() if name == name.lower() else name


def _table_key(sub: Submission) -> tuple[int, int, str]:
    digits = re.findall(r"\d+", sub.table)
    return (0, int(digits[0]), sub.table) if digits else (1, sub.order, sub.table)


def _similar(a: str, b: str) -> bool:
    if word_count(a) < 40 or word_count(b) < 40:
        return False
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return m.quick_ratio() >= DUPLICATE_SIMILARITY and m.ratio() >= DUPLICATE_SIMILARITY


def merge_duplicates(subs: list[Submission]) -> tuple[list[Submission], list[str]]:
    """Teams often submit the same project once per challenge. Count it once.

    Two submissions in the same event are the same project if their titles match, or their
    write-ups are at least 90% identical. Opt-ins and links are combined; the longer write-up is kept.
    """
    kept: list[Submission] = []
    notes: list[str] = []
    for sub in subs:
        twin = next(
            (k for k in kept if k.event == sub.event
             and (norm(k.title) == norm(sub.title) or _similar(k.description, sub.description))),
            None,
        )
        if twin is None:
            kept.append(sub)
            continue
        primary, other = (twin, sub) if len(twin.description) >= len(sub.description) else (sub, twin)
        if primary is sub:
            kept[kept.index(twin)] = sub
        opt = [o.strip() for o in f"{primary.opted_in_raw},{other.opted_in_raw}".split(",") if o.strip()]
        primary.opted_in_raw = ", ".join(dict.fromkeys(opt))
        primary.links = list(dict.fromkeys([*primary.links, *other.links]))
        primary.video = primary.video or other.video
        primary.table = primary.table or other.table
        primary.redactions += other.redactions
        primary.aliases = list(dict.fromkeys([*primary.aliases, *([] if other.title == primary.title else [other.title]),
                                              *other.aliases]))
        notes.append(f"'{other.title}' merged into '{primary.title}' ({primary.event})")
    return kept, notes


def load_submissions(
    paths: Iterable[str | Path],
    overrides: dict[str, str] | None = None,
    max_chars: int = 12000,
    include_hidden: bool = False,
) -> IngestResult:
    submissions: list[Submission] = []
    reports: list[FileReport] = []
    used_ids: set[str] = set()
    order = 0
    for raw_path in paths:
        path = Path(raw_path)
        headers, rows = _read_rows(path)
        mapping, ignored, personal = map_columns(headers, overrides)
        missing = [f for f in REQUIRED_FIELDS if f not in mapping]
        if missing:
            raise IngestError(
                f"{path.name}: couldn't find a column for {', '.join(missing)}. "
                f'Name it with --map, e.g. --map description="About The Project". '
                f"Columns found: {', '.join(headers)}"
            )
        default_event = _event_from_filename(path)
        skipped: list[str] = []
        hidden: list[str] = []
        file_subs: list[Submission] = []
        for line_no, row in enumerate(rows, start=2):

            def get(fld: str) -> str:
                return (row.get(mapping[fld]) or "").strip() if fld in mapping else ""

            title = get("title")
            if not title:
                skipped.append(f"line {line_no}: no project title")
                continue
            status = get("status").lower()
            if not include_hidden and any(word in status for word in SKIP_STATUSES):
                hidden.append(title)
                continue
            description_raw, tagline_raw = get("description"), get("tagline")
            if not description_raw and not tagline_raw:
                skipped.append(f"line {line_no} ({title}): empty write-up")
                continue
            description, n_desc = redact_outcomes(description_raw)
            tagline, n_tag = redact_outcomes(tagline_raw)
            words = word_count(f"{tagline} {description}")
            description, was_truncated = truncate(description, max_chars)
            event = get("event") or default_event

            base = f"{slug(event)}/{slug(title)}"
            sid, n = base, 2
            while sid in used_ids:
                sid, n = f"{base}-{n}", n + 1
            used_ids.add(sid)

            file_subs.append(
                Submission(
                    id=sid,
                    event=event,
                    title=title,
                    description=description,
                    tagline=tagline,
                    built_with=split_list(get("built_with")),
                    links=find_urls(get("links")),
                    video=(find_urls(get("video")) or [""])[0],
                    url=(find_urls(get("url")) or [""])[0],
                    opted_in_raw=get("opted_in"),
                    table=get("table"),
                    word_count=words,
                    redactions=n_desc + n_tag,
                    truncated=was_truncated,
                    order=order,
                )
            )
            order += 1
        file_subs, merged = merge_duplicates(file_subs)
        submissions.extend(file_subs)
        reports.append(
            FileReport(
                path=str(path),
                event_default=default_event,
                rows=len(rows),
                kept=len(file_subs),
                mapping=mapping,
                ignored_columns=ignored,
                personal_columns=personal,
                skipped=skipped,
                hidden=hidden,
                merged=merged,
            )
        )
    if not submissions:
        raise IngestError("No usable submissions found.")
    submissions.sort(key=lambda s: (s.event, _table_key(s)))
    return IngestResult(submissions=submissions, files=reports)


def _events_match(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    return bool(na and nb) and (na == nb or na in nb or nb in na)


def load_winners(path: str | Path, submissions: list[Submission]) -> WinnerReport:
    """Attach prizes to submissions from a CSV with columns project, prize and (optionally) event."""
    headers, rows = _read_rows(Path(path))
    by_norm = {norm(h): h for h in headers}

    def find(*aliases: str) -> str | None:
        return next((by_norm[a] for a in aliases if a in by_norm), None)

    title_col = find("project", "project title", "submission title", "title", "project name", "name")
    prize_col = find("prize", "award", "prize name", "category")
    event_col = find("event", "hackathon", "year")
    if not title_col or not prize_col:
        raise IngestError(f"{Path(path).name}: needs a 'project' column and a 'prize' column (and optionally 'event').")

    for s in submissions:
        s.prizes = []
    matched, fuzzy, unmatched = 0, [], []
    for line_no, row in enumerate(rows, start=2):
        title = (row.get(title_col) or "").strip()
        prize = (row.get(prize_col) or "").strip()
        event = (row.get(event_col) or "").strip() if event_col else ""
        if not title or not prize:
            continue
        pool = [s for s in submissions if not event or _events_match(event, s.event)]
        exact = [s for s in pool if any(norm(t) == norm(title) for t in s.titles)]
        target: Submission | None = None
        if len(exact) == 1:
            target = exact[0]
        elif len(exact) > 1:
            unmatched.append(f"line {line_no}: '{title}' matches projects in several events; add an 'event' column")
            continue
        else:
            names = {norm(t): s for s in pool for t in s.titles}
            close = difflib.get_close_matches(norm(title), list(names), n=1, cutoff=0.85)
            if close:
                target = names[close[0]]
                fuzzy.append(f"'{title}' -> '{target.title}' ({target.event})")
            else:
                hint = difflib.get_close_matches(norm(title), list(names), n=1, cutoff=0.5)
                suggestion = f"; did you mean '{names[hint[0]].title}'?" if hint else ""
                unmatched.append(f"line {line_no}: no project called '{title}'{suggestion}")
                continue
        if prize not in target.prizes:
            target.prizes.append(prize)
        matched += 1
    return WinnerReport(
        rows=len(rows),
        matched=matched,
        fuzzy=fuzzy,
        unmatched=unmatched,
        winning_projects=sum(1 for s in submissions if s.prizes),
    )
