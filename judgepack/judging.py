"""Paper judging sheets: load them, match each one to a project, and compare the AI with the judges.

A sheet holds one team's scores in one sponsor challenge. Round 1 covers every team that
entered that challenge; round 2 re-scores the finalists, with different judges. Judges differ
between challenges, so every comparison is made within a challenge, never across them.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, SheetColumn
from .ingest import IngestError, Submission
from .pairwise import PairwiseResult
from .score import ProjectScore
from .stats import (
    bootstrap_ci,
    paired_bootstrap,
    within_group_auc,
    within_group_concordance,
    within_group_pairs,
)
from .util import norm


@dataclass
class SheetScores:
    line: int
    values: dict[str, float | None]  # sheet column -> score (None if blank or unreadable)
    typed_total: float | None

    def total(self, columns: list[SheetColumn]) -> float | None:
        vals = [self.values.get(c.column) for c in columns]
        return None if any(v is None for v in vals) else float(sum(vals))


@dataclass
class JudgedEntry:
    """One project's sheets in one challenge."""

    project_id: str
    title: str
    challenge_id: str
    challenge_name: str
    team: int
    rounds: dict[int, SheetScores] = field(default_factory=dict)
    opted_in: bool = True

    @property
    def finalist(self) -> bool:
        return 2 in self.rounds


@dataclass
class JudgingData:
    event: str
    columns: list[SheetColumn]
    entries: list[JudgedEntry]
    rows: int
    unmatched: list[str]
    warnings: list[str]
    total_mismatches: list[str]
    not_opted_in: list[str]
    opted_but_unjudged: list[str]
    challenges_without_sheets: list[str]

    @property
    def comparable_columns(self) -> list[SheetColumn]:
        """Sheet columns that a write-up can speak to (those mapped to an AI criterion)."""
        return [c for c in self.columns if c.criterion]


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise IngestError(f"File not found: {path}") from None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = [h for h in (reader.fieldnames or []) if h]
    return headers, [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in reader]


def _num(text: str) -> float | None:
    t = (text or "").strip().replace(",", ".")
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _load_team_overrides(path: Path | None) -> dict[tuple[int, str], str]:
    """Optional teams.csv: team, challenge (optional), project."""
    if path is None:
        return {}
    headers, rows = _read(path)
    by = {norm(h): h for h in headers}
    team_col, proj_col, ch_col = by.get("team"), by.get("project"), by.get("challenge")
    if not team_col or not proj_col:
        raise IngestError(f"{path.name}: needs 'team' and 'project' columns (and optionally 'challenge').")
    out: dict[tuple[int, str], str] = {}
    for r in rows:
        t = _num(r.get(team_col, ""))
        if t is None or not r.get(proj_col):
            continue
        out[(int(t), norm(r.get(ch_col, "")) if ch_col else "")] = r[proj_col]
    return out


def load_judging(path: str | Path, submissions: list[Submission], cfg: Config, event: str,
                 teams_path: str | Path | None = None) -> JudgingData:
    path = Path(path)
    event = cfg.canonical_event(event)
    if not cfg.sheet_for(event):
        raise IngestError(f"The config has no judging sheet for '{event}', so the sheet columns can't be read. "
                          "Add a 'judging_sheet' under that year in 'events'.")
    subs = [s for s in submissions if norm(s.event) == norm(event)]
    if not subs:
        known = sorted({s.event for s in submissions})
        raise IngestError(f"No submissions for '{event}'. Events loaded: {', '.join(known)}")
    columns = cfg.sheet_for(event)
    headers, rows = _read(path)
    by = {norm(h): h for h in headers}
    need = {"round": by.get("round"), "team": by.get("team"),
            "challenge": by.get("challenge") or by.get("challenges")}
    missing = [k for k, v in need.items() if not v] + [c.column for c in columns if norm(c.column) not in by]
    if missing:
        raise IngestError(f"{path.name}: missing column(s) {', '.join(missing)}. Columns found: {', '.join(headers)}")
    total_col = by.get("total")
    overrides = _load_team_overrides(Path(teams_path) if teams_path else None)
    title_lookup = {norm(t): s for s in subs for t in s.titles}

    entries: dict[tuple[str, str], JudgedEntry] = {}
    unmatched, warnings, mismatches, not_opted = [], [], [], []
    for line, r in enumerate(rows, start=2):
        rnd, team = _num(r[need["round"]]), _num(r[need["team"]])
        label = r[need["challenge"]]
        if rnd is None and team is None and not label:
            continue  # blank line
        if rnd not in (1.0, 2.0) or team is None or not label:
            warnings.append(f"line {line}: needs round 1 or 2, a team number and a challenge; skipped")
            continue
        challenge = cfg.challenge_for_label(event, label)
        if challenge is None:
            unmatched.append(f"line {line}: challenge '{label}' isn't one of this year's challenges in the config")
            continue

        values: dict[str, float | None] = {}
        for col in columns:
            raw = r[by[norm(col.column)]]
            v = _num(raw)
            if raw and v is None:
                warnings.append(f"line {line}: '{raw}' in {col.column} isn't a number; treated as blank")
            if v is not None and not 0 <= v <= col.max:
                warnings.append(f"line {line}: {col.column} = {v:g} is outside 0-{col.max:g}; treated as blank")
                v = None
            values[col.column] = v
        typed = _num(r[total_col]) if total_col else None
        sheet = SheetScores(line=line, values=values, typed_total=typed)
        computed = sheet.total(columns)
        if typed is not None and computed is not None and abs(typed - computed) > 0.01:
            mismatches.append(f"line {line}: team {int(team)} {challenge.name} round {int(rnd)}: "
                              f"total written {typed:g}, scores add up to {computed:g}")

        # Which project? An override first, then the table number, then the challenge opt-in.
        project: Submission | None = None
        key = (int(team), norm(label))
        name = overrides.get(key) or overrides.get((int(team), norm(challenge.id))) or overrides.get((int(team), ""))
        if name:
            project = title_lookup.get(norm(name))
            if project is None:
                unmatched.append(f"line {line}: teams file names '{name}', which isn't a {event} project")
                continue
        else:
            at_table = [s for s in subs if s.table_no == int(team)]
            if len(at_table) == 1:
                project = at_table[0]
            elif len(at_table) > 1:
                opted = [s for s in at_table if challenge.matches(s.opted_in_raw)]
                if len(opted) == 1:
                    project = opted[0]
                else:
                    names = ", ".join(s.title for s in at_table)
                    unmatched.append(f"line {line}: team {int(team)} {challenge.name}: several projects at that table "
                                     f"({names}) and the opt-ins don't settle it; add it to a teams file")
                    continue
            else:
                unmatched.append(f"line {line}: no {event} project has table number {int(team)}")
                continue

        ekey = (project.id, challenge.id)
        entry = entries.get(ekey)
        if entry is None:
            entry = entries[ekey] = JudgedEntry(project_id=project.id, title=project.title,
                                                challenge_id=challenge.id, challenge_name=challenge.name,
                                                team=int(team), opted_in=challenge.matches(project.opted_in_raw))
            if not entry.opted_in:
                not_opted.append(f"{project.title} (table {int(team)}) was judged for {challenge.name} "
                                 "but didn't opt into it on Devpost")
        if int(rnd) in entry.rounds:
            warnings.append(f"line {line}: a second round {int(rnd)} sheet for {project.title} in {challenge.name}; "
                            f"kept the first (line {entry.rounds[int(rnd)].line})")
            continue
        entry.rounds[int(rnd)] = sheet

    for e in entries.values():
        if 2 in e.rounds and 1 not in e.rounds:
            warnings.append(f"{e.title} has a round 2 sheet for {e.challenge_name} but no round 1 sheet")

    judged = {(e.project_id, e.challenge_id) for e in entries.values()}
    challenges = cfg.challenges_for(event)
    unjudged = [f"{s.title} ({c.name})" for s in subs for c in challenges
                if c.matches(s.opted_in_raw) and (s.id, c.id) not in judged]
    with_sheets = {e.challenge_id for e in entries.values()}
    no_sheets = [c.name for c in challenges if c.id not in with_sheets]
    return JudgingData(event=event, columns=columns, entries=list(entries.values()), rows=len(rows),
                       unmatched=unmatched, warnings=warnings, total_mismatches=mismatches,
                       not_opted_in=not_opted, opted_but_unjudged=unjudged, challenges_without_sheets=no_sheets)


# --------------------------------------------------------------------------- evaluation


def _r(v: float | None) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _ci(t: tuple[float, float, float]) -> list[float | None]:
    return [_r(x) for x in t]


def ai_sheet_score(score: ProjectScore, columns: list[SheetColumn]) -> float:
    """The AI's criterion scores combined with the sheet's own weights (a 20-mark column counts double)."""
    parts = [(score.criterion_means.get(c.criterion), c.max) for c in columns if c.criterion]
    parts = [(v, w) for v, w in parts if v is not None]
    if not parts:
        return math.nan
    return sum(v * w for v, w in parts) / sum(w for _, w in parts)


def compute_judging_results(data: JudgingData, subs: list[Submission], scores: dict[str, ProjectScore], cfg: Config,
                            *, pairwise: PairwiseResult | None = None, n_boot: int | None = None,
                            mock: bool = False, demo: bool = False, provider: str = "", model: str = "") -> dict[str, Any]:
    n_boot = n_boot or cfg.evaluation.bootstrap
    seed = cfg.evaluation.seed
    by_id = {s.id: s for s in subs}
    criteria_ids = {c.id for c in cfg.criteria_for(data.event)}
    # Only sheet columns the AI could score this year (e.g. theme is dropped if the theme is unknown).
    cols = [c for c in data.comparable_columns if c.criterion in criteria_ids]
    col_names = ", ".join(c.column for c in cols)
    all_cols = data.columns

    def arrays(rows: list[JudgedEntry], rnd: int):
        groups = np.array([e.challenge_id for e in rows])
        judge = np.array([e.rounds[rnd].total(cols) if rnd in e.rounds and e.rounds[rnd].total(cols) is not None
                          else np.nan for e in rows], dtype=float)
        judge_all = np.array([e.rounds[rnd].total(all_cols) if rnd in e.rounds and e.rounds[rnd].total(all_cols)
                              is not None else np.nan for e in rows], dtype=float)
        ai = np.array([ai_sheet_score(scores[e.project_id], cols) if e.project_id in scores else np.nan
                       for e in rows], dtype=float)
        words = np.array([by_id[e.project_id].word_count for e in rows], dtype=float)
        bt = None
        if pairwise is not None:
            bt = np.array([pairwise.strengths.get(e.project_id, np.nan) for e in rows], dtype=float)
        return groups, judge, judge_all, ai, words, bt

    def conc(a, b, g) -> dict[str, Any]:
        ok = np.isfinite(a) & np.isfinite(b)
        point, lo, hi = bootstrap_ci(within_group_concordance, a[ok], b[ok], g[ok], n_boot, seed)
        return {"value": _ci((point, lo, hi)), "pairs": within_group_pairs(b[ok], g[ok])}

    r1 = [e for e in data.entries if 1 in e.rounds]
    g1, j1, j1_all, ai1, w1, bt1 = arrays(r1, 1)
    methods = [
        {"key": "ai", "label": "AI rubric score", "is_ai": True, **conc(ai1, j1, g1)},
    ]
    if bt1 is not None:
        methods.append({"key": "pairwise", "label": "AI pairwise comparisons", "is_ai": True, **conc(bt1, j1, g1)})
    methods.append({"key": "length", "label": "Write-up length", "is_ai": False, **conc(w1, j1, g1)})

    def paired(a, b, j, g):
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(j)
        r = paired_bootstrap(within_group_concordance, a[ok], b[ok], j[ok], g[ok], n_boot, seed)
        return {k: _r(v) for k, v in r.items()}

    ai_vs_length = paired(ai1, w1, j1, g1)

    per_criterion = []
    for c in cols:
        crit = next(x for x in cfg.criteria_for(data.event) if x.id == c.criterion)
        jc = np.array([e.rounds[1].values.get(c.column) if e.rounds[1].values.get(c.column) is not None else np.nan
                       for e in r1], dtype=float)
        ac = np.array([scores[e.project_id].criterion_means.get(c.criterion, np.nan) if e.project_id in scores
                       else np.nan for e in r1], dtype=float)
        per_criterion.append({"column": c.column, "criterion": crit.name, "max": c.max, **conc(ac, jc, g1)})

    full_sheet = {"label": "AI vs the judges' full round 1 total (all five marks)", **conc(ai1, j1_all, g1)}

    # Finalists: the only teams scored by two different sets of judges.
    fin = [e for e in data.entries if 1 in e.rounds and 2 in e.rounds]
    finalists: dict[str, Any] = {"n": len(fin)}
    if fin:
        gf, jf1, _, aif, wf, _ = arrays(fin, 1)
        _, jf2, _, _, _, _ = arrays(fin, 2)
        finalists.update(
            judges=conc(jf1, jf2, gf),
            ai_round2=conc(aif, jf2, gf),
            ai_round1=conc(aif, jf1, gf),
            length_round2=conc(wf, jf2, gf),
        )

    # Did the AI rank the teams that reached the final above those that didn't?
    reach: dict[str, Any] | None = None
    fl = np.array([int(e.finalist) for e in r1])
    if fl.any() and not fl.all():
        def auc(a):
            ok = np.isfinite(a)
            return _ci(bootstrap_ci(within_group_auc, a[ok], fl[ok], g1[ok], n_boot, seed))
        reach = {"ai": auc(ai1), "length": auc(w1), "judges_round1_total": auc(j1_all),
                 "finalists": int(fl.sum()), "entries": len(fl)}
        if bt1 is not None:
            reach["pairwise"] = auc(bt1)

    per_challenge = []
    for cid in dict.fromkeys(e.challenge_id for e in r1):
        m = g1 == cid
        name = next(e.challenge_name for e in r1 if e.challenge_id == cid)
        okj = m & np.isfinite(j1)
        per_challenge.append({
            "challenge": name,
            "entries": int(m.sum()),
            "finalists": int(sum(1 for e in r1 if e.challenge_id == cid and e.finalist)),
            "pairs": within_group_pairs(j1[okj], g1[okj]),
            "ai": _r(within_group_concordance(ai1[okj], j1[okj], g1[okj])),
            "length": _r(within_group_concordance(w1[okj], j1[okj], g1[okj])),
        })

    rows_out = []
    for e, a, j in zip(r1, ai1, j1):
        sc = scores.get(e.project_id)
        rows_out.append({
            "challenge": e.challenge_name, "team": e.team, "title": e.title, "finalist": e.finalist,
            "judges_round1": _r(j), "judges_round2": _r(e.rounds[2].total(cols)) if 2 in e.rounds else None,
            "ai": _r(a), "ai_sd": _r(sc.overall_sd) if sc else None,
            "words": by_id[e.project_id].word_count, "opted_in": e.opted_in,
        })

    ai_main = methods[0]["value"]
    length_main = next(m for m in methods if m["key"] == "length")["value"]
    judges_agree = finalists.get("judges", {}).get("value") if fin else None
    results = {
        "meta": {
            "event": data.event,
            "provider": provider, "model": model, "mock": mock, "demo": demo,
            "sheets": data.rows, "entries": len(data.entries), "round1": len(r1), "finalists": len(fin),
            "projects": len({e.project_id for e in data.entries}),
            "challenges": len({e.challenge_id for e in data.entries}),
            "compared_columns": [c.column for c in cols],
            "compared_max": sum(c.max for c in cols),
            "left_out_columns": [c.column for c in all_cols if c not in cols],
            "bootstrap": n_boot,
            "samples": cfg.model.samples,
        },
        "methods": methods,
        "ai_vs_length": ai_vs_length,
        "per_criterion": per_criterion,
        "full_sheet": full_sheet,
        "finalists": finalists,
        "reach_final": reach,
        "per_challenge": per_challenge,
        "rows": rows_out,
        "data_notes": {
            "unmatched": data.unmatched, "warnings": data.warnings, "total_mismatches": data.total_mismatches,
            "not_opted_in": data.not_opted_in, "opted_but_unjudged": data.opted_but_unjudged,
            "challenges_without_sheets": data.challenges_without_sheets,
        },
    }
    results["headline"] = _headline(ai_main, length_main, judges_agree, finalists, col_names)
    results["caveats"] = _caveats(results)
    return results


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _headline(ai: list, length: list, judges: list | None, finalists: dict, cols: str) -> dict[str, Any]:
    s = (f"For two teams in the same challenge, the AI put them in the same order as the round 1 judges "
         f"{_pct(ai[0])} of the time")
    if ai[1] is not None:
        s += f" (95% interval {_pct(ai[1])} to {_pct(ai[2])})"
    s += f", counting only the marks a write-up can show ({cols}). Write-up length alone managed {_pct(length[0])}."
    ceiling = ""
    if judges and judges[0] is not None:
        ceiling = (f"For comparison, when two different sets of judges scored the same finalists, they agreed on the "
                   f"order {_pct(judges[0])} of the time, across {finalists['judges']['pairs']} pairs.")
    return {"sentence": s, "ceiling": ceiling}


def _caveats(r: dict[str, Any]) -> list[str]:
    meta, notes = r["meta"], r["data_notes"]
    out = []
    if meta.get("demo"):
        out.append("This is the built-in demo: the projects, sheets and scores were invented, so these numbers mean nothing.")
    if meta["mock"]:
        out.append("These numbers come from the mock provider, which uses simple text rules, not an AI model. "
                   "They must not be quoted as results.")
    pairs = r["methods"][0]["pairs"]
    out.append(f"The main result rests on {pairs} within-challenge pairs from one event, so the intervals are wide.")
    if meta["left_out_columns"]:
        cols = meta["left_out_columns"]
        named = cols[0] if len(cols) == 1 else ", ".join(cols[:-1]) + " and " + cols[-1]
        out.append(f"{named.capitalize()} can't be judged from a write-up, so they're left out "
                   "of the comparison. The judges' full totals are shown separately.")
    out.append("Judges saw a live demo and presentation; the AI only read the write-up. "
               "Full agreement isn't possible and isn't the aim.")
    if r["finalists"].get("n"):
        out.append("Finalists were picked for being strong, so they're closer together than the field as a whole. "
                   "That makes agreement among finalists look lower than it would across every team, for judges and AI alike.")
    if notes["total_mismatches"]:
        out.append(f"{len(notes['total_mismatches'])} sheet total(s) didn't match the individual marks. "
                   "The marks were used; check the sheets or the typing.")
    if notes["unmatched"]:
        out.append(f"{len(notes['unmatched'])} sheet row(s) couldn't be matched to a project and were left out.")
    return out
