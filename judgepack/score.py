"""Rubric scoring with self-consistency: several independent samples per project.

Averaging samples steadies the score; the spread between samples tells us when the
model is unsure, which is exactly when a human should look more closely.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from .config import Config
from .ingest import Submission
from .llm import LLM, LLMError, QuotaExhausted
from .prompts import score_prompt, system_prompt
from .schemas import RubricScore
from .util import run_parallel

# A spread this large between samples (on the 1-5 scale) means the model isn't sure.
UNSTABLE_SD = 0.5


@dataclass
class ProjectScore:
    project_id: str
    overall: float
    overall_sd: float
    sample_overalls: list[float]
    criterion_means: dict[str, float]
    evidence: dict[str, str]
    evidence_levels: dict[str, str]
    biggest_uncertainty: str
    thin_evidence: bool
    unstable: bool = field(default=False)

    @property
    def needs_human_look(self) -> bool:
        return self.thin_evidence or self.unstable


def make_score_check(cfg: Config, event: str):
    wanted = [c.id for c in cfg.criteria_for(event)]

    def check(result: RubricScore) -> None:
        found: dict[str, int] = {}
        for cs in result.criteria:
            cid = cfg.canonical_criterion(cs.criterion_id)
            if cid is None or cid not in wanted:
                continue
            if not 1 <= cs.score <= 5:
                raise ValueError(f"score for '{cs.criterion_id}' must be an integer from 1 to 5, got {cs.score}")
            found[cid] = cs.score
        missing = [c for c in wanted if c not in found]
        if missing:
            raise ValueError(f"missing scores for: {', '.join(missing)}")

    return check


def aggregate(project_id: str, samples: list[RubricScore], cfg: Config, event: str) -> ProjectScore:
    criteria = cfg.criteria_for(event)
    weights = {c.id: c.weight for c in criteria}
    per_criterion: dict[str, list[int]] = {c.id: [] for c in criteria}
    levels: dict[str, list[str]] = {c.id: [] for c in criteria}
    evidence: dict[str, str] = {}
    overalls: list[float] = []
    for k, sample in enumerate(samples):
        by_id = {}
        for cs in sample.criteria:
            cid = cfg.canonical_criterion(cs.criterion_id)
            if cid and cid not in by_id:
                by_id[cid] = cs
        total = 0.0
        for c in criteria:
            cs = by_id[c.id]
            per_criterion[c.id].append(cs.score)
            levels[c.id].append(cs.evidence_level)
            if k == 0:
                evidence[c.id] = cs.evidence.strip()
            total += weights[c.id] * cs.score
        overalls.append(total / sum(weights.values()))
    level_mode = {cid: Counter(v).most_common(1)[0][0] for cid, v in levels.items()}
    sd = statistics.stdev(overalls) if len(overalls) > 1 else 0.0
    no_evidence = sum(1 for v in level_mode.values() if v == "none")
    return ProjectScore(
        project_id=project_id,
        overall=statistics.fmean(overalls),
        overall_sd=sd,
        sample_overalls=overalls,
        criterion_means={cid: statistics.fmean(v) for cid, v in per_criterion.items()},
        evidence=evidence,
        evidence_levels=level_mode,
        biggest_uncertainty=samples[0].biggest_uncertainty.strip(),
        thin_evidence=no_evidence * 2 >= len(criteria),
        unstable=sd >= UNSTABLE_SD,
    )


def score_projects(subs: list[Submission], cfg: Config, llm: LLM, workers: int) -> tuple[dict[str, ProjectScore], list[str]]:
    samples = cfg.model.samples
    checks = {e: make_score_check(cfg, e) for e in {s.event for s in subs}}
    jobs = [(sub, k) for sub in subs for k in range(samples)]

    def one(job: tuple[Submission, int]) -> RubricScore:
        sub, k = job
        return llm.run(
            task="score",
            system=system_prompt(cfg, sub.event),
            prompt=score_prompt(cfg, sub),
            schema=RubricScore,
            temperature=cfg.model.temperature,
            sample=k,
            context={"submission": sub, "criteria": cfg.criteria_for(sub.event), "sample": k},
            check=checks[sub.event],
        )

    results, errors = run_parallel(one, jobs, workers, "scores", catch=(LLMError,), stop=(QuotaExhausted,))
    grouped: dict[str, list[RubricScore]] = {}
    for (sub, _), res in zip(jobs, results):
        if res is not None:
            grouped.setdefault(sub.id, []).append(res)
    event_of = {s.id: s.event for s in subs}
    scores = {sid: aggregate(sid, rs, cfg, event_of[sid]) for sid, rs in grouped.items()}
    incomplete = [sid for sid, rs in grouped.items() if len(rs) < samples]
    if incomplete:
        errors.append(f"{len(incomplete)} project(s) scored with fewer than {samples} samples")
    return scores, errors
