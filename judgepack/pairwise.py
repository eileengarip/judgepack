"""Pairwise comparison: "which of these two is better?" instead of "score this out of 5".

People and models are often more consistent at comparisons than at absolute scores.
Each pair is shown in both orders, which cancels any preference for whichever project
comes first and lets us measure that preference directly. A Bradley-Terry model then
turns the wins and losses into one strength per project.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .config import Config
from .ingest import Submission
from .llm import LLM, LLMError, QuotaExhausted
from .prompts import pairwise_prompt, system_prompt
from .schemas import PairwiseJudgement
from .util import run_parallel

PAIRWISE_TEMPERATURE = 0.2


@dataclass
class Judgement:
    first: str   # project shown as A
    second: str  # project shown as B
    winner: str
    confidence: str
    reason: str


@dataclass
class PairwiseResult:
    judgements: list[Judgement]
    strengths: dict[str, float]  # log-strength; comparable within an event only

    @property
    def first_position_wins(self) -> int:
        return sum(1 for j in self.judgements if j.winner == j.first)

    def order_consistency(self) -> tuple[int, int]:
        """(pairs where both orders agree, pairs judged in both orders)."""
        by_pair: dict[tuple[str, str], list[str]] = {}
        for j in self.judgements:
            by_pair.setdefault(tuple(sorted((j.first, j.second))), []).append(j.winner)
        both = [w for w in by_pair.values() if len(w) == 2]
        return sum(1 for w in both if w[0] == w[1]), len(both)


def sample_pairs(ids: list[str], opponents: int, rng: random.Random) -> list[tuple[str, str]]:
    """Each project meets roughly `opponents` others, chosen at random."""
    n = len(ids)
    if n < 2:
        return []
    if n - 1 <= opponents:
        return [(ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)]
    pairs: set[tuple[str, str]] = set()
    for _ in range(opponents):
        order = ids[:]
        rng.shuffle(order)
        for i in range(0, n - 1, 2):
            pairs.add(tuple(sorted((order[i], order[i + 1]))))
        if n % 2:
            a = order[-1]
            b = rng.choice([x for x in ids if x != a])
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def bradley_terry(ids: list[str], games: list[tuple[str, str]], prior: float = 1.0,
                  max_iter: int = 10_000, tol: float = 1e-10) -> dict[str, float]:
    """Fit Bradley-Terry strengths by the MM algorithm (Hunter, 2004).

    games are (winner, loser). P(i beats j) = p_i / (p_i + p_j). Each project also gets
    `prior` wins and `prior` losses against a virtual opponent of strength 1, which keeps
    estimates finite for a project that won or lost every comparison.
    """
    if not ids:
        return {}
    index = {p: i for i, p in enumerate(ids)}
    n = len(ids)
    wins = np.zeros(n)
    played = np.zeros((n, n))
    for w, l in games:
        i, j = index[w], index[l]
        wins[i] += 1
        played[i, j] += 1
        played[j, i] += 1
    p = np.ones(n)
    for _ in range(max_iter):
        denom = (played / (p[:, None] + p[None, :])).sum(axis=1) + 2 * prior / (p + 1.0)
        new = (wins + prior) / denom
        converged = np.max(np.abs(np.log(new) - np.log(p))) < tol
        p = new
        if converged:
            break
    return {pid: float(np.log(p[index[pid]])) for pid in ids}


def run_pairwise(subs: list[Submission], cfg: Config, llm: LLM, workers: int) -> tuple[PairwiseResult, list[str]]:
    rng = random.Random(cfg.pairwise.seed)
    by_id = {s.id: s for s in subs}
    events: dict[str, list[str]] = {}
    for s in subs:
        events.setdefault(s.event, []).append(s.id)

    jobs: list[tuple[str, str]] = []
    for ids in events.values():
        for a, b in sample_pairs(ids, cfg.pairwise.opponents_per_project, rng):
            jobs.extend([(a, b), (b, a)])  # both orders

    def one(job: tuple[str, str]) -> PairwiseJudgement:
        a, b = by_id[job[0]], by_id[job[1]]
        return llm.run(
            task="pairwise",
            system=system_prompt(cfg, a.event),
            prompt=pairwise_prompt(cfg, a, b),
            schema=PairwiseJudgement,
            temperature=PAIRWISE_TEMPERATURE,
            context={"a": a, "b": b},
        )

    results, errors = run_parallel(one, jobs, workers, "pairwise", catch=(LLMError,), stop=(QuotaExhausted,))
    judgements = [
        Judgement(first=a, second=b, winner=a if r.better == "A" else b, confidence=r.confidence, reason=r.reason)
        for (a, b), r in zip(jobs, results)
        if r is not None
    ]
    strengths: dict[str, float] = {}
    for ids in events.values():
        idset = set(ids)
        games = [
            (j.winner, j.second if j.winner == j.first else j.first)
            for j in judgements
            if j.first in idset and j.second in idset
        ]
        strengths.update(bradley_terry(ids, games))
    return PairwiseResult(judgements=judgements, strengths=strengths), errors
