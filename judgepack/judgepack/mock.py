"""A deterministic stand-in for a real model, so everything can be run and tested offline.

These outputs come from simple text rules plus hashing, not from any AI model. They exist
to exercise the pipeline and preview the reports. Never quote numbers produced with the
mock provider as findings.
"""

from __future__ import annotations

import hashlib
import math
import re

from .config import Config
from .ingest import Submission

POSITIVE = (
    "tested", "users", "working", "works", "demo", "deployed", "feedback", "accuracy",
    "live", "interviewed", "pilot", "measured", "volunteer",
)
NEGATIVE = (
    "plan to", "will be", "in the future", "revolution", "blockchain", "mockup", "mock-up",
    "eventually", "aim to", "vision", "disrupt", "game-changing", "next-generation",
)


def _u(*parts: object) -> float:
    """Deterministic pseudo-random number in [0, 1)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def latent_quality(sub: Submission) -> float:
    text = f"{sub.tagline} {sub.description}".lower()
    pos = sum(text.count(w) for w in POSITIVE)
    neg = sum(text.count(w) for w in NEGATIVE)
    q = (
        0.25 * min(1.0, math.log1p(sub.word_count) / math.log1p(300))
        + 0.10 * sub.has_video
        + 0.10 * sub.has_repo
        + 0.30 * min(1.0, pos / 4)
        - 0.25 * min(1.0, neg / 3)
        + 0.25 * _u("quality", sub.id)
    )
    return max(0.0, min(1.0, q))


def mock_brief(sub: Submission, cfg: Config) -> dict:
    challenges = cfg.briefed_challenges(sub.event)
    sents = _sentences(sub.description)
    one_line = " ".join((sub.tagline or (sents[0] if sents else sub.title)).split()[:20])
    tech = ", ".join(sub.built_with[:4]) or "technologies that the write-up does not state"
    strengths = [s for s in sents if any(w in s.lower() for w in POSITIVE)][:2]
    claims = [
        s
        for s in sents
        if re.search(r"\b(ai|machine learning|accura\w*|blockchain|revolution\w*|thousands|millions)\b", s, re.I)
    ][:2]
    text_words = set(re.findall(r"[a-z]{5,}", sub.description.lower()))
    fits = []
    for ch in challenges:
        brief_words = set(re.findall(r"[a-z]{5,}", ch.brief.lower()))
        overlap = len(brief_words & text_words) / max(1, len(brief_words))
        score = overlap + 0.15 * _u("fit", sub.id, ch.id)
        fit = "strong" if score > 0.24 else "possible" if score > 0.12 else "none"
        fits.append(
            {
                "challenge_id": ch.id,
                "reason": f"Mock rule: {round(overlap * 100)}% of the challenge brief's key words appear in the write-up.",
                "fit": fit,
            }
        )
    return {
        "one_line": one_line,
        "summary": " ".join(sents[:2]) if sents else sub.title,
        "how_it_works": f"Built with {tech}.",
        "strengths": strengths,
        "questions_for_team": [
            "What can you show working live right now?",
            "Who did you speak to about this problem, and what did they tell you?",
            "What was the hardest part to build in the time you had?",
        ],
        "unverified_claims": claims,
        "challenge_fits": fits,
    }


def mock_score(sub: Submission, criteria: list, sample: int) -> dict:
    q = latent_quality(sub)
    sents = _sentences(sub.description)
    supporting = [s for s in sents if any(w in s.lower() for w in POSITIVE)]
    level = "strong" if len(supporting) >= 2 else "weak" if supporting else "none"
    out = []
    for c in criteria:
        x = q + (_u("crit", sub.id, c.id) - 0.5) * 0.30 + (_u("noise", sub.id, c.id, sample) - 0.5) * 0.25
        score = 1 + round(4 * max(0.0, min(1.0, x)))
        if level == "none":
            score = min(score, 3)
        out.append(
            {
                "criterion_id": c.id,
                "evidence": supporting[0] if supporting else "No direct evidence in the write-up.",
                "evidence_level": level,
                "score": score,
            }
        )
    return {"criteria": out, "biggest_uncertainty": "Whether the core feature works end to end in a live demo."}


def mock_pairwise(a: Submission, b: Submission) -> dict:
    qa = latent_quality(a) + 0.05 + (_u("pair", a.id, b.id) - 0.5) * 0.2  # +0.05: a small first-position bias
    qb = latent_quality(b) + (_u("pair", b.id, a.id) - 0.5) * 0.2
    gap = abs(qa - qb)
    return {
        "reason": "Mock rule: compares simple text signals, not the projects themselves.",
        "better": "A" if qa >= qb else "B",
        "confidence": "high" if gap > 0.25 else "medium" if gap > 0.1 else "low",
    }
