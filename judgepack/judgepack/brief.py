"""Judge briefings: what each project is, what to ask the team, which challenges it fits."""

from __future__ import annotations

from .config import Config
from .ingest import Submission
from .llm import LLM, LLMError, QuotaExhausted
from .prompts import brief_prompt, system_prompt
from .schemas import Brief, ChallengeFit
from .util import norm, run_parallel

BRIEF_TEMPERATURE = 0.2


def make_brief_check(cfg: Config):
    def check(brief: Brief) -> None:
        if not brief.summary.strip() or not brief.one_line.strip():
            raise ValueError("summary and one_line must not be empty")
        if len([q for q in brief.questions_for_team if q.strip()]) < 2:
            raise ValueError("questions_for_team needs 3 questions")

    return check


def normalise_brief(brief: Brief, cfg: Config, event: str) -> Brief:
    """Trim lists to their stated sizes and make sure every briefed challenge appears exactly once."""
    challenges = cfg.briefed_challenges(event)
    known = {norm(c.id): c.id for c in challenges}
    known.update({norm(c.name): c.id for c in challenges})
    fits: dict[str, ChallengeFit] = {}
    for f in brief.challenge_fits:
        cid = known.get(norm(f.challenge_id))
        if cid and cid not in fits:
            fits[cid] = ChallengeFit(challenge_id=cid, reason=f.reason.strip(), fit=f.fit)
    for c in challenges:
        fits.setdefault(c.id, ChallengeFit(challenge_id=c.id, reason="Not assessed.", fit="none"))
    return Brief(
        one_line=brief.one_line.strip(),
        summary=brief.summary.strip(),
        how_it_works=brief.how_it_works.strip(),
        strengths=[s.strip() for s in brief.strengths if s.strip()][:3],
        questions_for_team=[q.strip() for q in brief.questions_for_team if q.strip()][:3],
        unverified_claims=[c.strip() for c in brief.unverified_claims if c.strip()][:4],
        challenge_fits=[fits[c.id] for c in challenges],
    )


def generate_briefs(subs: list[Submission], cfg: Config, llm: LLM, workers: int) -> tuple[dict[str, Brief], list[str]]:
    check = make_brief_check(cfg)

    def one(sub: Submission) -> Brief:
        brief = llm.run(
            task="brief",
            system=system_prompt(cfg, sub.event),
            prompt=brief_prompt(cfg, sub),
            schema=Brief,
            temperature=BRIEF_TEMPERATURE,
            context={"submission": sub, "config": cfg},
            check=check,
        )
        return normalise_brief(brief, cfg, sub.event)

    results, errors = run_parallel(one, subs, workers, "briefs", catch=(LLMError,), stop=(QuotaExhausted,))
    return {s.id: b for s, b in zip(subs, results) if b is not None}, errors
