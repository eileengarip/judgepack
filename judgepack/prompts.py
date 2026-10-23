"""Prompt templates. Kept in one place so they can be read, reviewed and changed together."""

from __future__ import annotations

from .config import Config
from .ingest import Submission

SYSTEM_TEMPLATE = """You are helping the volunteer judges of {event_name}.{context}

Your job is to help judges prepare and stay consistent. You do not decide anything.

Rules:
- Use only the submission text provided. Never invent features, users, results or technologies.
- Links are listed but you cannot open them. Do not guess what they contain.
- If something is unclear or unsupported, say so plainly instead of filling the gap.
- Do not reward length, polish, jargon or ambition for their own sake. A short write-up of a small, working, well-scoped project can be better than a long one describing plans.
- Judge what the team built during the event, not what they intend to build next.
- Be specific and fair. These are students, often at their first hackathon.
- Write in plain, neutral British English.
- Reply with a single JSON object and nothing else."""


def system_prompt(cfg: Config, event: str) -> str:
    context = cfg.context_for(event)
    return SYSTEM_TEMPLATE.format(event_name=cfg.canonical_event(event), context=f" {context}" if context else "")


def render_submission(sub: Submission, max_chars: int | None = None, label: str = "SUBMISSION") -> str:
    """The project as the model sees it. Prize opt-ins and outcomes are deliberately absent."""
    description = sub.description
    if max_chars and len(description) > max_chars:
        description = description[:max_chars].rsplit(" ", 1)[0] + " […]"
    other_links = sum(
        1
        for link in sub.links
        if not any(h in link.lower() for h in ("github.com", "gitlab.com", "bitbucket.org", "youtube.com", "youtu.be"))
    )
    lines = [
        f"{label}",
        f"Title: {sub.title}",
        f"Tagline: {sub.tagline or '(none)'}",
        f"Built with: {', '.join(sub.built_with) if sub.built_with else '(not stated)'}",
        "Links (not opened): "
        f"code repository: {'yes' if sub.has_repo else 'no'}; "
        f"demo video: {'yes' if sub.has_video else 'no'}; "
        f"other links: {other_links}",
        "Write-up:",
        "<<<",
        description.strip() or "(empty)",
        ">>>",
    ]
    return "\n".join(lines)


def render_rubric(cfg: Config, event: str, with_anchors: bool = True) -> str:
    lines = ["JUDGING CRITERIA"]
    for c in cfg.criteria_for(event):
        lines.append(f"- {c.id}: {c.name} (weight {c.weight:g}). {c.description}")
        if with_anchors and c.anchors:
            lines.append("  " + "; ".join(f"{k} = {v}" for k, v in c.anchors.items()))
    return "\n".join(lines)


def render_challenges(cfg: Config, event: str) -> str:
    challenges = cfg.briefed_challenges(event)
    if not challenges:
        return "SPONSOR CHALLENGES\n(none)"
    lines = ["SPONSOR CHALLENGES"]
    for c in challenges:
        sponsor = f" (sponsor: {c.sponsor})" if c.sponsor else ""
        lines.append(f"- {c.id}: {c.name}{sponsor}. {c.brief}")
    return "\n".join(lines)


BRIEF_TASK = """Write a briefing for a judge who will spend about five minutes with this team.

Return JSON with these fields:
- one_line: what the project is, in at most 20 words.
- summary: 2-3 sentences covering the problem, who it is for, and what the team actually built.
- how_it_works: 1-2 plain-English sentences on how it works, naming key technologies only if the write-up states them.
- strengths: up to 3 specific strengths, each tied to something in the write-up. Use an empty list if none are clear.
- questions_for_team: exactly 3 questions that would help a judge check or understand the project. Prioritise the riskiest or least supported claim, what works today versus what is planned, and evidence that the problem is real for its users.
- unverified_claims: claims a judge should check in the demo, such as accuracy, performance, AI use or user numbers asserted without support. Use an empty list if there are none.
- challenge_fits: one entry for every challenge id listed above, with fit "strong", "possible" or "none" and a one-sentence reason grounded in the write-up. Judge fit from the project itself."""


BRIEF_TASK_NO_CHALLENGES = BRIEF_TASK.replace(
    "- challenge_fits: one entry for every challenge id listed above, with fit \"strong\", \"possible\" or \"none\" and a one-sentence reason grounded in the write-up. Judge fit from the project itself.",
    "- challenge_fits: an empty list (no challenge briefs are available for this event).",
)
assert BRIEF_TASK_NO_CHALLENGES != BRIEF_TASK


SCORE_TASK = """Score this submission against each criterion as an experienced, fair hackathon judge would, from the write-up alone.

For each criterion, in this order:
1. evidence: the specific facts from the write-up that bear on it, quoting short phrases where possible. If there are none, say what is missing.
2. evidence_level: "strong" if the write-up gives concrete detail, "weak" if it gives a little, "none" if it gives nothing usable.
3. score: an integer from 1 to 5 using the anchors. 3 is a typical student hackathon project; use the whole range. When evidence_level is "none", give your best estimate but no higher than 3.

Then biggest_uncertainty: the one thing a judge would most need to see in the demo to be confident in these scores.

Return JSON: {{"criteria": [one object per criterion id ({ids}), each with criterion_id, evidence, evidence_level, score], "biggest_uncertainty": "..."}}"""


PAIRWISE_TASK = """Two write-ups from the same hackathon are shown, labelled A and B. Using the criteria, decide which project experienced judges would most likely rate higher overall.

- The order of A and B is random and means nothing.
- Do not prefer a write-up for being longer or more polished. Prefer evidence of a working, well-scoped project that addresses a real problem.

Return JSON: {"reason": "one or two sentences comparing them", "better": "A" or "B", "confidence": "low", "medium" or "high"}"""


def brief_prompt(cfg: Config, sub: Submission) -> str:
    task = BRIEF_TASK if cfg.briefed_challenges(sub.event) else BRIEF_TASK_NO_CHALLENGES
    return "\n\n".join(
        [render_rubric(cfg, sub.event, with_anchors=False), render_challenges(cfg, sub.event), render_submission(sub), task]
    )


def score_prompt(cfg: Config, sub: Submission) -> str:
    ids = ", ".join(c.id for c in cfg.criteria_for(sub.event))
    return "\n\n".join([render_rubric(cfg, sub.event), render_submission(sub), SCORE_TASK.format(ids=ids)])


def pairwise_prompt(cfg: Config, a: Submission, b: Submission) -> str:
    limit = cfg.pairwise.max_chars_each
    return "\n\n".join(
        [
            render_rubric(cfg, a.event, with_anchors=False),
            render_submission(a, limit, label="PROJECT A"),
            render_submission(b, limit, label="PROJECT B"),
            PAIRWISE_TASK,
        ]
    )
