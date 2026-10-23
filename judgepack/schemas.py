"""The JSON shapes requested from the model.

Field order is deliberate: evidence and reasons come before scores and verdicts, so the
model writes its justification first and commits to a number second.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChallengeFit(BaseModel):
    challenge_id: str = Field(description="One of the challenge ids listed in the prompt.")
    reason: str = Field(description="One sentence, grounded in the write-up.")
    fit: Literal["strong", "possible", "none"]


class Brief(BaseModel):
    one_line: str = Field(description="What the project is, in at most 20 words.")
    summary: str = Field(description="2-3 sentences: the problem, who it is for, and what was actually built.")
    how_it_works: str = Field(description="1-2 plain-English sentences on how it works.")
    strengths: list[str] = Field(description="Up to 3 specific strengths tied to the write-up.")
    questions_for_team: list[str] = Field(description="Exactly 3 questions a judge should ask the team.")
    unverified_claims: list[str] = Field(description="Claims a judge should check in the demo.")
    challenge_fits: list[ChallengeFit]


class CriterionScore(BaseModel):
    criterion_id: str
    evidence: str = Field(description="Facts from the write-up bearing on this criterion, or what is missing.")
    evidence_level: Literal["strong", "weak", "none"]
    score: int = Field(description="Integer from 1 to 5.")


class RubricScore(BaseModel):
    criteria: list[CriterionScore]
    biggest_uncertainty: str = Field(description="What a judge most needs to see in the demo.")


class PairwiseJudgement(BaseModel):
    reason: str = Field(description="One or two sentences comparing the two projects.")
    better: Literal["A", "B"]
    confidence: Literal["low", "medium", "high"]
