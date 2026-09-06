"""Strict Answer Schema - mirrors the answer-validator-api contract.

The agent must emit precisely one of these four shapes; the format node always
validates against these before the payload leaves the service.
"""
from typing import Literal, Union

from pydantic import BaseModel, Field


class EvidenceCitation(BaseModel):
    document_id: str
    page: int
    section: str | None = None


class DirectParams(BaseModel):
    value: str | int | float


class DirectAnswer(BaseModel):
    answer_type: Literal["direct"] = "direct"
    evidence: list[EvidenceCitation] = Field(min_length=1)
    params: DirectParams


class CalculatedParams(BaseModel):
    value: float
    formula: str


class CalculatedAnswer(BaseModel):
    answer_type: Literal["calculated"] = "calculated"
    evidence: list[EvidenceCitation] = Field(min_length=1)
    params: CalculatedParams


class MultiSpanParams(BaseModel):
    values: list[str | int | float] = Field(min_length=2)


class MultiSpanAnswer(BaseModel):
    answer_type: Literal["multi_span"] = "multi_span"
    evidence: list[EvidenceCitation] = Field(min_length=1)
    params: MultiSpanParams


class InsufficientParams(BaseModel):
    reason: str


class InsufficientEvidenceAnswer(BaseModel):
    answer_type: Literal["insufficient_evidence"] = "insufficient_evidence"
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    params: InsufficientParams


StrictAnswer = Union[
    DirectAnswer,
    CalculatedAnswer,
    MultiSpanAnswer,
    InsufficientEvidenceAnswer,
]