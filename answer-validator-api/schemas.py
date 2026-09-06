"""
Pydantic models for answer-validator-api.

We keep the incoming request model DELIBERATELY LOOSE (answer_type: str,
evidence: list, params: dict) instead of a strict discriminated union.
Why: the spec requires very specific, hand-written rejection reasons
(e.g. "Missing required key 'formula'"), which is much easier to produce
from plain dict checks in validators.py than to coax out of Pydantic's
own error format. Pydantic here just guarantees we got valid JSON with
the right top-level shape before we hand it to the real validation logic.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    document_id: Optional[str] = None
    page_number: Optional[Any] = None  # validated for real (must be int-like) in validators.py
    section: Optional[str] = None


class AnswerRequest(BaseModel):
    """Mirrors the 'Base Structure' in the Strict Answer Schema."""
    answer_type: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class ValidatedResponse(BaseModel):
    status: str = "validated"
    answer: Dict[str, Any]


class RejectedResponse(BaseModel):
    status: str = "rejected"
    reason: str