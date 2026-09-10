"""
Pydantic models mirroring the "Project LEDGER Contracts & JSON Reference"
and the "Strict Answer Schema" from the Final Project brief.

A couple of the source documents disagree with each other on field names
(see README "Known Contract Discrepancies"). These models are intentionally
lenient on *input* (accept either spelling) and strict/normalized on
*output*, so the orchestrator doesn't become the thing that breaks the
pipeline over a naming mismatch.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

AnswerType = Literal["direct", "calculated", "multi_span", "insufficient_evidence"]


# ---------------------------------------------------------------------------
# Inbound: UI -> Orchestrator
# ---------------------------------------------------------------------------
class UIQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question from the user")


# ---------------------------------------------------------------------------
# Evidence — normalized to carry BOTH `page` and `page_number`.
#
# doc-processor-api / retrieval-api / agent-service (per the Contracts doc)
# use "page_number"; the Strict Answer Schema in the Final Project brief
# uses "page". Rather than guess which one any given teammate's service
# actually implements, every evidence object the orchestrator forwards
# carries both keys with the same value.
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    document_id: str
    page: Optional[int] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    content_type: Optional[str] = None
    snippet: Optional[str] = None

    @model_validator(mode="after")
    def _mirror_page_fields(self) -> "Evidence":
        if self.page is None and self.page_number is not None:
            self.page = self.page_number
        if self.page_number is None and self.page is not None:
            self.page_number = self.page
        return self

    def as_contract_dict(self) -> dict:
        """Emit both page keys so downstream services can read whichever they expect."""
        d = {
            "document_id": self.document_id,
            "page": self.page,
            "page_number": self.page_number,
        }

        if self.section is not None:
            d["section"] = self.section

        if self.content_type is not None:
            d["content_type"] = self.content_type

        if self.snippet is not None:
            d["snippet"] = self.snippet

        return d

# ---------------------------------------------------------------------------
# Agent-service response (Orchestrator <- agent-service)
# ---------------------------------------------------------------------------
class AgentAnswer(BaseModel):
    answer_type: AnswerType
    evidence: List[Evidence] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


class AgentServiceResponse(BaseModel):
    """
    Accepts either the contract's documented shape:
        {"answer": {"answer_type":..., "evidence":[...], "params":{...}}}
    or a flat shape (some implementations emit the answer fields directly
    at the top level without the "answer" wrapper):
        {"answer_type":..., "evidence":[...], "params":{...}}
    """
    answer: Optional[AgentAnswer] = None
    answer_type: Optional[AnswerType] = None
    evidence: Optional[List[Evidence]] = None
    params: Optional[dict] = None

    def normalized(self) -> AgentAnswer:
        if self.answer is not None:
            return self.answer
        if self.answer_type is not None:
            return AgentAnswer(
                answer_type=self.answer_type,
                evidence=self.evidence or [],
                params=self.params or {},
            )
        raise ValueError("agent-service response did not contain a recognizable answer payload")


# ---------------------------------------------------------------------------
# Outbound: Orchestrator -> Answer Validator
# ---------------------------------------------------------------------------
class ValidatorRequest(BaseModel):
    answer_type: AnswerType
    evidence: List[dict]
    params: dict
    trace_id: str


# ---------------------------------------------------------------------------
# Inbound: Answer Validator -> Orchestrator
# ---------------------------------------------------------------------------
class ValidatorSuccessResponse(BaseModel):
    status: Literal["validated"]
    answer: AgentAnswer


class ValidatorRejectedResponse(BaseModel):
    status: Literal["rejected"]
    reason: str


# ---------------------------------------------------------------------------
# Outbound: Orchestrator -> UI  (final contract shape)
# ---------------------------------------------------------------------------
class UIFinalResponse(BaseModel):
    status: str
    answer: Optional[Union[str, float, int, List[Any]]] = None
    # Extension beyond the base contract: the documented "To User Interface"
    # shape only has status/answer/evidence, but the UI benefits from
    # knowing which of the 4 LEDGER answer types produced this answer
    # (e.g. to render calculated vs multi_span differently).
    answer_type: Optional[AnswerType] = None
    params: Optional[dict] = None
    evidence: List[dict] = Field(default_factory=list)
    # Extension beyond the base contract: surfaced only on non-"validated"
    # outcomes so the UI can show *why* (rejected / upstream error / timeout).
    reason: Optional[str] = None
    trace_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Outbound: Orchestrator -> UI  (POST /documents/upload contract shape)
#
# doc-processor-api's and retrieval-api's own response bodies are internal
# to the upload flow (consumed and adapted inside orchestrator.py) — the UI
# only ever sees one of these two simple shapes.
# ---------------------------------------------------------------------------
class UploadSuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    document_id: str
    message: str = "Document processed and indexed successfully"


class UploadErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    stage: Literal["processor", "retrieval"]
    message: str
# ---------------------------------------------------------------------------
# Outbound: Orchestrator -> UI  (GET /documents contract shape)
#
# NOT part of the official Contracts PDF — this is an orchestrator-only
# addition for the Gradio dashboard.
# ---------------------------------------------------------------------------
class StructuredValue(BaseModel):
    """One extracted cell value — the Final Project brief's "any extracted
    structured values" dashboard requirement."""
    row: Optional[str] = None
    column: Optional[str] = None
    value: Optional[Union[str, float, int]] = None


class TableSummary(BaseModel):
    """One detected table — the Final Project brief's "detected tables"
    dashboard requirement."""
    table_id: Optional[str] = None
    page_number: Optional[int] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    sample_values: List[StructuredValue] = Field(default_factory=list)


class IndexedDocument(BaseModel):
    document_id: str
    document_title: str
    filename: str
    page_count: Optional[int] = None
    table_count: Optional[int] = None
    tables: List[TableSummary] = Field(default_factory=list)
    indexed_at: str  # ISO-8601 timestamp, UTC


class DocumentsListResponse(BaseModel):
    count: int
    documents: List[IndexedDocument]