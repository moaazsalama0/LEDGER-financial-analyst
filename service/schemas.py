"""Pydantic contracts for the LEDGER eval-service API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

AnswerValue = str | int | float | list[str] | list[int] | list[float] | None


class ContractModel(BaseModel):
    # Ignore extra fields so upstream services may send richer payloads without
    # breaking eval-service as long as required fields are present.
    model_config = ConfigDict(extra="ignore")


class Evidence(ContractModel):
    document_id: str
    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = None
    content_type: str | None = None
    table_id: str | None = None
    document_title: str | None = None


class RetrievalEvent(ContractModel):
    trace_id: str = Field(min_length=1)
    question_id: str | None = None
    latency: float = Field(ge=0)
    evidence: list[Evidence]


class AgentEvent(ContractModel):
    trace_id: str = Field(min_length=1)
    question_id: str | None = None
    latency: float = Field(ge=0)
    predicted_answer: AnswerValue
    unit: str | None = None
    answer_type: str | None = None
    selected_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidatorEvent(ContractModel):
    trace_id: str = Field(min_length=1)
    question_id: str | None = None
    latency: float = Field(default=0.0, ge=0)
    status: Literal["validated", "rejected"]
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class DocumentProcessorEvent(ContractModel):
    trace_id: str = Field(min_length=1)
    latency: float = Field(ge=0)
    document_id: str
    total_page_count: int = Field(
        ge=0,
        validation_alias=AliasChoices("total_page_count", "page_count"),
    )
    errors: list[str] = Field(default_factory=list)


class GroundTruth(ContractModel):
    question_id: str | None = None
    answer: AnswerValue
    scale: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)


class EvaluateTraceRequest(ContractModel):
    trace_id: str = Field(min_length=1)
    ground_truth: GroundTruth


class RetrievalMetricsResponse(ContractModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    precision_at_1: float | None = None
    precision_at_3: float | None = None
    precision_at_5: float | None = None


class AnswerMetricsResponse(ContractModel):
    exact_match: float | None = None
    f1: float | None = None
    scale_accuracy: float | None = None
    numerical_accuracy: float | None = None


class LatencyMetricsResponse(ContractModel):
    document_processor_latency: float | None = None
    retrieval_latency: float | None = None
    agent_latency: float | None = None
    validator_latency: float | None = None
    sum_component_latency: float | None = None


class FailureAnalysisResponse(ContractModel):
    retrieval_failure: bool
    answer_failure: bool
    validator_rejected: bool
    document_processing_error: bool
    missing_components: list[str]
    reasons: list[str]


class EvaluationResponse(ContractModel):
    trace_id: str
    question_id: str | None = None
    retrieval_metrics: RetrievalMetricsResponse | None = None
    answer_metrics: AnswerMetricsResponse
    latency_metrics: LatencyMetricsResponse
    trace_failure_analysis: FailureAnalysisResponse


class AckResponse(ContractModel):
    status: Literal["received"] = "received"
    trace_id: str
    langfuse_observation_id: str | None = None
