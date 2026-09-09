"""FastAPI application for Project LEDGER eval-service (port 8005)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from evaluation.evaluator import evaluate_trace
from observability.langfuse_client import (
    attach_evaluation_scores,
    flush_langfuse,
    is_langfuse_configured,
    record_observation,
)
from service.schemas import (
    AckResponse,
    AgentEvent,
    DocumentProcessorEvent,
    EvaluateTraceRequest,
    EvaluationResponse,
    RetrievalEvent,
    ValidatorEvent,
)
from service.store import trace_store

app = FastAPI(
    title="Project LEDGER Evaluation & Observability Service",
    version="1.0.0",
    description=(
        "Collects component telemetry, evaluates retrieval/answer quality, "
        "and attaches evaluation scores to Langfuse traces."
    ),
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "eval-service",
        "langfuse_configured": is_langfuse_configured(),
    }


@app.post("/events/retrieval", response_model=AckResponse)
def receive_retrieval(event: RetrievalEvent) -> AckResponse:
    data = event.model_dump()
    trace_store.set_retrieval(event.trace_id, data)

    observation_id = record_observation(
        external_trace_id=event.trace_id,
        name="retrieval",
        as_type="retriever",
        input_data={
            "question_id": event.question_id,
            "requested_result_count": len(event.evidence),
        },
        output_data={"evidence": [item.model_dump() for item in event.evidence]},
        metadata={"latency_seconds": event.latency},
    )
    return AckResponse(trace_id=event.trace_id, langfuse_observation_id=observation_id)


@app.post("/events/agent", response_model=AckResponse)
def receive_agent(event: AgentEvent) -> AckResponse:
    data = event.model_dump()
    trace_store.set_agent(event.trace_id, data)

    observation_id = record_observation(
        external_trace_id=event.trace_id,
        name="agent-reasoning",
        as_type="agent",
        input_data={"question_id": event.question_id},
        output_data={
            "answer_type": event.answer_type,
            "predicted_answer": event.predicted_answer,
            "unit": event.unit,
            "selected_evidence_ids": event.selected_evidence_ids,
        },
        metadata={"latency_seconds": event.latency, **event.metadata},
    )
    return AckResponse(trace_id=event.trace_id, langfuse_observation_id=observation_id)


@app.post("/events/validator", response_model=AckResponse)
def receive_validator(event: ValidatorEvent) -> AckResponse:
    data = event.model_dump()
    trace_store.set_validator(event.trace_id, data)

    observation_id = record_observation(
        external_trace_id=event.trace_id,
        name="answer-validator",
        as_type="guardrail",
        input_data={"question_id": event.question_id},
        output_data={
            "status": event.status,
            "reason": event.reason,
            "confidence": event.confidence,
        },
        metadata={"latency_seconds": event.latency},
    )
    return AckResponse(trace_id=event.trace_id, langfuse_observation_id=observation_id)


@app.post("/events/document-processor", response_model=AckResponse)
def receive_document_processor(event: DocumentProcessorEvent) -> AckResponse:
    data = event.model_dump()
    trace_store.add_document_processor(event.trace_id, data)

    observation_id = record_observation(
        external_trace_id=event.trace_id,
        name="document-processor",
        as_type="span",
        input_data={"document_id": event.document_id},
        output_data={
            "total_page_count": event.total_page_count,
            "errors": event.errors,
        },
        metadata={"latency_seconds": event.latency},
    )
    return AckResponse(trace_id=event.trace_id, langfuse_observation_id=observation_id)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluateTraceRequest) -> EvaluationResponse:
    """Evaluate a trace after the pipeline has produced its outputs.

    Ground truth belongs only to eval-service and is supplied here by the
    benchmark runner. It must never be exposed to retrieval/reasoning services.
    """
    snapshot = trace_store.get(request.trace_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for trace_id={request.trace_id!r}",
        )

    ground_truth = request.ground_truth.model_dump()
    question_id = ground_truth.get("question_id")

    evaluation = evaluate_trace(
        trace_id=request.trace_id,
        question_id=question_id,
        ground_truth=ground_truth,
        retrieval=snapshot.get("retrieval"),
        agent=snapshot.get("agent"),
        validator=snapshot.get("validator"),
        document_processor=snapshot.get("document_processor", []),
    )

    attach_evaluation_scores(
        external_trace_id=request.trace_id,
        evaluation=evaluation,
    )
    flush_langfuse()
    return EvaluationResponse.model_validate(evaluation)


@app.get("/traces/{trace_id}")
def get_trace_snapshot(trace_id: str) -> dict[str, object]:
    """Development/debug endpoint showing the events currently held in memory."""
    snapshot = trace_store.get(trace_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "events": snapshot}


@app.delete("/traces/{trace_id}")
def clear_trace(trace_id: str) -> dict[str, object]:
    """Remove completed in-memory state; Langfuse data is not deleted."""
    removed = trace_store.clear(trace_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"status": "cleared", "trace_id": trace_id}
