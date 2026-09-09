"""Trace-level evaluator used by both the FastAPI service and batch runner."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evaluation import metrics as m


def _latency(event: Mapping[str, Any] | None) -> float | None:
    if not event:
        return None
    value = event.get("latency")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def evaluate_trace(
    *,
    trace_id: str,
    question_id: str | None,
    ground_truth: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None = None,
    agent: Mapping[str, Any] | None = None,
    validator: Mapping[str, Any] | None = None,
    document_processor: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate one completed (or partially completed) LEDGER trace."""
    retrieval_metrics = None
    if retrieval is not None:
        retrieval_metrics = m.retrieval_metrics(
            ground_truth.get("gold_doc_ids"),
            retrieval.get("evidence", []),
        )

    answer_metrics = {
        "exact_match": None,
        "f1": None,
        "scale_accuracy": None,
        "numerical_accuracy": None,
    }
    if agent is not None:
        predicted_answer = agent.get("predicted_answer")
        answer_metrics = {
            "exact_match": m.exact_match(ground_truth.get("answer"), predicted_answer),
            "f1": m.f1(ground_truth.get("answer"), predicted_answer),
            "scale_accuracy": m.scale_accuracy(
                ground_truth.get("scale"),
                agent.get("unit"),
                predicted_answer,
            ),
            "numerical_accuracy": m.numerical_accuracy(
                ground_truth.get("answer"), predicted_answer
            ),
        }

    doc_events = list(document_processor or [])
    doc_latency = sum(
        latency for event in doc_events if (latency := _latency(event)) is not None
    )
    retrieval_latency = _latency(retrieval)
    agent_latency = _latency(agent)
    validator_latency = _latency(validator)

    component_latencies = [
        value
        for value in (doc_latency, retrieval_latency, agent_latency, validator_latency)
        if value is not None
    ]
    latency_metrics = {
        "document_processor_latency": round(doc_latency, 6) if doc_events else None,
        "retrieval_latency": retrieval_latency,
        "agent_latency": agent_latency,
        "validator_latency": validator_latency,
        # Sum of the component-reported durations. This is not claimed to be
        # exact wall-clock latency if components overlap.
        "sum_component_latency": round(sum(component_latencies), 6)
        if component_latencies
        else None,
    }

    reasons: list[str] = []
    missing_components: list[str] = []
    if retrieval is None:
        missing_components.append("retrieval")
    if agent is None:
        missing_components.append("agent")
    if validator is None:
        missing_components.append("validator")

    retrieval_failure = bool(
        retrieval_metrics is not None and retrieval_metrics.get("recall_at_5", 0.0) == 0.0
    )
    if retrieval_failure:
        reasons.append("gold document not found in top-5 retrieval results")

    exact_match_failure = answer_metrics["exact_match"] == 0.0
    scale_failure = answer_metrics["scale_accuracy"] == 0.0
    answer_failure = bool(exact_match_failure or scale_failure)
    if exact_match_failure:
        reasons.append("predicted answer did not exactly match ground truth")
    if scale_failure:
        reasons.append("predicted scale/unit did not match ground truth")

    validator_rejected = bool(validator and validator.get("status") == "rejected")
    if validator_rejected:
        reason = validator.get("reason") or "validator rejected the answer"
        reasons.append(str(reason))

    doc_errors: list[str] = []
    for event in doc_events:
        for error in event.get("errors", []) or []:
            doc_errors.append(str(error))
    document_processing_error = bool(doc_errors)
    if document_processing_error:
        reasons.extend(f"document processor: {error}" for error in doc_errors)

    return {
        "trace_id": trace_id,
        "question_id": question_id,
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "latency_metrics": latency_metrics,
        "trace_failure_analysis": {
            "retrieval_failure": retrieval_failure,
            "answer_failure": answer_failure,
            "validator_rejected": validator_rejected,
            "document_processing_error": document_processing_error,
            "missing_components": missing_components,
            "reasons": reasons,
        },
    }
