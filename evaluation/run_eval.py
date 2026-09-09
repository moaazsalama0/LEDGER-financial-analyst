"""Offline batch evaluator for Project LEDGER.

This script DOES NOT run retrieval or reasoning. It only evaluates already
produced component outputs against ground truth.

Usage:
    python -m evaluation.run_eval path/to/eval_records.json
    python -m evaluation.run_eval path/to/eval_records.json --no-langfuse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.evaluator import evaluate_trace
from observability.langfuse_client import (
    attach_evaluation_scores,
    flush_langfuse,
    is_langfuse_configured,
)


def avg(values: list[float | None]) -> float:
    present = [float(v) for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else 0.0


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation file must contain a JSON array of records.")
    return payload


def _evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    ground_truth = record.get("ground_truth") or {}
    trace_id = str(record["trace_id"])
    question_id = record.get("question_id") or ground_truth.get("question_id")

    return evaluate_trace(
        trace_id=trace_id,
        question_id=question_id,
        ground_truth=ground_truth,
        retrieval=record.get("retrieval"),
        agent=record.get("agent"),
        validator=record.get("validator"),
        document_processor=record.get("document_processor") or [],
    )


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [r.get("retrieval_metrics") for r in results if r.get("retrieval_metrics")]
    answers = [r.get("answer_metrics") or {} for r in results]
    latencies = [r.get("latency_metrics") or {} for r in results]
    failures = [r.get("trace_failure_analysis") or {} for r in results]

    summary = {
        "retrieval_metrics": {
            "recall_at_1": avg([m.get("recall_at_1") for m in retrieval]),
            "recall_at_3": avg([m.get("recall_at_3") for m in retrieval]),
            "recall_at_5": avg([m.get("recall_at_5") for m in retrieval]),
            "mrr": avg([m.get("mrr") for m in retrieval]),
        },
        "answer_metrics": {
            "exact_match": avg([m.get("exact_match") for m in answers]),
            "f1": avg([m.get("f1") for m in answers]),
            "scale_accuracy": avg([m.get("scale_accuracy") for m in answers]),
            "numerical_accuracy": avg([m.get("numerical_accuracy") for m in answers]),
        },
        "latency_metrics": {
            "avg_document_processor_latency": avg(
                [m.get("document_processor_latency") for m in latencies]
            ),
            "avg_retrieval_latency": avg([m.get("retrieval_latency") for m in latencies]),
            "avg_agent_latency": avg([m.get("agent_latency") for m in latencies]),
            "avg_validator_latency": avg([m.get("validator_latency") for m in latencies]),
            "avg_sum_component_latency": avg(
                [m.get("sum_component_latency") for m in latencies]
            ),
        },
        "trace_failure_analysis": {
            "total_traces": len(results),
            "retrieval_failures": sum(bool(f.get("retrieval_failure")) for f in failures),
            "answer_failures": sum(bool(f.get("answer_failure")) for f in failures),
            "validator_rejections": sum(bool(f.get("validator_rejected")) for f in failures),
            "document_processing_errors": sum(
                bool(f.get("document_processing_error")) for f in failures
            ),
            "traces_with_missing_components": sum(
                bool(f.get("missing_components")) for f in failures
            ),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed LEDGER RAG traces.")
    parser.add_argument("input", type=Path, help="JSON file containing completed evaluation records")
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Calculate metrics without attaching scores to Langfuse",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=None,
        help="Optional path to write per-trace evaluation results as JSON",
    )
    args = parser.parse_args()

    records = _load_records(args.input)
    results: list[dict[str, Any]] = []

    print(f"Evaluating {len(records)} completed trace(s)...")
    print(f"{'question':16} {'EM':>5} {'F1':>6} {'R@5':>6} {'MRR':>6}")

    for record in records:
        result = _evaluate_record(record)
        results.append(result)

        answer = result["answer_metrics"]
        retrieval = result.get("retrieval_metrics") or {}
        question_id = result.get("question_id") or "-"
        print(
            f"{str(question_id)[:16]:16} "
            f"{(answer.get('exact_match') or 0.0):5.2f} "
            f"{(answer.get('f1') or 0.0):6.2f} "
            f"{(retrieval.get('recall_at_5') or 0.0):6.2f} "
            f"{(retrieval.get('mrr') or 0.0):6.2f}"
        )

        if not args.no_langfuse:
            attach_evaluation_scores(
                external_trace_id=result["trace_id"],
                evaluation=result,
            )

    summary = _aggregate(results)
    print("\n" + json.dumps(summary, indent=2))

    if args.details:
        args.details.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nPer-trace details written to {args.details}")

    if not args.no_langfuse:
        if is_langfuse_configured():
            flush_langfuse()
            print("\n✨ Evaluation scores sent to Langfuse.")
        else:
            print("\nℹ️ Langfuse credentials not configured; scores were not uploaded.")


if __name__ == "__main__":
    main()
