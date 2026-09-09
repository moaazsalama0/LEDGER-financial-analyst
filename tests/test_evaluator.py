from evaluation.evaluator import evaluate_trace


def test_evaluate_trace_happy_path():
    result = evaluate_trace(
        trace_id="trace-1",
        question_id="P001",
        ground_truth={
            "answer": "38.4 million",
            "scale": "million",
            "gold_doc_ids": ["doc-1"],
        },
        retrieval={
            "latency": 0.1,
            "evidence": [
                {"document_id": "doc-1", "chunk_id": "c1", "rank": 1, "score": 0.9}
            ],
        },
        agent={
            "latency": 0.8,
            "predicted_answer": "38.4 million",
            "unit": "million",
        },
        validator={"latency": 0.01, "status": "validated"},
        document_processor=[
            {"latency": 0.2, "document_id": "doc-1", "errors": []}
        ],
    )

    assert result["retrieval_metrics"]["recall_at_1"] == 1.0
    assert result["answer_metrics"]["exact_match"] == 1.0
    assert result["answer_metrics"]["scale_accuracy"] == 1.0
    assert result["trace_failure_analysis"]["reasons"] == []
