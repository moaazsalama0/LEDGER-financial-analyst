from evaluation.metrics import (
    exact_match,
    f1,
    numerical_accuracy,
    retrieval_metrics,
    scale_accuracy,
    to_number,
)


def test_exact_match_is_separate_from_numeric_accuracy():
    assert exact_match("38.4", "38.4 million") == 0.0
    assert numerical_accuracy("38.4", "38.4 million") == 1.0


def test_numeric_tolerance():
    assert numerical_accuracy(100, 100.9) == 1.0
    assert numerical_accuracy(100, 102) == 0.0


def test_accounting_negative_number():
    assert to_number("(567)") == -567.0


def test_multispan_f1():
    assert f1(["Cost of goods", "R&D"], ["R&D", "Cost of goods"]) == 1.0


def test_scale_accuracy_explicit_and_inferred():
    assert scale_accuracy("million", "millions", None) == 1.0
    assert scale_accuracy("percent", None, "12.5%") == 1.0
    assert scale_accuracy("thousand", None, "12.5 million") == 0.0


def test_retrieval_deduplicates_documents():
    evidence = [
        {"document_id": "doc-a"},
        {"document_id": "doc-a"},
        {"document_id": "doc-b"},
    ]
    result = retrieval_metrics(["doc-a"], evidence)
    assert result is not None
    assert result["recall_at_3"] == 1.0
    assert result["mrr"] == 1.0


def test_mrr_when_gold_is_rank_three():
    evidence = [
        {"document_id": "doc-a"},
        {"document_id": "doc-b"},
        {"document_id": "doc-c"},
    ]
    result = retrieval_metrics(["doc-c"], evidence)
    assert result is not None
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_3"] == 1.0
    assert result["mrr"] == 1 / 3
