"""Deterministic evaluation metrics for Project LEDGER.

The eval-service intentionally keeps answer correctness and numerical correctness
separate. For example, ``38.4`` and ``38.4 million`` are not an Exact Match,
but they can still have the same numerical value.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_SCALE_ALIASES = {
    "": "",
    "none": "",
    "unitless": "",
    "thousand": "thousand",
    "thousands": "thousand",
    "k": "thousand",
    "million": "million",
    "millions": "million",
    "m": "million",
    "billion": "billion",
    "billions": "billion",
    "bn": "billion",
    "percent": "percent",
    "percentage": "percent",
    "%": "percent",
}


def to_number(value: Any) -> float | None:
    """Best-effort extraction of the first numeric value from an answer.

    Examples:
        38.4 -> 38.4
        "$38.4 million" -> 38.4
        "3,035" -> 3035.0
        "(567)" -> -567.0
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    accounting_negative = text.startswith("(") and text.endswith(")")
    match = _NUM_RE.search(text)
    if not match:
        return None

    try:
        number = float(match.group().replace(",", ""))
    except ValueError:
        return None

    if accounting_negative and number > 0:
        number = -number
    return number


def normalize_text(value: Any) -> str:
    """Normalize text for EM/F1 without changing its semantic scale/unit."""
    if value is None:
        return ""
    text = str(value).lower().strip()
    # Commas inside numbers are formatting, not token boundaries.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9.%+\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> list[str]:
    return normalize_text(value).split()


def exact_match(gold: Any, pred: Any) -> float:
    """Normalized exact match.

    This function is intentionally strict. Numeric tolerance belongs in
    :func:`numerical_accuracy`, not Exact Match.
    """
    if gold is None:
        return 1.0 if pred is None else 0.0

    if isinstance(gold, list):
        pred_list = pred if isinstance(pred, list) else ([] if pred is None else [pred])
        gold_counter = Counter(normalize_text(item) for item in gold)
        pred_counter = Counter(normalize_text(item) for item in pred_list)
        return 1.0 if gold_counter == pred_counter else 0.0

    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0


def f1(gold: Any, pred: Any) -> float:
    """Token F1 for scalar answers and item F1 for multi-span answers."""
    if gold is None:
        return 1.0 if pred is None else 0.0

    if isinstance(gold, list):
        pred_list = pred if isinstance(pred, list) else ([] if pred is None else [pred])
        gold_items = Counter(normalize_text(item) for item in gold)
        pred_items = Counter(normalize_text(item) for item in pred_list)
        common = sum((gold_items & pred_items).values())
        if not gold_items or not pred_items or common == 0:
            return 0.0
        precision = common / sum(pred_items.values())
        recall = common / sum(gold_items.values())
    else:
        gold_tokens = Counter(_tokens(gold))
        pred_tokens = Counter(_tokens(pred))
        common = sum((gold_tokens & pred_tokens).values())
        if not gold_tokens or not pred_tokens or common == 0:
            return 0.0
        precision = common / sum(pred_tokens.values())
        recall = common / sum(gold_tokens.values())

    return 2 * precision * recall / (precision + recall)


def numerical_accuracy(
    gold: Any,
    pred: Any,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
) -> float | None:
    """Return 1 when numeric answers agree within tolerance.

    Returns ``None`` when the gold answer is not numeric, so non-numeric
    questions do not lower the aggregate numerical accuracy.
    """
    gold_num = to_number(gold)
    if gold_num is None:
        return None

    pred_num = to_number(pred)
    if pred_num is None:
        return 0.0

    tolerance = max(abs_tol, rel_tol * abs(gold_num))
    return 1.0 if abs(pred_num - gold_num) <= tolerance else 0.0


def normalize_scale(value: Any) -> str:
    """Normalize common TAT-DQA/financial scale spellings."""
    if value is None:
        return ""
    text = normalize_text(value)
    if text in _SCALE_ALIASES:
        return _SCALE_ALIASES[text]
    # Preserve unknown scale labels so equality can still be tested.
    return text


def infer_scale_from_text(value: Any) -> str:
    """Infer a common scale/unit label from free-form answer text."""
    text = normalize_text(value)
    if not text:
        return ""
    if "%" in text or re.search(r"\bpercent(?:age)?\b", text):
        return "percent"
    if re.search(r"\bbillions?\b|\bbn\b", text):
        return "billion"
    if re.search(r"\bmillions?\b", text):
        return "million"
    if re.search(r"\bthousands?\b", text):
        return "thousand"
    return ""


def scale_accuracy(
    expected_scale: Any,
    predicted_scale: Any = None,
    predicted_answer: Any = None,
) -> float | None:
    """Compare the expected financial scale/unit with the prediction.

    If the reasoning service does not send an explicit scale, the evaluator
    falls back to inferring it from the answer text.
    """
    expected = normalize_scale(expected_scale)
    if not expected:
        return None

    predicted = normalize_scale(predicted_scale)
    if not predicted:
        predicted = infer_scale_from_text(predicted_answer)
    return 1.0 if expected == predicted else 0.0


def _evidence_document_ids(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return unique document IDs in retrieval order.

    Retrieval often returns several chunks from the same document. De-duplicating
    at document level prevents Recall@K from exceeding 1.0.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        doc_id = item.get("document_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(str(doc_id))
    return ordered


def retrieval_metrics(
    gold_doc_ids: Iterable[str] | None,
    evidence: Sequence[Mapping[str, Any]],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float] | None:
    """Document-level Recall@K, Precision@K and MRR.

    Returns ``None`` when no gold document is defined (for example, a genuinely
    unanswerable benchmark item).
    """
    gold = {str(doc_id) for doc_id in (gold_doc_ids or []) if doc_id}
    if not gold:
        return None

    retrieved = _evidence_document_ids(evidence)
    out: dict[str, float] = {}

    for k in ks:
        top_k = retrieved[:k]
        hits = len(gold.intersection(top_k))
        out[f"recall_at_{k}"] = hits / len(gold)
        out[f"precision_at_{k}"] = hits / len(top_k) if top_k else 0.0

    out["mrr"] = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            out["mrr"] = 1.0 / rank
            break

    return out
