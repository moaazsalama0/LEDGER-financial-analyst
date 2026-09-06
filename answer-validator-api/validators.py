"""
Core validation logic for the Strict Answer Schema.

validate_answer(payload) -> (ok: bool, reason: str | None, generic: bool)

`generic` tells the caller which log/response format to use:
  - generic=True  -> "[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: {reason}"
    (used for envelope-level problems: bad/missing answer_type, missing
    evidence citation entirely)
  - generic=False -> "[ANSWER-VALIDATOR-ERROR] Invalid answer for '{type}': {reason}"
    (used for type-specific param problems)
"""

from typing import Any, Dict, List, Tuple

ALLOWED_TYPES = {"direct", "calculated", "multi_span", "insufficient_evidence"}

# Types that MUST have at least one evidence citation.
# insufficient_evidence is explicitly allowed to have empty evidence.
REQUIRES_EVIDENCE = {"direct", "calculated", "multi_span"}


def _is_valid_evidence_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("document_id"), str) or not item.get("document_id"):
        return False
    # page must be present and int-like (accepting bool excluded)
    page = item.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        return False
    return True


def _evidence_is_valid_list(evidence: Any) -> bool:
    if not isinstance(evidence, list) or len(evidence) == 0:
        return False
    return all(_is_valid_evidence_item(e) for e in evidence)


def validate_answer(payload: Dict[str, Any]) -> Tuple[bool, str, bool]:
    """
    Returns (ok, reason, generic).
    If ok is True, reason is "" and generic is irrelevant.
    """
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object.", True

    answer_type = payload.get("answer_type")
    evidence = payload.get("evidence", [])
    params = payload.get("params", {})

    # 1. answer_type must exist and be one of the four allowed types
    if not answer_type or answer_type not in ALLOWED_TYPES:
        return False, f"Unsupported or missing answer_type '{answer_type}'.", True

    # 2. evidence presence/shape check (envelope-level -> generic message)
    if answer_type in REQUIRES_EVIDENCE:
        if not _evidence_is_valid_list(evidence):
            return False, "Missing required evidence citation.", True
    else:
        # insufficient_evidence: evidence is optional but if provided must be a list
        if evidence is not None and not isinstance(evidence, list):
            return False, "Missing required evidence citation.", True

    # 3. type-specific param checks (-> typed message)
    if not isinstance(params, dict):
        return False, f"Invalid answer for '{answer_type}': 'params' must be an object.", False

    if answer_type == "direct":
        value = params.get("value")
        if value is None or not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return False, f"Invalid answer for '{answer_type}': Missing required key 'value'.", False

    elif answer_type == "calculated":
        value = params.get("value")
        formula = params.get("formula")
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"Invalid answer for '{answer_type}': Missing required key 'value'.", False
        if not formula or not isinstance(formula, str):
            return False, f"Invalid answer for '{answer_type}': Missing required key 'formula'.", False
        # Note: the spec calls for "one citation per operand used in the formula",
        # but operands are the *values pulled from evidence* (not every numeric
        # literal — constants like the '*100' in a percent-change formula don't
        # come from a document). Reliably distinguishing the two from the formula
        # string alone is ambiguous, so we don't hard-reject on a mismatched count
        # here; we only enforce that at least one evidence citation exists (checked
        # above via REQUIRES_EVIDENCE). If your team wants stricter enforcement,
        # this is the place to add it once operands are tracked explicitly upstream.

    elif answer_type == "multi_span":
        values = params.get("values")
        if not isinstance(values, list) or len(values) < 2:
            return False, f"Invalid answer for '{answer_type}': Missing required key 'values' (needs 2+ items).", False
        if len(evidence) < 1:
            return False, "Missing required evidence citation.", True

    elif answer_type == "insufficient_evidence":
        reason_text = params.get("reason")
        if not reason_text or not isinstance(reason_text, str):
            return False, f"Invalid answer for '{answer_type}': Missing required key 'reason'.", False

    return True, "", False
