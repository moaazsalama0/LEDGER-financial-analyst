"""
Logging helpers producing the exact console formats required by the spec:

Success:
  [ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type 'calculated'
  with evidence {'document_id': 'doc_041', 'page': 2}

Error (generic / envelope-level):
  [ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: Missing required evidence citation.

Error (type-specific):
  [ANSWER-VALIDATOR-ERROR] Invalid answer for 'calculated': Missing required key 'formula'.
"""

import logging

logger = logging.getLogger("answer-validator-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


def _first_evidence_repr(evidence: list) -> dict:
    """The spec's success example logs a single evidence dict (document_id,
    page_number), not the full list. We log the first item to match that;
    empty list -> {}. Field is 'page_number' per the agreed contract file."""
    if not evidence:
        return {}
    first = evidence[0]
    return {"document_id": first.get("document_id"), "page_number": first.get("page_number")}


def log_success(answer_type: str, evidence: list) -> None:
    logger.info(
        "[ANSWER-VALIDATOR-SUCCESS] Received and validated answer of type '%s' "
        "with evidence %s",
        answer_type,
        _first_evidence_repr(evidence),
    )


def log_error(reason: str, answer_type: str = None, generic: bool = True) -> None:
    if generic or not answer_type:
        logger.info("[ANSWER-VALIDATOR-ERROR] Invalid answer. Reason: %s", reason)
    else:
        logger.info("[ANSWER-VALIDATOR-ERROR] %s", reason)