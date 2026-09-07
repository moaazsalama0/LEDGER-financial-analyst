import uuid

import httpx

from app.clients.agent_client import call_agent_service
from app.clients.base import DownstreamServiceError
from app.clients.validator_client import call_answer_validator
from app.logging_config import get_logger
from app.schemas import UIFinalResponse, ValidatorRejectedResponse, ValidatorSuccessResponse

logger = get_logger(__name__)


def _extract_display_answer(params: dict):
    """
    Picks the single most sensible "answer" value to surface to the UI,
    per the Strict Answer Schema's per-type params:
      - direct/calculated -> params.value
      - multi_span        -> params.values (a list)
      - insufficient_evidence -> None (reason carried separately)
    """
    if "value" in params:
        return params["value"]
    if "values" in params:
        return params["values"]
    return None


async def process_question(question: str, client: httpx.AsyncClient) -> UIFinalResponse:
    trace_id = str(uuid.uuid4())
    logger.info("[ORCHESTRATOR] received question trace_id=%s question=%r", trace_id, question)

    # 1. Ask the reasoning agent ------------------------------------------------
    try:
        agent_response = await call_agent_service(client, question, trace_id)
        answer = agent_response.normalized()
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] agent-service call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"agent-service unavailable: {exc.detail}",
            trace_id=trace_id,
        )
    except Exception as exc:  # malformed agent response, etc.
        logger.error("[ORCHESTRATOR-ERROR] could not parse agent-service response trace_id=%s error=%s", trace_id, exc)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"agent-service returned an unrecognized response shape: {exc}",
            trace_id=trace_id,
        )

    # 2. Validate the candidate answer before it ever reaches the user ---------
    try:
        validation = await call_answer_validator(client, answer, trace_id)
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] answer-validator-api call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"answer-validator-api unavailable: {exc.detail}",
            trace_id=trace_id,
        )

    # 3. Shape the final, UI-facing contract ------------------------------------
    if isinstance(validation, ValidatorSuccessResponse):
        validated_answer = validation.answer
        logger.info(
            "[ORCHESTRATOR-SUCCESS] trace_id=%s answer_type=%s evidence_count=%d",
            trace_id, validated_answer.answer_type, len(validated_answer.evidence),
        )
        return UIFinalResponse(
            status="validated",
            answer=_extract_display_answer(validated_answer.params),
            evidence=[e.as_contract_dict() for e in validated_answer.evidence],
            trace_id=trace_id,
        )

    if isinstance(validation, ValidatorRejectedResponse):
        logger.warning("[ORCHESTRATOR-REJECTED] trace_id=%s reason=%s", trace_id, validation.reason)
        return UIFinalResponse(
            status="rejected",
            answer=None,
            evidence=[],
            reason=validation.reason,
            trace_id=trace_id,
        )

    # Should be unreachable, but fail closed rather than silently succeeding.
    logger.error("[ORCHESTRATOR-ERROR] unrecognized validator response trace_id=%s", trace_id)
    return UIFinalResponse(
        status="error",
        answer=None,
        evidence=[],
        reason="answer-validator-api returned an unrecognized response shape",
        trace_id=trace_id,
    )
