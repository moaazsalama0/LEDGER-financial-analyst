from typing import Union

import httpx

from app.clients.base import post_with_retry
from app.config import settings
from app.schemas import (
    AgentAnswer,
    ValidatorRejectedResponse,
    ValidatorRequest,
    ValidatorSuccessResponse,
)


async def call_answer_validator(
    client: httpx.AsyncClient, answer: AgentAnswer, trace_id: str
) -> Union[ValidatorSuccessResponse, ValidatorRejectedResponse]:
    """
    Calls answer-validator-api's POST /validate_answer with the flattened
    payload shape from the contract:
        {"answer_type":..., "evidence":[...], "params":{...}, "trace_id":...}
    (i.e. NOT wrapped in an "answer" key — the validator's documented
    Request Contract takes these fields at the top level.)
    """
    url = f"{settings.VALIDATOR_SERVICE_URL}{settings.VALIDATOR_ENDPOINT}"

    request_model = ValidatorRequest(
        answer_type=answer.answer_type,
        evidence=[e.as_contract_dict() for e in answer.evidence],
        params=answer.params,
        trace_id=trace_id,
    )

    raw = await post_with_retry(
        client, url, request_model.model_dump(), service_name="answer-validator-api", trace_id=trace_id
    )

    if raw.get("status") == "validated":
        return ValidatorSuccessResponse.model_validate(raw)
    return ValidatorRejectedResponse.model_validate(raw)
