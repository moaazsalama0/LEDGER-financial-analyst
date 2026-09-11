import httpx

from app.clients.base import post_with_retry
from app.config import settings
from app.schemas import AgentServiceResponse


async def call_agent_service(client: httpx.AsyncClient, question: str, trace_id: str) -> AgentServiceResponse:
    """
    Calls agent-service's POST /query.

    NOTE — field-name discrepancy between source docs:
    The Contracts PDF documents the payload key as "query":
        {"trace_id": "...", "query": "...", "top_k": 5}
    but agent-service's own posted quick-start example calls it "question":
        {"question": "...", "trace_id": "..."}

    Both keys are sent with the same value so the orchestrator works against
    either implementation without the two teams having to re-sync first.
    Confirm with the agent-service owner which one their /query handler
    actually reads, then narrow this payload if you want to tidy it up.
    """
    url = f"{settings.AGENT_SERVICE_URL}{settings.AGENT_QUERY_ENDPOINT}"
    payload = {
        "trace_id": trace_id,
        "query": question,
        "question": question,
        "top_k": settings.TOP_K,
    }

    raw = await post_with_retry(client, url, payload, service_name="agent-service", trace_id=trace_id)
    return AgentServiceResponse.model_validate(raw)
