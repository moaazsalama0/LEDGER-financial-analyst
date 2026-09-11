import httpx

from app.clients.base import post_with_retry
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


async def call_retrieval_ingest(client: httpx.AsyncClient, ingest_payload: dict, trace_id: str) -> dict:
    """
    Calls retrieval-api's POST /ingest endpoint (document ingestion/indexing).

    The published contract shows a single document at the request root, but
    the implemented retrieval service accepts a batch envelope. Keep the
    orchestrator's internal payload as one document and adapt it here.

    This is a distinct endpoint from POST /search, which the existing
    question-answering flow (agent-service -> retrieval-api /search) already
    uses and which this change does not touch.
    """
    url = f"{settings.RETRIEVAL_SERVICE_URL}{settings.RETRIEVAL_INGEST_ENDPOINT}"
    return await post_with_retry(
        client,
        url,
        {"documents": [ingest_payload]},
        service_name="retrieval-api",
        trace_id=trace_id,
    )

async def list_retrieval_documents(client: httpx.AsyncClient) -> list[dict]:
    """
    Calls retrieval-api's document-listing endpoint for the dashboard, so
    GET /documents can show documents that were indexed WITHOUT going
    through this orchestrator's own /documents/upload (e.g. the full
    TAT-DQA corpus batch-ingested straight into retrieval-api).

    IMPORTANT — this endpoint is NOT in the official Contracts PDF
    (retrieval-api only documents POST /ingest and POST /search there).
    RETRIEVAL_DOCUMENTS_ENDPOINT below is a placeholder assumption
    ("/documents") until the retrieval-api owner confirms the real path
    and response shape. Two response shapes are accepted defensively:
        {"documents": [{"document_id":..., "document_title":...}, ...]}
        [{"document_id":..., "document_title":...}, ...]            (bare list)
    Any other shape, or any failure (endpoint doesn't exist yet, service
    down, timeout), is treated as "no extra documents available" rather
    than an error — the dashboard must still work off the local registry
    alone while this is unconfirmed. No retries: this is a best-effort
    dashboard enrichment call, not a step in a user-facing critical path.
    """
    url = f"{settings.RETRIEVAL_SERVICE_URL}{settings.RETRIEVAL_DOCUMENTS_ENDPOINT}"
    try:
        resp = await client.get(url, timeout=5.0)
        if resp.status_code >= 400:
            logger.warning(
                "[ORCHESTRATOR] retrieval-api document listing returned HTTP %d (endpoint may not exist yet) — "
                "dashboard will show local registry only for this refresh",
                resp.status_code,
            )
            return []
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "[ORCHESTRATOR] could not reach retrieval-api for document listing (%s) — "
            "dashboard will show local registry only for this refresh",
            exc,
        )
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        return data["documents"]

    logger.warning("[ORCHESTRATOR] retrieval-api document listing returned an unrecognized shape: %r", data)
    return []
