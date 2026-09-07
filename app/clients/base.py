import asyncio
from typing import Any, Optional

import httpx

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class DownstreamServiceError(Exception):
    """Raised when a downstream LEDGER service is unreachable or returns a
    non-2xx / unparsable response after all retries are exhausted."""

    def __init__(self, service: str, detail: str):
        self.service = service
        self.detail = detail
        super().__init__(f"[{service}] {detail}")


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    *,
    service_name: str,
    trace_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    POSTs `payload` to `url`, retrying transient failures (timeouts,
    connection errors, 5xx) with exponential-ish backoff.

    Raises DownstreamServiceError if every attempt fails.
    Returns the parsed JSON body on success.
    """
    last_error: Optional[str] = None

    for attempt in range(1, settings.MAX_RETRIES + 2):  # e.g. MAX_RETRIES=2 -> 3 attempts total
        try:
            logger.info(
                "[ORCHESTRATOR] -> %s (attempt %d/%d) trace_id=%s url=%s",
                service_name, attempt, settings.MAX_RETRIES + 1, trace_id, url,
            )
            response = await client.post(url, json=payload, timeout=settings.REQUEST_TIMEOUT_SECONDS)

            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                logger.warning("[ORCHESTRATOR] %s returned server error: %s", service_name, last_error)
            elif response.status_code >= 400:
                # Client errors (e.g. 422 schema mismatch, 400 bad request) are not
                # retried — retrying the same malformed payload won't help.
                detail = f"HTTP {response.status_code}: {response.text[:500]}"
                logger.error("[ORCHESTRATOR] %s rejected request (no retry): %s", service_name, detail)
                raise DownstreamServiceError(service_name, detail)
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    last_error = f"non-JSON response body: {exc}"
                    logger.error("[ORCHESTRATOR] %s returned unparsable JSON: %s", service_name, last_error)

        except httpx.TimeoutException:
            last_error = f"request timed out after {settings.REQUEST_TIMEOUT_SECONDS}s"
            logger.warning("[ORCHESTRATOR] %s timeout on attempt %d", service_name, attempt)
        except httpx.ConnectError as exc:
            last_error = f"connection failed: {exc}"
            logger.warning("[ORCHESTRATOR] %s unreachable on attempt %d: %s", service_name, attempt, exc)
        except DownstreamServiceError:
            raise

        if attempt <= settings.MAX_RETRIES:
            backoff = settings.RETRY_BACKOFF_SECONDS * attempt
            await asyncio.sleep(backoff)

    raise DownstreamServiceError(service_name, last_error or "unknown failure")
