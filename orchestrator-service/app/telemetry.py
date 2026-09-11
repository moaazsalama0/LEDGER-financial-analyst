import httpx

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


async def emit_validator_event(
    trace_id: str,
    status: str,
    reason: str | None,
    latency_ms: float,
) -> None:
    payload = {
        "trace_id": trace_id,
        "status": status,
        "reason": reason,
        "latency_ms": latency_ms,
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{settings.EVAL_SERVICE_URL}/events/validator",
                json=payload,
            )
    except Exception as exc:
        logger.debug(
            "[TELEMETRY] failed to emit validator event: %s",
            exc,
        )