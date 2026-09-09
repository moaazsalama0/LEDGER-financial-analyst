"""Small Langfuse v4 wrapper used by eval-service.

The service can run without Langfuse credentials. In that case metric calculation
still works; only cloud tracing/scoring is skipped.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any, Mapping

from dotenv import load_dotenv

try:
    from langfuse import Langfuse
except ImportError:  # Allows local metric tests before optional deps are installed.
    Langfuse = None  # type: ignore[assignment]

_TRACE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


@lru_cache(maxsize=1)
def get_langfuse():
    """Return a configured Langfuse client, or ``None`` when disabled."""
    load_dotenv()

    if Langfuse is None:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    # LANGFUSE_BASE_URL is the v4 name. LANGFUSE_HOST remains as a backwards-
    # compatible fallback for the user's existing local setup.
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )
    environment = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
    )


def is_langfuse_configured() -> bool:
    return get_langfuse() is not None


def langfuse_trace_id(external_trace_id: str) -> str:
    """Convert the team's trace ID to a W3C-compatible 32-hex Langfuse trace ID.

    If the orchestrator already sends a valid 32-hex trace ID, it is preserved.
    Otherwise the mapping is deterministic, so every event with the same external
    trace ID is attached to the same Langfuse trace.
    """
    if _TRACE_ID_RE.fullmatch(external_trace_id):
        return external_trace_id.lower()

    client = get_langfuse()
    if client is not None:
        return client.create_trace_id(seed=external_trace_id)

    # Deterministic fallback used when Langfuse is disabled.
    return hashlib.sha256(external_trace_id.encode("utf-8")).hexdigest()[:32]


def record_observation(
    *,
    external_trace_id: str,
    name: str,
    as_type: str = "span",
    input_data: Any = None,
    output_data: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Record one service event as an observation in Langfuse."""
    client = get_langfuse()
    if client is None:
        return None

    trace_id = langfuse_trace_id(external_trace_id)
    meta = {"external_trace_id": external_trace_id, **dict(metadata or {})}

    with client.start_as_current_observation(
        as_type=as_type,
        name=name,
        trace_context={"trace_id": trace_id},
        input=input_data,
        output=output_data,
        metadata=meta,
    ) as observation:
        return observation.id


def attach_evaluation_scores(
    *,
    external_trace_id: str,
    evaluation: Mapping[str, Any],
) -> None:
    """Attach deterministic evaluation metrics to a Langfuse trace."""
    client = get_langfuse()
    if client is None:
        return

    trace_id = langfuse_trace_id(external_trace_id)

    metric_groups = (
        evaluation.get("retrieval_metrics") or {},
        evaluation.get("answer_metrics") or {},
    )
    for group in metric_groups:
        for name, value in group.items():
            if value is None or name.startswith("precision_at_"):
                continue
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                data_type="NUMERIC",
                metadata={"external_trace_id": external_trace_id},
            )


def flush_langfuse() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
