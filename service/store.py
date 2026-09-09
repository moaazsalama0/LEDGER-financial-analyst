"""Small in-memory event store for the eval-service prototype.

This keeps the API easy to integrate while the team is developing. Pending event
state is intentionally process-local and is lost on restart. Langfuse remains the
persistent observability backend.
"""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


class TraceStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._traces: dict[str, dict[str, Any]] = {}

    def _trace(self, trace_id: str) -> dict[str, Any]:
        return self._traces.setdefault(
            trace_id,
            {
                "retrieval": None,
                "agent": None,
                "validator": None,
                "document_processor": [],
            },
        )

    def set_retrieval(self, trace_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._trace(trace_id)["retrieval"] = deepcopy(event)

    def set_agent(self, trace_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._trace(trace_id)["agent"] = deepcopy(event)

    def set_validator(self, trace_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._trace(trace_id)["validator"] = deepcopy(event)

    def add_document_processor(self, trace_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._trace(trace_id)["document_processor"].append(deepcopy(event))

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._traces.get(trace_id)
            return deepcopy(value) if value is not None else None

    def clear(self, trace_id: str) -> bool:
        with self._lock:
            return self._traces.pop(trace_id, None) is not None


trace_store = TraceStore()
