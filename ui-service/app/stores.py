"""
stores.py
=========
In-memory, class-based state for the ui-service.

The Final Project brief asks the dashboard to show "number of indexed
documents, the document list, detected tables, any extracted structured
values, and recent queries with their latency." orchestrator-api's
contract, however, only exposes /ask and /documents/upload — there is no
"list everything that has ever been indexed" endpoint anywhere in the
pipeline. So the UI's dashboard can only be a faithful mirror of what has
passed through THIS ui-service session: every document uploaded here,
and every question asked here. That is documented in the README rather
than hidden — it's the honest scope of what the UI, as specified, can
observe.

Two small classes hold that state:

    DocumentRecord / DocumentStore  -> the "Documents" tab + dashboard table
    QueryRecord    / QueryLogStore  -> the "recent queries with latency" panel

Both are plain in-process stores (a list protected by nothing fancier
than append-only usage), which is enough for a single-process Gradio
demo. They also cache each uploaded PDF's raw bytes, which is what
powers the bounding-box evidence preview in evidence_renderer.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    status: str  # "success" | "error"
    message: str
    stage: Optional[str] = None
    page_count: Optional[int] = None
    uploaded_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    latency_ms: float = 0.0


@dataclass
class QueryRecord:
    question: str
    status: str  # "validated" | "rejected" | "error" | "network_error"
    answer_preview: str
    reason: Optional[str]
    evidence_count: int
    latency_ms: float
    trace_id: Optional[str]
    asked_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class DocumentStore:
    """Tracks every document uploaded through this UI session, plus its raw bytes for previews."""

    def __init__(self) -> None:
        self._records: list[DocumentRecord] = []
        self._pdf_bytes_by_doc_id: dict[str, bytes] = {}

    def add(self, record: DocumentRecord, pdf_bytes: Optional[bytes] = None) -> None:
        self._records.append(record)
        if pdf_bytes is not None and record.document_id:
            self._pdf_bytes_by_doc_id[record.document_id] = pdf_bytes

    def all(self) -> list[DocumentRecord]:
        return list(self._records)

    def successful(self) -> list[DocumentRecord]:
        return [r for r in self._records if r.status == "success"]

    def get_pdf_bytes(self, document_id: str) -> Optional[bytes]:
        return self._pdf_bytes_by_doc_id.get(document_id)

    def document_ids(self) -> list[str]:
        return [r.document_id for r in self.successful() if r.document_id]

    def count_indexed(self) -> int:
        return len(self.successful())


class QueryLogStore:
    """Tracks every question asked through this UI session, for the dashboard's recent-queries panel."""

    def __init__(self, max_records: int = 200) -> None:
        self._records: list[QueryRecord] = []
        self._max_records = max_records

    def add(self, record: QueryRecord) -> None:
        self._records.append(record)
        # Keep the log bounded so a long demo session doesn't grow forever.
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    def all(self) -> list[QueryRecord]:
        return list(self._records)

    def recent(self, n: int = 20) -> list[QueryRecord]:
        return list(reversed(self._records[-n:]))

    def count(self) -> int:
        return len(self._records)

    def average_latency_ms(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.latency_ms for r in self._records) / len(self._records)

    def status_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for r in self._records:
            breakdown[r.status] = breakdown.get(r.status, 0) + 1
        return breakdown
