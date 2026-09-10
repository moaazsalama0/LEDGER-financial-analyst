"""
orchestrator_client.py
=======================
A thin, class-based wrapper around orchestrator-api's HTTP contract.

This is the ONLY module in ui-service that is allowed to know an HTTP
endpoint path. Every other module talks to an OrchestratorClient
instance and gets back plain Python objects (dicts / dataclasses) with
a "did it work" story already resolved, so the Gradio callbacks stay
about UI behaviour, not network plumbing.

Contract reference (from orchestrator-api's own app/main.py + app/schemas.py):

    GET  /health
        -> {"service": "orchestrator-api", "status": "ok",
            "dependencies": {"agent-service": "reachable"|"degraded"|"unreachable", ...}}

    POST /ask            body: {"question": "<str>"}
        -> {"status": "validated" | "rejected" | "error",
            "answer": <str|number|list|None>,
            "evidence": [{"document_id","page","page_number","section"}, ...],
            "reason": <str|None>,
            "trace_id": <str|None>}

    POST /documents/upload   multipart file
        -> {"status": "success", "document_id": "...", "message": "..."}
        or {"status": "error", "stage": "processor"|"retrieval", "message": "..."}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class AskResult:
    """Normalized result of a POST /ask call — never raises past this point."""

    ok: bool
    status: str  # "validated" | "rejected" | "error" | "network_error"
    answer: Optional[Any] = None
    evidence: list[dict] = field(default_factory=list)
    reason: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: float = 0.0
    raw_question: str = ""


@dataclass
class UploadResult:
    """Normalized result of a POST /documents/upload call."""

    ok: bool
    status: str  # "success" | "error" | "network_error"
    document_id: Optional[str] = None
    message: str = ""
    stage: Optional[str] = None
    latency_ms: float = 0.0
    filename: str = ""


@dataclass
class HealthResult:
    ok: bool
    service_status: str = "unknown"
    dependencies: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class RemoteStructuredValue:
    """One extracted cell value, as reported by orchestrator-api."""

    row: Optional[str] = None
    column: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class RemoteTableSummary:
    """One detected table, as reported by orchestrator-api's GET /documents."""

    table_id: Optional[str] = None
    document_id: str = ""  # filled in locally — not present on the raw table object itself
    page_number: Optional[int] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    sample_values: list[RemoteStructuredValue] = field(default_factory=list)


@dataclass
class RemoteDocument:
    """One document as reported by orchestrator-api's GET /documents."""

    document_id: str
    document_title: str = ""
    filename: str = ""
    page_count: Optional[int] = None
    table_count: Optional[int] = None
    indexed_at: str = ""
    tables: list[RemoteTableSummary] = field(default_factory=list)


@dataclass
class DocumentsListResult:
    """Normalized result of a GET /documents call — never raises past this point."""

    ok: bool
    count: int = 0
    documents: list[RemoteDocument] = field(default_factory=list)
    error: Optional[str] = None



class OrchestratorClient:
    """
    Synchronous HTTP client for orchestrator-api.

    A plain (sync) httpx.Client is used deliberately: Gradio already runs
    each callback in its own worker thread via its internal queue, so a
    blocking call here does not freeze the rest of the app for other
    users, and it keeps every callback in ui.py a normal, easy-to-read
    function instead of an async one.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 60.0, health_timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    # -- /health -----------------------------------------------------------
    def check_health(self) -> HealthResult:
        try:
            resp = self._client.get("/health", timeout=self.health_timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            return HealthResult(
                ok=True,
                service_status=data.get("status", "unknown"),
                dependencies=data.get("dependencies", {}),
            )
        except Exception as exc:  # orchestrator down, timeout, bad JSON, etc.
            return HealthResult(ok=False, error=str(exc))

    # -- /ask ----------------------------------------------------------------
    def ask(self, question: str) -> AskResult:
        question = (question or "").strip()
        if not question:
            return AskResult(ok=False, status="error", reason="Empty question.", raw_question=question)

        started = time.perf_counter()
        try:
            resp = self._client.post("/ask", json={"question": question})
            latency_ms = (time.perf_counter() - started) * 1000.0

            if resp.status_code >= 400:
                return AskResult(
                    ok=False,
                    status="error",
                    reason=f"orchestrator-api returned HTTP {resp.status_code}: {resp.text[:300]}",
                    latency_ms=latency_ms,
                    raw_question=question,
                )

            data = resp.json()
            status = data.get("status", "error")
            return AskResult(
                ok=(status == "validated"),
                status=status,
                answer=data.get("answer"),
                evidence=data.get("evidence") or [],
                reason=data.get("reason"),
                trace_id=data.get("trace_id"),
                latency_ms=latency_ms,
                raw_question=question,
            )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return AskResult(
                ok=False,
                status="network_error",
                reason="orchestrator-api did not respond in time (timeout).",
                latency_ms=latency_ms,
                raw_question=question,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return AskResult(
                ok=False,
                status="network_error",
                reason=f"Could not reach orchestrator-api: {exc}",
                latency_ms=latency_ms,
                raw_question=question,
            )

    # -- /documents/upload ----------------------------------------------------
    def upload_document(self, filename: str, file_bytes: bytes, content_type: str = "application/pdf") -> UploadResult:
        started = time.perf_counter()
        try:
            files = {"file": (filename, file_bytes, content_type or "application/pdf")}
            resp = self._client.post("/documents/upload", files=files)
            latency_ms = (time.perf_counter() - started) * 1000.0

            if resp.status_code >= 400:
                return UploadResult(
                    ok=False,
                    status="error",
                    message=f"orchestrator-api returned HTTP {resp.status_code}: {resp.text[:300]}",
                    latency_ms=latency_ms,
                    filename=filename,
                )

            data = resp.json()
            status = data.get("status", "error")
            return UploadResult(
                ok=(status == "success"),
                status=status,
                document_id=data.get("document_id"),
                message=data.get("message", ""),
                stage=data.get("stage"),
                latency_ms=latency_ms,
                filename=filename,
            )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return UploadResult(
                ok=False,
                status="network_error",
                message="orchestrator-api did not respond in time while processing the document.",
                latency_ms=latency_ms,
                filename=filename,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return UploadResult(
                ok=False,
                status="network_error",
                message=f"Could not reach orchestrator-api: {exc}",
                latency_ms=latency_ms,
                filename=filename,
            )


    
    # -- /documents (dashboard enrichment; may not exist yet) ----------------
    def list_documents(self) -> DocumentsListResult:
        """
        Calls orchestrator-api's GET /documents. This endpoint merges the
        orchestrator's own upload registry with retrieval-api's listing —
        so once retrieval-api ships its side, this call starts returning
        documents indexed OUTSIDE this UI session too (e.g. a batch-ingested
        corpus), with zero changes needed here.

        Fails silently (ok=False) rather than raising if the endpoint isn't
        built yet, retrieval-api is down, or the network call times out —
        callers should fall back to local session data, not crash.
        """
        try:
            resp = self._client.get("/documents", timeout=self.timeout_seconds)
            if resp.status_code >= 400:
                return DocumentsListResult(
                    ok=False, error=f"orchestrator-api returned HTTP {resp.status_code}: {resp.text[:300]}"
                )

            data = resp.json()
            raw_docs = data.get("documents") or []
            docs = []
            for d in raw_docs:
                doc_id = d.get("document_id")
                if not doc_id:
                    continue
                tables = [
                    RemoteTableSummary(
                        table_id=t.get("table_id"),
                        document_id=doc_id,
                        page_number=t.get("page_number"),
                        n_rows=t.get("n_rows"),
                        n_cols=t.get("n_cols"),
                        sample_values=[
                            RemoteStructuredValue(row=v.get("row"), column=v.get("column"), value=v.get("value"))
                            for v in (t.get("sample_values") or [])
                        ],
                    )
                    for t in (d.get("tables") or [])
                ]
                docs.append(
                    RemoteDocument(
                        document_id=doc_id,
                        document_title=d.get("document_title", ""),
                        filename=d.get("filename", ""),
                        page_count=d.get("page_count"),
                        table_count=d.get("table_count"),
                        indexed_at=d.get("indexed_at", ""),
                        tables=tables,
                    )
                )
            return DocumentsListResult(ok=True, count=len(docs), documents=docs)
        except Exception as exc:
            return DocumentsListResult(ok=False, error=str(exc))