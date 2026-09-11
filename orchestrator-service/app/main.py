"""
orchestrator-api
================
Port: 8000 (per Project LEDGER contract)

The central coordinator: receives a question from the UI, routes it to
agent-service for reasoning, sends the candidate answer to
answer-validator-api for grounding/schema verification, and returns a
single validated (or clearly-explained rejected/error) response to the UI.
"""
from typing import Union

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.orchestrator import get_dashboard_documents, process_document_upload, process_question
from app.schemas import (
    DocumentsListResponse,
    UIFinalResponse,
    UIQuestionRequest,
    UploadErrorResponse,
    UploadSuccessResponse,
)

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="orchestrator-api",
    description="Central coordinator for Project LEDGER — routes questions between UI, agent-service, and answer-validator-api.",
    version="1.0.0",
)

# Permissive CORS so the Gradio ui-service (running on its own port) can call this freely during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared async HTTP client for the service's lifetime (connection pooling).
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup() -> None:
    global _http_client
    _http_client = httpx.AsyncClient()
    logger.info(
        "[ORCHESTRATOR] startup complete — agent_service=%s validator_service=%s ui_service=%s",
        settings.AGENT_SERVICE_URL, settings.VALIDATOR_SERVICE_URL, settings.UI_SERVICE_URL,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if _http_client is not None:
        await _http_client.aclose()
    logger.info("[ORCHESTRATOR] shutdown complete")


@app.get("/health")
async def health() -> dict:
    """
    Liveness check for orchestrator-api itself, plus a best-effort reachability
    probe of the other services it's configured to work with. Does not raise
    on downstream failure — this endpoint is for observability, not to gate
    the main flow.
    """
    result = {"service": "orchestrator-api", "status": "ok", "dependencies": {}}
    assert _http_client is not None

    for name, base_url in (
        ("agent-service", settings.AGENT_SERVICE_URL),
        ("answer-validator-api", settings.VALIDATOR_SERVICE_URL),
        ("ui-service", settings.UI_SERVICE_URL),
        ("retrieval-api", settings.RETRIEVAL_SERVICE_URL),
    ):
        try:
            resp = await _http_client.get(f"{base_url}/health", timeout=3.0)
            result["dependencies"][name] = "reachable" if resp.status_code < 500 else "degraded"
        except Exception:
            result["dependencies"][name] = "unreachable"

    # doc-processor-api is checked via /ready, not /health: a process that is
    # up but whose OCR/layout model hasn't finished loading will answer
    # /health fine while still failing every real /process call. /ready is
    # the endpoint that actually reflects that.
    try:
        resp = await _http_client.get(f"{settings.PROCESSOR_SERVICE_URL}/ready", timeout=3.0)
        if resp.status_code < 500 and resp.json().get("ready") is True:
            result["dependencies"]["doc-processor-api"] = "reachable"
        else:
            result["dependencies"]["doc-processor-api"] = "degraded"
    except Exception:
        result["dependencies"]["doc-processor-api"] = "unreachable"

    return result



@app.post("/ask", response_model=UIFinalResponse)
async def ask(request: UIQuestionRequest) -> UIFinalResponse:
    """
    Main entrypoint from the UI. Contract input: {"question": "..."}.
    Contract output: {"status": "...", "answer": "...", "evidence": [...]}.
    """
    if _http_client is None:
        raise HTTPException(status_code=503, detail="orchestrator-api is not ready")

    return await process_question(request.question, _http_client)


@app.post("/documents/upload", response_model=Union[UploadSuccessResponse, UploadErrorResponse])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload entrypoint from the UI. Receives a PDF, forwards it to
    doc-processor-api's existing POST /process, adapts the result into
    retrieval-api's existing POST /ingest shape, and returns a simple
    success/error status to the UI. Does not touch the /ask flow above.
    """
    if _http_client is None:
        raise HTTPException(status_code=503, detail="orchestrator-api is not ready")

    file_bytes = await file.read()
    return await process_document_upload(file.filename, file_bytes, file.content_type, _http_client)


@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents() -> DocumentsListResponse:
    """
    Dashboard endpoint. Merges documents indexed through THIS orchestrator
    process's own /documents/upload flow (in-memory, resets on restart)
    with retrieval-api's own document listing, so documents indexed
    outside orchestrator still show up.
    """
    if _http_client is None:
        raise HTTPException(status_code=503, detail="orchestrator-api is not ready")

    docs = await get_dashboard_documents(_http_client)
    return DocumentsListResponse(count=len(docs), documents=docs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
