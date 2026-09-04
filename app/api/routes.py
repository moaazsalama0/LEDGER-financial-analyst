"""HTTP surface of doc-processor-api.

``POST /process`` is the contract; everything else exists so an orchestrator
can tell whether this service is worth sending traffic to, and so the
retrieval team can fetch the schema they are coding against.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile

from app.api.errors import BackendFailureError, DocProcessorError
from app.backends import registry
from app.cache.store import DocumentCache, cache_variant
from app.core.settings import SERVICE_VERSION, get_settings
from app.ingestion.render import PdfDocument
from app.ingestion.validate import (
    content_sha256,
    parse_options,
    resolve_document_id,
    validate_content_type,
    validate_magic_bytes,
    validate_page_count,
    validate_size,
)
from app.processing.assemble import assemble
from ledger_doc_contract.v1.models import CONTRACT_VERSION, ProcessedDocument

router = APIRouter()


@router.post(
    "/process",
    response_model=ProcessedDocument,
    response_model_exclude_none=False,
    summary="Convert a financial PDF into the structured document contract",
)
async def process(
    file: Annotated[UploadFile, File(description="The source PDF.")],
    document_id: Annotated[
        str | None,
        Form(description="Stable id. Defaults to doc_<sha256[:16]> of the file."),
    ] = None,
    options: Annotated[
        str | None,
        Form(description='JSON object: {"backend": "...", "force_ocr": false}'),
    ] = None,
) -> ProcessedDocument:
    """Parse one PDF and return its structured representation.

    Validation runs cheapest-first so a bad upload is refused before any model
    is asked to do work.
    """
    settings = get_settings()

    validate_content_type(file.content_type)
    data = await file.read()
    validate_size(data, settings)
    validate_magic_bytes(data)

    resolved_id = resolve_document_id(document_id, data)
    parsed_options = parse_options(options)
    backend_name = parsed_options.get("backend", settings.backend)
    force_ocr = bool(parsed_options.get("force_ocr", False))

    sha = content_sha256(data)

    # Page count is checked against the real document rather than a header, so
    # a small file claiming a huge page tree cannot slip through.
    with PdfDocument(data) as doc:
        validate_page_count(doc.page_count, settings)

    backend = registry.get_backend(backend_name)

    # The cache namespace covers the models the backend is configured with,
    # not just its name. Two runs of "docling" with TableFormer in accurate
    # and fast mode are different systems and must not share an entry.
    variant = cache_variant(backend.name, backend.model_versions)

    cache = DocumentCache(settings.cache_dir, enabled=settings.cache_enabled)
    # A forced-OCR run is a different parse from the cached one, so it must
    # not be served from, or written to, the same key.
    if not force_ocr:
        cached = cache.get(sha, variant)
        if cached is not None:
            return cached.model_copy(
                update={
                    "document_id": resolved_id,
                    "filename": file.filename or cached.filename,
                    "processing": cached.processing.model_copy(
                        update={"from_cache": True}
                    ),
                }
            )

    started = time.perf_counter()
    try:
        raw = backend.parse(data, force_ocr=force_ocr)
    except DocProcessorError:
        raise
    except Exception as exc:
        raise BackendFailureError(
            f"Backend {backend_name!r} failed while parsing the document.",
            document_id=resolved_id,
            detail={"backend": backend_name, "exception": type(exc).__name__},
        ) from exc
    duration_ms = int((time.perf_counter() - started) * 1000)

    document = assemble(
        raw,
        document_id=resolved_id,
        filename=file.filename or "unknown.pdf",
        content_sha256=sha,
        backend_name=backend.name,
        backend_version=backend.version,
        model_versions=backend.model_versions,
        duration_ms=duration_ms,
        settings=settings,
    )

    if not force_ocr:
        cache.put(document, variant)

    return document


@router.get("/health", summary="Liveness")
async def health() -> dict[str, str]:
    """Report that the process is running. Does not touch the backend."""
    return {"status": "ok", "service": "doc-processor-api"}


@router.get("/ready", summary="Readiness")
async def ready() -> dict[str, Any]:
    """Report whether the configured backend can serve requests.

    Distinct from ``/health`` on purpose: a service that is up but whose
    models are still loading should not be sent traffic.
    """
    settings = get_settings()
    try:
        backend = registry.get_backend(settings.backend)
    except DocProcessorError as exc:
        return {"ready": False, "backend": settings.backend, "reason": exc.message}
    return {"ready": backend.is_ready(), "backend": backend.name}


@router.get("/version", summary="Service, backend, and contract versions")
async def version() -> dict[str, Any]:
    """Everything a consumer needs to reproduce or explain an output."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "service": "doc-processor-api",
        "service_version": SERVICE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "default_backend": settings.backend,
        "available_backends": list(registry.available()),
    }
    try:
        backend = registry.get_backend(settings.backend)
        payload["backend_version"] = backend.version
        payload["model_versions"] = backend.model_versions
    except DocProcessorError as exc:
        payload["backend_version"] = None
        payload["backend_error"] = exc.message
    return payload


@router.get("/contract/schema", summary="JSON Schema of the output contract")
async def contract_schema() -> dict[str, Any]:
    """Serve the schema the retrieval service codes against.

    Published from the live models rather than a checked-in copy, so it can
    never describe a different shape than the service actually returns.
    """
    return ProcessedDocument.model_json_schema()


__all__ = ["router"]
