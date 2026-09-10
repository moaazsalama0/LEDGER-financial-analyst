import uuid
from datetime import datetime, timezone
from typing import Union

import httpx

from app.clients.agent_client import call_agent_service
from app.clients.base import DownstreamServiceError
from app.clients.processor_client import call_doc_processor
from app.clients.retrieval_client import call_retrieval_ingest, list_retrieval_documents
from app.clients.validator_client import call_answer_validator
from app.logging_config import get_logger
from app.telemetry import emit_validator_event
from app.schemas import (
    IndexedDocument,
    UIFinalResponse,
    UploadErrorResponse,
    UploadSuccessResponse,
    ValidatorRejectedResponse,
    ValidatorSuccessResponse,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# In-memory document registry backing GET /documents (dashboard).
#
# Deliberately NOT a database and NOT a proxy to retrieval-api — per team
# decision, this only tracks documents that were indexed through THIS
# orchestrator process's own /documents/upload flow below. It resets on
# restart, and it will NOT reflect documents ingested by a separate batch
# script that talks to doc-processor-api/retrieval-api directly without
# going through orchestrator. If that becomes a problem, the fix is either
# (a) route batch ingestion through orchestrator too, or (b) have
# retrieval-api expose its own GET /documents and proxy to it here instead.
# ---------------------------------------------------------------------------
_indexed_documents: list[IndexedDocument] = []


def get_indexed_documents() -> list[IndexedDocument]:
    return list(_indexed_documents)


async def get_dashboard_documents(client: httpx.AsyncClient) -> list[IndexedDocument]:
    """
    Merges the local registry (documents uploaded through THIS orchestrator's
    own /documents/upload — has rich metadata: filename, table_count,
    indexed_at) with retrieval-api's document listing (the full corpus as
    retrieval-api actually sees it, including anything batch-ingested
    outside orchestrator).

    Deduped by document_id. A document present in both sources keeps the
    local registry's richer entry — the local one is not overwritten by
    the thinner retrieval-api one. A document retrieval-api knows about
    but the local registry doesn't (batch-ingested corpus) is added with
    whatever fields retrieval-api provides; missing fields stay None.

    If retrieval-api's listing call fails for any reason (endpoint not
    confirmed/built yet, service down), this still returns the local
    registry alone rather than raising — see list_retrieval_documents().
    """
    merged: dict[str, IndexedDocument] = {doc.document_id: doc for doc in _indexed_documents}

    remote_docs = await list_retrieval_documents(client)
    for raw in remote_docs:
        doc_id = raw.get("document_id")
        if not doc_id or doc_id in merged:
            continue  # keep the richer local entry when both sources have it
        merged[doc_id] = IndexedDocument(
            document_id=doc_id,
            document_title=raw.get("document_title") or doc_id,
            filename=raw.get("filename") or raw.get("document_title") or doc_id,
            page_count=raw.get("page_count"),
            table_count=raw.get("table_count"),
            indexed_at=raw.get("indexed_at") or "",
        )

    return list(merged.values())


def _build_table_summary(table: dict, max_samples: int = 5) -> "TableSummary":
    """
    Converts one raw table object from doc-processor-api's output (per the
    Contracts PDF's doc-processor-api schema — column_headers, row_headers,
    cells with row/col indices) into a TableSummary for the dashboard.

    Only the first `max_samples` cells become StructuredValue previews —
    this is a dashboard summary, not the full table (the full table/cells
    already flow through to retrieval-api for search; duplicating all of
    it here would just bloat the in-memory registry for no dashboard
    benefit).
    """
    from app.schemas import StructuredValue, TableSummary  # local import avoids a circular import at module load

    column_headers = table.get("column_headers") or []
    row_headers = table.get("row_headers") or []

    samples: list[StructuredValue] = []
    for cell in (table.get("cells") or [])[:max_samples]:
        row_idx = cell.get("row")
        col_idx = cell.get("col")
        row_label = row_headers[row_idx] if isinstance(row_idx, int) and row_idx < len(row_headers) else None
        col_label = column_headers[col_idx] if isinstance(col_idx, int) and col_idx < len(column_headers) else None

        value = cell.get("text")
        cell_value = cell.get("value") or {}
        if isinstance(cell_value, dict) and "num" in cell_value:
            value = cell_value["num"]  # prefer the parsed number over the raw text when available

        samples.append(StructuredValue(row=row_label, column=col_label, value=value))

    return TableSummary(
        table_id=table.get("table_id"),
        page_number=table.get("page_number"),
        n_rows=table.get("n_rows"),
        n_cols=table.get("n_cols"),
        sample_values=samples,
    )


def _record_indexed_document(processor_doc: dict, ingest_payload: dict, filename: str) -> None:
    raw_tables = processor_doc.get("tables") or []
    _indexed_documents.append(
        IndexedDocument(
            document_id=ingest_payload["document_id"],
            document_title=ingest_payload["document_title"],
            filename=filename,
            page_count=processor_doc.get("page_count"),
            table_count=len(raw_tables),
            tables=[_build_table_summary(t) for t in raw_tables],
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
    )


def _extract_display_answer(params: dict):
    """
    Picks the single most sensible "answer" value to surface to the UI,
    per the Strict Answer Schema's per-type params:
      - direct/calculated -> params.value
      - multi_span        -> params.values (a list)
      - insufficient_evidence -> None (reason carried separately)
    """
    if "value" in params:
        return params["value"]
    if "values" in params:
        return params["values"]
    return None


async def process_question(question: str, client: httpx.AsyncClient) -> UIFinalResponse:
    trace_id = str(uuid.uuid4())
    logger.info("[ORCHESTRATOR] received question trace_id=%s question=%r", trace_id, question)

    # 1. Ask the reasoning agent ------------------------------------------------
    try:
        agent_response = await call_agent_service(client, question, trace_id)
        answer = agent_response.normalized()
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] agent-service call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"agent-service unavailable: {exc.detail}",
            trace_id=trace_id,
        )
    except Exception as exc:  # malformed agent response, etc.
        logger.error("[ORCHESTRATOR-ERROR] could not parse agent-service response trace_id=%s error=%s", trace_id, exc)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"agent-service returned an unrecognized response shape: {exc}",
            trace_id=trace_id,
        )

    # 2. Validate the candidate answer before it ever reaches the user ---------
    validator_start = datetime.now(timezone.utc)
    try:
        validation = await call_answer_validator(client, answer, trace_id)
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] answer-validator-api call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UIFinalResponse(
            status="error",
            answer=None,
            evidence=[],
            reason=f"answer-validator-api unavailable: {exc.detail}",
            trace_id=trace_id,
        )

    validator_latency_ms = (datetime.now(timezone.utc) - validator_start).total_seconds() * 1000

    # 3. Shape the final, UI-facing contract ------------------------------------
    if isinstance(validation, ValidatorSuccessResponse):
        validated_answer = validation.answer
        logger.info(
            "[ORCHESTRATOR-SUCCESS] trace_id=%s answer_type=%s evidence_count=%d",
            trace_id, validated_answer.answer_type, len(validated_answer.evidence),
        )

        await emit_validator_event(
            trace_id=trace_id,
            status="validated",
            reason=None,
            latency_ms=validator_latency_ms,
        )

        return UIFinalResponse(
            status="validated",
            answer=_extract_display_answer(validated_answer.params),
            answer_type=validated_answer.answer_type,
            params=validated_answer.params,
            evidence=[e.as_contract_dict() for e in validated_answer.evidence],
            trace_id=trace_id,
        )

    if isinstance(validation, ValidatorRejectedResponse):
        logger.warning("[ORCHESTRATOR-REJECTED] trace_id=%s reason=%s", trace_id, validation.reason)

        await emit_validator_event(
            trace_id=trace_id,
            status="rejected",
            reason=validation.reason,
            latency_ms=validator_latency_ms,
        )

        return UIFinalResponse(
            status="rejected",
            answer=None,
            evidence=[],
            reason=validation.reason,
            trace_id=trace_id,
        )

    # Should be unreachable, but fail closed rather than silently succeeding.
    logger.error("[ORCHESTRATOR-ERROR] unrecognized validator response trace_id=%s", trace_id)
    return UIFinalResponse(
        status="error",
        answer=None,
        evidence=[],
        reason="answer-validator-api returned an unrecognized response shape",
        trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# Document upload flow: UI -> Orchestrator -> doc-processor-api -> Orchestrator
#                        -> retrieval-api -> Orchestrator -> UI
#
# This does not touch process_question() or any of the agent/validator
# clients above — it is a second, independent flow through the same
# orchestrator process.
# ---------------------------------------------------------------------------
def _render_table_block(table: dict) -> str:
    """
    doc-processor-api represents a table as a separate object (with its own
    column/row headers and per-cell values) tagged with the block_id of the
    "table" block it belongs to. retrieval-api's /ingest, however, only has
    a single plain-text `content` field per block — it has no dedicated
    table-cell structure.

    To satisfy "preserve table information when available" without
    inventing a new retrieval-api field, this renders the table into a
    compact, readable text blob that gets appended to that block's content.
    """
    column_headers = table.get("column_headers") or []
    row_headers = table.get("row_headers") or []
    cells = table.get("cells") or []

    lines = []
    if column_headers:
        lines.append("columns: " + " | ".join(str(h) for h in column_headers))
    if row_headers:
        lines.append("rows: " + " | ".join(str(h) for h in row_headers))
    for cell in cells:
        lines.append(f"[r{cell.get('row')},c{cell.get('col')}]={cell.get('text')}")

    return " ; ".join(lines)


def _build_ingest_payload(processor_doc: dict, fallback_document_title: str) -> dict:
    """
    Adapts doc-processor-api's POST /process output into the shape
    retrieval-api's EXISTING POST /ingest endpoint expects.

    The two contracts were designed independently and do not line up
    field-for-field — most notably, doc-processor-api has no
    document-level title, and it lists tables in a separate top-level
    array (keyed by block_id) rather than inline on the block, the way
    retrieval-api's /ingest expects. This is the one place that
    reconciles the difference; the raw processor response is never
    forwarded to retrieval-api as-is.
    """
    document_id = processor_doc.get("document_id", "")

    sections = processor_doc.get("sections") or []
    section_title_by_id = {s.get("section_id"): s.get("title") for s in sections}

    # doc-processor-api has no document-level title field. The closest
    # available proxy is the root of a section's `path`
    # (e.g. ["Annual Report 2019", "12. Income Taxes"]); fall back to the
    # uploaded filename if there are no sections to read a path from.
    document_title = fallback_document_title
    for section in sections:
        path = section.get("path") or []
        if path:
            document_title = path[0]
            break

    # Tables are listed separately from blocks; index them by the block_id
    # of the table block they belong to so they can be attached below.
    table_by_block_id = {t.get("block_id"): t for t in (processor_doc.get("tables") or [])}

    pages_out = []
    for page in processor_doc.get("pages") or []:
        blocks_by_id = {b.get("block_id"): b for b in (page.get("blocks") or [])}
        # Prefer the processor's recovered reading order; fall back to
        # block list order if reading_order is missing for some reason.
        reading_order = page.get("reading_order") or list(blocks_by_id.keys())

        blocks_out = []
        for block_id in reading_order:
            block = blocks_by_id.get(block_id)
            if block is None:
                continue

            table = table_by_block_id.get(block_id)
            content = block.get("text") or ""
            if table is not None:
                rendered_table = _render_table_block(table)
                content = f"{content} {rendered_table}".strip() if content else rendered_table

            bbox = block.get("bbox") or {}
            bounding_box = (
                [bbox.get("x0"), bbox.get("y0"), bbox.get("x1"), bbox.get("y1")] if bbox else None
            )

            blocks_out.append(
                {
                    "section": section_title_by_id.get(block.get("section_id")),
                    "content": content,
                    "content_type": "table" if table is not None else block.get("type", "text"),
                    "bounding_box": bounding_box,
                    "table_id": table.get("table_id") if table is not None else None,
                }
            )

        pages_out.append({"page_number": page.get("page_number"), "blocks": blocks_out})

    return {
        "document_id": document_id,
        "document_title": document_title,
        "pages": pages_out,
    }


async def process_document_upload(
    filename: str,
    file_bytes: bytes,
    content_type: str | None,
    client: httpx.AsyncClient,
) -> Union[UploadSuccessResponse, UploadErrorResponse]:
    trace_id = str(uuid.uuid4())
    logger.info("[ORCHESTRATOR] received upload trace_id=%s filename=%r", trace_id, filename)

    # 1. Send the PDF to doc-processor-api's existing POST /process ------------
    try:
        processor_doc = await call_doc_processor(client, filename, file_bytes, content_type, trace_id)
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] doc-processor-api call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UploadErrorResponse(stage="processor", message=f"doc-processor-api unavailable: {exc.detail}")
    except Exception as exc:  # malformed processor response, etc.
        logger.error(
            "[ORCHESTRATOR-ERROR] could not parse doc-processor-api response trace_id=%s error=%s", trace_id, exc
        )
        return UploadErrorResponse(
            stage="processor", message=f"doc-processor-api returned an unrecognized response shape: {exc}"
        )

    # 2. Adapt the processor output into retrieval-api's /ingest shape ---------
    fallback_title = filename.rsplit(".", 1)[0] if filename else "untitled document"
    try:
        ingest_payload = _build_ingest_payload(processor_doc, fallback_title)
    except Exception as exc:
        logger.error("[ORCHESTRATOR-ERROR] failed to adapt processor output trace_id=%s error=%s", trace_id, exc)
        return UploadErrorResponse(
            stage="processor", message=f"could not adapt doc-processor-api output for indexing: {exc}"
        )

    # 3. Send the adapted payload to retrieval-api's existing POST /ingest -----
    try:
        await call_retrieval_ingest(client, ingest_payload, trace_id)
    except DownstreamServiceError as exc:
        logger.error("[ORCHESTRATOR-ERROR] retrieval-api /ingest call failed trace_id=%s detail=%s", trace_id, exc.detail)
        return UploadErrorResponse(stage="retrieval", message=f"retrieval-api unavailable: {exc.detail}")

    _record_indexed_document(processor_doc, ingest_payload, filename)

    logger.info(
        "[ORCHESTRATOR-SUCCESS] trace_id=%s document processed and indexed document_id=%s",
        trace_id, ingest_payload["document_id"],
    )
    return UploadSuccessResponse(document_id=ingest_payload["document_id"])
