"""Assembling a backend's raw parse into the published contract.

This is the only module that knows both the backend seam and the output
contract, which is what keeps every backend interchangeable: swapping Docling
for Surya changes what arrives here, never what leaves.

Determinism is a requirement, not an accident. Ids come from position, floats
are rounded, and nothing consults a clock or a random source. Two runs over
the same bytes with the same backend produce identical documents, which is
what makes the cache safe and the golden-file tests meaningful.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.backends.base import RawBlock, RawParse
from app.core.settings import SERVICE_VERSION, Settings
from app.processing.headings import assign_levels, build_sections, extend_section_pages
from app.processing.page_furniture import reclassify
from app.processing.reading_order import order_blocks
from app.processing.tables import build_table
from ledger_doc_contract.v1.enums import WarningCode
from ledger_doc_contract.v1.models import (
    BBox,
    Block,
    Page,
    ProcessedDocument,
    ProcessingInfo,
    ProcessingWarning,
    Table,
)


def _block_id(page_number: int, index: int) -> str:
    """Deterministic block id, matching thorn-nlp's proven scheme."""
    return f"p{page_number:04d}_b{index:03d}"


def _to_bbox(raw: object | None) -> BBox | None:
    if raw is None:
        return None
    return BBox(
        x0=round(raw.x0, 2),  # type: ignore[attr-defined]
        y0=round(raw.y0, 2),  # type: ignore[attr-defined]
        x1=round(raw.x1, 2),  # type: ignore[attr-defined]
        y1=round(raw.y1, 2),  # type: ignore[attr-defined]
    )


#: How many blocks above a table may be read for its magnitude. The Phase 2
#: corpus put most preceding declarations within three blocks and a few within
#: five; a wider window is safe because candidates are ranked nearest-first, so
#: a closer declaration always wins over a further one.
_PRECEDING_WINDOW = 5


def _preceding_texts(
    blocks: list[tuple[str, RawBlock]], table_index: int
) -> tuple[str, ...]:
    """Text of the blocks above a table, nearest first.

    Reports frequently print '(in millions)' as its own line above the table
    rather than as a caption. Order matters: the scale detector prefers the
    nearest declaration, so the block directly above the table must come
    first.
    """
    texts: list[str] = []
    for offset in range(1, _PRECEDING_WINDOW + 1):
        index = table_index - offset
        if 0 <= index < len(blocks):
            text = blocks[index][1].text.strip()
            if text:
                texts.append(text)
    return tuple(texts)


def assemble(
    parse: RawParse,
    *,
    document_id: str,
    filename: str,
    content_sha256: str,
    backend_name: str,
    backend_version: str | None,
    model_versions: dict[str, str],
    duration_ms: int,
    settings: Settings,
    from_cache: bool = False,
) -> ProcessedDocument:
    """Build the contract document from one backend run."""
    warnings: list[ProcessingWarning] = []
    for code, message, page_number in parse.warnings:
        try:
            warning_code = WarningCode(code)
        except ValueError:  # pragma: no cover - backends use known codes
            continue
        warnings.append(
            ProcessingWarning(code=warning_code, message=message, page=page_number)
        )

    # -- pass 0: demote running headers and footers --------------------------
    # A layout model classifies each page alone, so it cannot know that the
    # same line sits at the top of nine pages; it sees a short line in a
    # heading position and says "section header". Repetition is the signal no
    # single page carries, and it has to be applied before levels are assigned
    # — a spurious heading opens a section, and every block after it inherits
    # that section in its citation path.
    raw_pages = reclassify(list(parse.pages))

    # -- pass 1: identify blocks and order them ------------------------------
    # Ids are assigned in reading order so that sorting block ids sorts the
    # document, which makes the JSON readable and diffs meaningful.
    ordered_by_page: dict[int, list[tuple[str, RawBlock]]] = {}
    reading_order_by_page: dict[int, list[str]] = {}

    for raw_page in raw_pages:
        ordered = order_blocks(list(raw_page.blocks), page_width=raw_page.width)
        ordered_ids: list[tuple[str, RawBlock]] = []
        for entry in ordered:
            ordered_ids.append((_block_id(raw_page.page_number, entry.order_index), entry.block))

        # Furniture is excluded from the reading order but still emitted, so a
        # consumer can rebuild the full page while a chunker never sees it.
        furniture = [b for b in raw_page.blocks if b.type.is_page_furniture]
        start = len(ordered_ids)
        for offset, blk in enumerate(furniture):
            ordered_ids.append((_block_id(raw_page.page_number, start + offset), blk))

        ordered_by_page[raw_page.page_number] = ordered_ids
        reading_order_by_page[raw_page.page_number] = [
            block_id for block_id, blk in ordered_ids if not blk.type.is_page_furniture
        ]

    # -- pass 2: heading levels and the section tree -------------------------
    heading_inputs: list[tuple[RawBlock, int, str]] = []
    for raw_page in raw_pages:
        for block_id, blk in ordered_by_page[raw_page.page_number]:
            if blk.type.is_heading and blk.text.strip():
                heading_inputs.append((blk, raw_page.page_number, block_id))

    candidates = assign_levels(heading_inputs)
    sections = build_sections(candidates)
    section_by_heading_block = {
        c.block_id: s.section_id
        for c, s in zip(candidates, sections)
        if s.heading_block_id == c.block_id
    }

    if not sections and any(p.blocks for p in raw_pages):
        warnings.append(
            ProcessingWarning(
                code=WarningCode.NO_HEADINGS_DETECTED,
                message=(
                    "No headings were detected, so every block is reported "
                    "without a section. Retrieval citations will lack a "
                    "section label for this document."
                ),
            )
        )

    # -- pass 3: emit blocks, tracking the section each falls under ----------
    pages: list[Page] = []
    tables: list[Table] = []
    last_page_by_section: dict[str, int] = {}
    current_section: str | None = None

    for raw_page in raw_pages:
        entries = ordered_by_page[raw_page.page_number]
        blocks: list[Block] = []

        for order_index, (block_id, blk) in enumerate(entries):
            if block_id in section_by_heading_block:
                current_section = section_by_heading_block[block_id]

            section_id = current_section
            if section_id is not None:
                last_page_by_section[section_id] = max(
                    last_page_by_section.get(section_id, 0), raw_page.page_number
                )

            table_id: str | None = None
            if blk.table is not None:
                table_id = f"t{len(tables):03d}"
                tables.append(
                    build_table(
                        blk.table,
                        table_id=table_id,
                        block_id=block_id,
                        page_number=raw_page.page_number,
                        section_id=section_id,
                        bbox=_to_bbox(blk.bbox),
                        preceding_texts=_preceding_texts(entries, order_index),
                    )
                )

            heading_level = None
            if blk.type.is_heading:
                for candidate in candidates:
                    if candidate.block_id == block_id:
                        heading_level = candidate.level
                        break

            blocks.append(
                Block(
                    block_id=block_id,
                    page_number=raw_page.page_number,
                    type=blk.type,
                    text=blk.text,
                    bbox=_to_bbox(blk.bbox),
                    section_id=section_id,
                    order_index=order_index,
                    confidence=blk.confidence,
                    heading_level=heading_level,
                    table_id=table_id,
                )
            )

        low_confidence = (
            raw_page.ocr_confidence is not None
            and raw_page.ocr_confidence < settings.low_confidence_threshold
        )
        if low_confidence:
            warnings.append(
                ProcessingWarning(
                    code=WarningCode.LOW_OCR_CONFIDENCE,
                    message=(
                        f"Page {raw_page.page_number} OCR confidence "
                        f"{raw_page.ocr_confidence:.2f} is below the review "
                        f"threshold {settings.low_confidence_threshold:.2f}."
                    ),
                    page=raw_page.page_number,
                )
            )

        pages.append(
            Page(
                page_number=raw_page.page_number,
                width=raw_page.width,
                height=raw_page.height,
                rotation=raw_page.rotation,
                extraction_source=raw_page.extraction_source,
                ocr_confidence=raw_page.ocr_confidence,
                low_confidence=low_confidence,
                blocks=tuple(blocks),
                reading_order=tuple(reading_order_by_page[raw_page.page_number]),
            )
        )

    sections = extend_section_pages(sections, last_page_by_section)

    return ProcessedDocument(
        document_id=document_id,
        filename=filename,
        content_sha256=content_sha256,
        page_count=len(pages),
        language="en",
        sections=tuple(sections),
        pages=tuple(pages),
        tables=tuple(tables),
        processing=ProcessingInfo(
            backend=backend_name,
            backend_version=backend_version,
            model_versions=model_versions,
            service_version=SERVICE_VERSION,
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_ms=duration_ms,
            from_cache=from_cache,
            warnings=tuple(warnings),
        ),
    )


__all__ = ["assemble"]
