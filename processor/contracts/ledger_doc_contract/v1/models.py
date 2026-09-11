"""The v1 structured-document contract produced by ``doc-processor-api``.

This module is the interface between Task 2 (document processing) and Task 3
(retrieval). It depends on nothing heavier than pydantic so the retrieval
service can install it without pulling in a layout model.

Stability rules for contract version 1.x:

* fields may be ADDED, always optional with a default;
* existing fields never change type, meaning, or disappear;
* enum members may be added, so treat unknown members as opaque;
* a breaking change bumps the package to ``v2`` and both are served in
  parallel during migration.

Coordinate system: every :class:`BBox` is in PDF points (1/72 inch) with the
origin at the TOP-LEFT of the page and y increasing DOWNWARD. Page ``width``
and ``height`` use the same units, so a box can be converted to a fraction of
the page without reopening the PDF.
"""

from __future__ import annotations

from typing import Annotated, Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from ledger_doc_contract.v1.enums import (
    BlockType,
    ExtractionSource,
    NegativeStyle,
    NumericUnit,
    WarningCode,
)

CONTRACT_VERSION = "1.0"

DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,128}$"
DocumentId = Annotated[str, Field(pattern=DOCUMENT_ID_PATTERN)]


class _Frozen(BaseModel):
    """Shared config: immutable, and unknown input fields are rejected.

    Rejecting extras is deliberate. A typo in a field name should fail loudly
    during development rather than silently produce a document missing data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class BBox(_Frozen):
    """Axis-aligned box in PDF points, top-left origin, y increasing downward."""

    x0: float = Field(description="Left edge, PDF points from the page's left.")
    y0: float = Field(description="Top edge, PDF points from the page's top.")
    x1: float = Field(description="Right edge. Always >= x0.")
    y1: float = Field(description="Bottom edge. Always >= y0.")

    # Derived, and deliberately NOT serialised. Putting them on the wire would
    # mean a document could not be fed back through this model, since the
    # extra keys are rejected on validation — which would break the cache and
    # every consumer that round-trips a document.
    @property
    def width(self) -> float:
        return round(self.x1 - self.x0, 2)

    @property
    def height(self) -> float:
        return round(self.y1 - self.y0, 2)


class NumericValue(_Frozen):
    """A financial figure parsed out of a table cell.

    Purely additive: the cell's raw ``text`` is always preserved beside this,
    so a consumer that disagrees with the parse can redo it. The whole object
    is ``None`` for any cell whose text is not a number.
    """

    num: float | None = Field(
        default=None,
        description=(
            "The number as printed, sign applied, separators and currency "
            "symbols removed. None when the cell is a dash or placeholder."
        ),
    )
    scaled: float | None = Field(
        default=None,
        description=(
            "``num`` multiplied by the table's scale factor. In a table "
            "captioned 'in millions', a printed 1,234 yields 1234000000.0. "
            "Equal to ``num`` when no scale is declared."
        ),
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code inferred from the symbol, e.g. 'USD'.",
    )
    unit: NumericUnit | None = Field(default=None)
    is_negative: bool = Field(default=False)
    negative_style: NegativeStyle | None = Field(
        default=None,
        description="How the sign was written. Parentheses is the finance convention.",
    )
    is_dash: bool = Field(
        default=False,
        description=(
            "True when the cell held a dash, em-dash, or N/A. Means nil or not "
            "applicable, and is deliberately NOT coerced to 0.0 — summing a "
            "dash as zero silently changes the answer."
        ),
    )
    footnote_refs: tuple[str, ...] = Field(
        default=(),
        description="Markers stripped from the cell, e.g. ('1',) for '1,234 (1)'.",
    )


class Cell(_Frozen):
    """One cell of a recognised table."""

    cell_id: str = Field(description="Stable id, ``{table_id}_r{row}_c{col}``.")
    row: int = Field(ge=0, description="0-based row index in the reconstructed grid.")
    col: int = Field(ge=0, description="0-based column index.")
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    is_header: bool = Field(default=False)
    text: str = Field(description="Raw cell text exactly as printed. Never normalised.")
    bbox: BBox | None = Field(default=None)
    value: NumericValue | None = Field(
        default=None,
        description="Parsed figure, or None when the cell is not numeric.",
    )


class TableUnits(_Frozen):
    """Scale and currency declared for a whole table.

    Financial tables state their magnitude once, in the caption or a nearby
    line such as 'in millions, except per share data', then print bare
    numbers. Losing that multiplier turns a correct extraction into a wrong
    answer, so it is bound to the table here.
    """

    scale: float = Field(
        default=1.0, description="Multiplier, e.g. 1000000.0 for 'in millions'."
    )
    scale_label: str | None = Field(
        default=None, description="The caption phrase this was read from."
    )
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. 'USD'.")


class Table(_Frozen):
    """A table with its cell structure recovered.

    Hoisted to the document level so table search can iterate tables without
    walking every page, while the corresponding :class:`Block` on the page
    carries the same ``table_id`` for callers reading in document order.
    """

    table_id: str = Field(description="Stable id, ``t{index:03d}``, unique per document.")
    block_id: str = Field(description="The page block this table occupies.")
    page_number: int = Field(ge=1)
    section_id: str | None = Field(default=None)
    bbox: BBox | None = Field(default=None)

    caption: str | None = Field(default=None)
    caption_bbox: BBox | None = Field(default=None)

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    header_rows: int = Field(default=0, ge=0)
    header_cols: int = Field(default=0, ge=0)

    column_headers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Flattened header label per column, so a caller can resolve "
            "(row_label, column_label) to a cell without re-deriving the grid."
        ),
    )
    row_headers: tuple[str, ...] = Field(
        default=(), description="Flattened header label per body row."
    )

    units: TableUnits = Field(default_factory=TableUnits)
    cells: tuple[Cell, ...] = Field(default=())
    markdown: str = Field(
        default="",
        description="GitHub-flavoured serialisation, intended as the embeddable text.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    def cell_at(self, row: int, col: int) -> Cell | None:
        """Return the cell covering ``(row, col)``, honouring row and column spans."""
        for cell in self.cells:
            if (
                cell.row <= row < cell.row + cell.row_span
                and cell.col <= col < cell.col + cell.col_span
            ):
                return cell
        return None


class Block(_Frozen):
    """One layout region on a page — the unit a retrieval chunk is built from.

    Carries ``page_number``, ``section_id``, and ``type`` so a chunk built
    from it satisfies the brief's requirement that every chunk preserve
    document_id / page / section / content_type metadata.
    """

    block_id: str = Field(description="Stable id, ``p{page:04d}_b{index:03d}``.")
    page_number: int = Field(ge=1)
    type: BlockType
    text: str = Field(default="")
    bbox: BBox | None = Field(default=None)
    section_id: str | None = Field(
        default=None,
        description="Section this block sits in; None if it precedes any heading.",
    )
    order_index: int = Field(ge=0, description="Position within the page's reading order.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    heading_level: int | None = Field(
        default=None,
        ge=1,
        description="Set only on heading blocks. Level 1 is the outermost.",
    )
    table_id: str | None = Field(
        default=None, description="Set only when ``type`` is 'table'."
    )


class Section(_Frozen):
    """A node in the document's heading hierarchy.

    Emitted as a flat list, with the tree expressed through
    ``parent_section_id``. Flat is deliberate: it keeps the JSON easy to
    iterate and avoids nesting a chunker would only have to flatten again.
    """

    section_id: str = Field(description="Stable id, ``s{index:03d}``.")
    title: str
    level: int = Field(ge=1)
    parent_section_id: str | None = Field(default=None)
    path: tuple[str, ...] = Field(
        description=(
            "Titles from the root down to and including this section. Joining "
            "with ' > ' gives the section string an answer citation needs."
        )
    )
    heading_block_id: str | None = Field(default=None)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class Page(_Frozen):
    """One page of the source PDF."""

    page_number: int = Field(ge=1, description="1-based, matching what a reader sees.")
    width: float = Field(gt=0, description="Page width in PDF points.")
    height: float = Field(gt=0, description="Page height in PDF points.")
    rotation: int = Field(default=0, description="Page rotation in degrees: 0, 90, 180, 270.")

    extraction_source: ExtractionSource
    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="None on text-layer pages — not measured, rather than perfect.",
    )
    low_confidence: bool = Field(
        default=False,
        description="OCR confidence fell below the service's review threshold.",
    )

    blocks: tuple[Block, ...] = Field(default=())
    reading_order: tuple[str, ...] = Field(
        default=(),
        description="Block ids in reading order, with page furniture excluded.",
    )


class ProcessingWarning(_Frozen):
    """A non-fatal problem encountered while processing."""

    code: WarningCode
    message: str
    page: int | None = Field(default=None)


class ProcessingInfo(_Frozen):
    """Provenance for one processing run.

    Deliberately separate from the document body: this is the only part that
    differs between two runs over the same PDF, so a consumer hashing a
    document for change detection should exclude it.
    """

    backend: str
    backend_version: str | None = Field(default=None)
    model_versions: dict[str, str] = Field(default_factory=dict)
    service_version: str
    processed_at: str = Field(description="UTC ISO-8601 timestamp.")
    duration_ms: int = Field(ge=0)
    from_cache: bool = Field(default=False)
    warnings: tuple[ProcessingWarning, ...] = Field(default=())


class ProcessedDocument(_Frozen):
    """The complete structured representation of one financial PDF.

    Two access paths are supported on purpose:

    * walk ``pages[].blocks[]`` in ``reading_order`` for linear, section-aware
      chunking;
    * iterate ``tables`` directly for table-specific search and lookup.

    Tables appear in both, joined by ``table_id``.
    """

    contract_version: str = Field(default=CONTRACT_VERSION)
    document_id: DocumentId
    filename: str
    content_sha256: str = Field(
        description="SHA-256 of the source PDF bytes. Identifies the input exactly."
    )
    page_count: int = Field(ge=0)
    language: str | None = Field(default=None, description="BCP-47 tag, e.g. 'en'.")

    sections: tuple[Section, ...] = Field(default=())
    pages: tuple[Page, ...] = Field(default=())
    tables: tuple[Table, ...] = Field(default=())

    processing: ProcessingInfo

    # -- convenience accessors ------------------------------------------------
    # Provided so the retrieval team does not each rewrite these lookup loops.

    def page(self, page_number: int) -> Page | None:
        """Return the page with this 1-based number, or None."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def block(self, block_id: str) -> Block | None:
        """Return a block by id, searching every page."""
        for page in self.pages:
            for blk in page.blocks:
                if blk.block_id == block_id:
                    return blk
        return None

    def table(self, table_id: str) -> Table | None:
        """Return a table by id."""
        for tbl in self.tables:
            if tbl.table_id == table_id:
                return tbl
        return None

    def section(self, section_id: str) -> Section | None:
        """Return a section by id."""
        for sec in self.sections:
            if sec.section_id == section_id:
                return sec
        return None

    def section_path(self, section_id: str | None) -> str:
        """Render a section id as the ' > ' path used in answer citations.

        Returns an empty string for ``None`` so callers can use it unguarded.
        """
        if section_id is None:
            return ""
        sec = self.section(section_id)
        return " > ".join(sec.path) if sec is not None else ""

    def iter_blocks(self) -> Iterator[Block]:
        """Yield every block across all pages in reading order.

        Page furniture is skipped, matching each page's ``reading_order``.
        """
        for page in self.pages:
            by_id = {b.block_id: b for b in page.blocks}
            for block_id in page.reading_order:
                blk = by_id.get(block_id)
                if blk is not None:
                    yield blk

    def to_stable_dict(self) -> dict[str, Any]:
        """Serialise the document body without the run-specific provenance.

        Two runs over the same PDF with the same backend produce identical
        output here, which is what makes caching and golden-file tests
        meaningful.
        """
        data = self.model_dump(mode="json")
        data.pop("processing", None)
        return data


__all__ = [
    "CONTRACT_VERSION",
    "DOCUMENT_ID_PATTERN",
    "BBox",
    "Block",
    "Cell",
    "DocumentId",
    "NumericValue",
    "Page",
    "ProcessedDocument",
    "ProcessingInfo",
    "ProcessingWarning",
    "Section",
    "Table",
    "TableUnits",
]
