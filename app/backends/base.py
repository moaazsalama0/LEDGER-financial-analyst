"""The seam between the API and whatever model actually reads a PDF.

The brief requires the OCR/layout backend to be replaceable without rewriting
the API layer. That is enforced structurally here: a backend returns
:class:`RawParse`, a deliberately model-agnostic description of what was found
on each page, and everything downstream — reading order, section building,
numeric parsing, contract assembly — operates only on that.

A backend's job stops at "here is what is on the page". It does not assign
reading order, build sections, parse numbers, or know the output contract
exists. Keeping those out of the backend is what allows Docling, Surya, and a
text-layer-only reader to be compared on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ledger_doc_contract.v1.enums import BlockType, ExtractionSource


@dataclass(frozen=True)
class RawBBox:
    """A box in PDF points, top-left origin. Backends must convert to this."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class RawCell:
    """One table cell as the backend recovered it."""

    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    bbox: RawBBox | None = None


@dataclass(frozen=True)
class RawTable:
    """Table structure, before numeric parsing or serialisation."""

    n_rows: int
    n_cols: int
    cells: tuple[RawCell, ...] = ()
    header_rows: int = 0
    header_cols: int = 0
    caption: str | None = None
    caption_bbox: RawBBox | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class RawBlock:
    """A region the backend found on a page.

    ``font_size`` and ``is_bold`` are optional typographic hints. The layout
    model says *that* a block is a heading; these are what let the heading
    stage work out its level, which no layout model reports.
    """

    type: BlockType
    text: str
    bbox: RawBBox | None = None
    confidence: float | None = None
    table: RawTable | None = None
    font_size: float | None = None
    is_bold: bool = False


@dataclass(frozen=True)
class RawPage:
    """Everything a backend found on one page."""

    page_number: int
    width: float
    height: float
    rotation: int = 0
    extraction_source: ExtractionSource = ExtractionSource.TEXT_LAYER
    ocr_confidence: float | None = None
    blocks: tuple[RawBlock, ...] = ()


@dataclass(frozen=True)
class RawParse:
    """A backend's complete output for one document."""

    pages: tuple[RawPage, ...] = ()
    warnings: tuple[tuple[str, str, int | None], ...] = field(default=())
    """Non-fatal problems as ``(warning_code, message, page_number)``."""


@runtime_checkable
class LayoutBackend(Protocol):
    """What every document-parsing backend must provide.

    Declared as a Protocol rather than a base class so an adapter around a
    third-party library does not have to inherit from us, and so tests can
    inject a fake with no model, no torch, and no GPU. That last property is
    what keeps continuous integration fast enough to run on every commit.
    """

    name: str

    @property
    def version(self) -> str:
        """Version of the underlying library, for output provenance."""
        ...

    @property
    def model_versions(self) -> dict[str, str]:
        """Identifiers of the loaded models, for output provenance."""
        ...

    def is_ready(self) -> bool:
        """True once the backend can serve requests.

        Backends that load models lazily report False until warmed, which is
        what ``GET /ready`` reports so an orchestrator does not send traffic
        into a cold service.
        """
        ...

    def warm(self) -> None:
        """Load any models needed. Safe to call more than once."""
        ...

    def parse(self, data: bytes, *, force_ocr: bool = False) -> RawParse:
        """Parse PDF bytes into per-page blocks.

        Must raise only on a failure of the whole run. A single page that
        cannot be read is emitted with ``ExtractionSource.FAILED`` and a
        warning, so one bad page never costs the caller the document.
        """
        ...


__all__ = [
    "LayoutBackend",
    "RawBBox",
    "RawBlock",
    "RawCell",
    "RawPage",
    "RawParse",
    "RawTable",
]
