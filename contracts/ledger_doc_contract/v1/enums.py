"""Closed vocabularies used by the v1 document contract.

Every enum here is part of the public contract. Within contract version 1.x
members may be ADDED but never removed or renamed, so consumers must treat an
unrecognised member as "some other kind of block" rather than as an error.
"""

from __future__ import annotations

from enum import Enum


class BlockType(str, Enum):
    """What a layout block is, aligned with the DocLayNet label set.

    ``retrieval-api`` uses this as the ``content_type`` metadata field the
    project brief requires on every chunk.
    """

    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FIGURE = "figure"
    FORMULA = "formula"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"

    @property
    def is_heading(self) -> bool:
        """True for blocks that open a section."""
        return self in (BlockType.TITLE, BlockType.SECTION_HEADER)

    @property
    def is_page_furniture(self) -> bool:
        """True for running headers/footers that repeat across pages.

        Page furniture is excluded from ``Page.reading_order`` so it never
        leaks into a retrieval chunk, but the blocks are still emitted so a
        caller can reconstruct the full page if it needs to.
        """
        return self in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER)


class ExtractionSource(str, Enum):
    """How the text on a page was obtained.

    Kept explicit rather than inferred from a confidence sentinel: a
    text-layer page is *not measured*, which is a different thing from being
    measured at 1.0.
    """

    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    HYBRID = "hybrid"
    FAILED = "failed"


class NumericUnit(str, Enum):
    """What kind of quantity a parsed cell value represents."""

    CURRENCY = "currency"
    PERCENT = "percent"
    COUNT = "count"
    RATIO = "ratio"


class NegativeStyle(str, Enum):
    """How a negative number was written on the page.

    ``PARENTHESES`` is the one that matters: financial statements write
    ``(1,234)`` for -1234, and a parser that misses it produces an answer
    with the wrong sign.
    """

    PARENTHESES = "parentheses"
    MINUS = "minus"


class ErrorCode(str, Enum):
    """Machine-readable failure reasons returned by the service."""

    INVALID_PDF = "INVALID_PDF"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    INVALID_DOCUMENT_ID = "INVALID_DOCUMENT_ID"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    BACKEND_FAILURE = "BACKEND_FAILURE"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


class WarningCode(str, Enum):
    """Non-fatal conditions recorded in ``ProcessingInfo.warnings``."""

    SCANNED_PAGE_OCR_FALLBACK = "SCANNED_PAGE_OCR_FALLBACK"
    PAGE_EXTRACTION_FAILED = "PAGE_EXTRACTION_FAILED"
    LOW_OCR_CONFIDENCE = "LOW_OCR_CONFIDENCE"
    TABLE_STRUCTURE_UNCERTAIN = "TABLE_STRUCTURE_UNCERTAIN"
    NO_HEADINGS_DETECTED = "NO_HEADINGS_DETECTED"


__all__ = [
    "BlockType",
    "ErrorCode",
    "ExtractionSource",
    "NegativeStyle",
    "NumericUnit",
    "WarningCode",
]
