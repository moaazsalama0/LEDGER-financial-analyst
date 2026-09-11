"""PDF rasterisation and text-layer probing via pypdfium2.

Adapted from thorn-nlp ``services/vision/pdf_render.py``. The reason that
module was written still applies here: pypdfium2 renders in-process, so the
service has no Poppler binary to install, which matters on the Windows
development machines this project is built on.

What changed for LEDGER: the original only rasterised. TAT-DQA documents are
overwhelmingly born-digital, so the primary path is reading the existing text
layer with exact character positions, and rasterisation is reserved for pages
that genuinely need OCR.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from app.api.errors import EncryptedPDFError, InvalidPDFError

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


@dataclass(frozen=True)
class TextPiece:
    """A run of text from the PDF's own text layer, with its position.

    Coordinates are already converted to the contract's convention: PDF
    points, top-left origin, y increasing downward.
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextChar:
    """A single glyph with its box, type size, and weight.

    The declared font size matters more than it might appear: the glyph box
    of ``(567)`` is taller than that of ``567`` because parentheses have
    deeper descenders, so sizing text by its box makes a table row look like a
    heading. Reading the real font size removes that whole class of error.

    ``font_weight`` is the font's declared weight on the usual 100-900 scale,
    or ``0`` when pdfium cannot report one. Financial notes set many of their
    sub-headings bold at body size, where size alone says nothing.
    """

    char: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_weight: int = 0

    @property
    def is_bold(self) -> bool:
        """True for semibold and heavier.

        The threshold sits at 600 rather than 700 so semibold faces, which
        reports use for run-in headings, are not read as body text.
        """
        return self.font_weight >= 600


@dataclass(frozen=True)
class PageGeometry:
    """Physical description of one page."""

    page_number: int
    width: float
    height: float
    rotation: int


class PdfDocument:
    """A borrowed handle on an open PDF.

    Used as a context manager so the underlying pdfium document is always
    closed, including when parsing raises partway through.
    """

    def __init__(self, data: bytes) -> None:
        try:
            self._doc = pdfium.PdfDocument(data, autoclose=False)
        except pdfium.PdfiumError as exc:
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise EncryptedPDFError(
                    "The PDF is password-protected and cannot be read."
                ) from exc
            raise InvalidPDFError(
                "The PDF could not be opened.", detail={"reason": str(exc)}
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidPDFError(
                "The PDF could not be opened.", detail={"reason": str(exc)}
            ) from exc

    def __enter__(self) -> "PdfDocument":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:  # pragma: no cover - closing must never raise
            pass

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def geometry(self, page_number: int) -> PageGeometry:
        """Return the physical geometry of a 1-based page number."""
        page = self._doc[page_number - 1]
        try:
            width, height = page.get_size()
            rotation = int(page.get_rotation())
            return PageGeometry(
                page_number=page_number,
                width=round(float(width), 2),
                height=round(float(height), 2),
                rotation=rotation,
            )
        finally:
            page.close()

    def text_pieces(self, page_number: int) -> list[TextPiece]:
        """Extract the page's text layer as positioned rectangles.

        pdfium reports text rectangles with a bottom-left origin, matching the
        PDF coordinate system. They are flipped here so every bbox leaving
        this service shares one convention, and a caller never has to ask
        which way up a given box is.

        Returns an empty list for a page with no text layer, which is the
        signal the scanned-page probe looks for.
        """
        page = self._doc[page_number - 1]
        try:
            _, page_height = page.get_size()
            textpage = page.get_textpage()
            try:
                pieces: list[TextPiece] = []
                for index in range(textpage.count_rects()):
                    left, bottom, right, top = textpage.get_rect(index)
                    text = textpage.get_text_bounded(left, bottom, right, top)
                    if not text or not text.strip():
                        continue
                    pieces.append(
                        TextPiece(
                            text=text,
                            x0=round(float(left), 2),
                            y0=round(float(page_height - top), 2),
                            x1=round(float(right), 2),
                            y1=round(float(page_height - bottom), 2),
                        )
                    )
                return pieces
            finally:
                textpage.close()
        finally:
            page.close()

    def text_chars(self, page_number: int) -> list[TextChar]:
        """Extract the page's text layer glyph by glyph, with font sizes.

        Slower than :meth:`text_pieces`, and worth it: the per-character font
        size and weight are the only trustworthy signals for distinguishing a
        heading from body text in a born-digital PDF, and neither is exposed
        at the rectangle level.

        Returns an empty list for a page with no text layer, which is the
        signal the scanned-page probe looks for.
        """
        page = self._doc[page_number - 1]
        try:
            _, page_height = page.get_size()
            textpage = page.get_textpage()
            try:
                count = pdfium_raw.FPDFText_CountChars(textpage.raw)
                if count <= 0:
                    return []

                left = ctypes.c_double()
                right = ctypes.c_double()
                bottom = ctypes.c_double()
                top = ctypes.c_double()

                chars: list[TextChar] = []
                for index in range(count):
                    code = pdfium_raw.FPDFText_GetUnicode(textpage.raw, index)
                    if code == 0:
                        continue
                    char = chr(code)
                    if char in "\r\n":
                        continue

                    pdfium_raw.FPDFText_GetCharBox(
                        textpage.raw, index, left, right, bottom, top
                    )
                    size = float(pdfium_raw.FPDFText_GetFontSize(textpage.raw, index))
                    # Returns -1 when the weight is unknown; normalised to 0
                    # so "not reported" never reads as "very light".
                    weight = int(pdfium_raw.FPDFText_GetFontWeight(textpage.raw, index))

                    chars.append(
                        TextChar(
                            char=char,
                            x0=round(float(left.value), 2),
                            y0=round(float(page_height - top.value), 2),
                            x1=round(float(right.value), 2),
                            y1=round(float(page_height - bottom.value), 2),
                            font_size=round(size, 2),
                            font_weight=max(weight, 0),
                        )
                    )
                return chars
            finally:
                textpage.close()
        finally:
            page.close()

    def page_text(self, page_number: int) -> str:
        """Return the plain text of a page, used by the density probe."""
        page = self._doc[page_number - 1]
        try:
            textpage = page.get_textpage()
            try:
                return textpage.get_text_range() or ""
            finally:
                textpage.close()
        finally:
            page.close()

    def render(self, page_number: int, *, dpi: int = 300) -> "Image.Image":
        """Rasterise one page to an RGB image, for pages that need OCR."""
        page = self._doc[page_number - 1]
        try:
            return page.render(scale=dpi / 72.0).to_pil().convert("RGB")
        finally:
            page.close()

    def iter_geometry(self) -> Iterator[PageGeometry]:
        for page_number in range(1, self.page_count + 1):
            yield self.geometry(page_number)


__all__ = ["PageGeometry", "PdfDocument", "TextChar", "TextPiece"]
