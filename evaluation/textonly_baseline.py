"""The Phase 2 text-layer baseline. Evaluation only — not a production backend.

This is the no-model reader every Phase 2 number was measured *against*: it
groups the PDF's own glyphs into lines and blocks geometrically and infers
headings from font size. It recovers no table structure at all, which is
precisely what made it the right control — 77.7% of everything it lost, it
lost for that one reason, and that is the measured case for Docling.

It lives here, in ``evaluation``, and **not** in ``app.backends`` because it is
no longer a backend this service offers. Docling is the only production path;
``app.backends.registry`` does not know this class exists, so ``POST /process``
with ``{"backend": "textonly"}`` is refused with ``BACKEND_UNAVAILABLE`` like
any other name the service does not serve.

It is kept, rather than deleted, for two jobs that are still real:

* **reproducing Phase 2.** ``baselines/PHASE_2_BASELINE/dev-textonly.jsonl`` is
  a frozen record, and a record nobody can regenerate is an assertion. Call
  :func:`register` and ``scripts/run_extraction_eval.py --backend textonly``
  reruns the historical arm exactly as it ran;
* **driving the fast test suite.** It parses a real PDF into a real
  :class:`~app.backends.base.RawParse` on CPU in milliseconds with no model, no
  GPU, and no network, which is what lets the whole service — validation,
  assembly, contract, HTTP, cache — be exercised on every commit.

``name`` stays ``"textonly"`` so a rerun's provenance and cache variant match
the frozen Phase 2 records byte for byte.

Line and block assembly works from characters rather than pdfium's text
rectangles so that the real font size is available. Sizing text by its glyph
box instead would read a table row containing ``(567)`` as a heading, because
parentheses are taller than digits.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.backends.base import LayoutBackend, RawBBox, RawBlock, RawPage, RawParse
from app.ingestion.render import PdfDocument, TextChar
from ledger_doc_contract.v1.enums import BlockType, ExtractionSource

#: The name this baseline was recorded under in the Phase 2 results and in
#: the document cache. Kept unchanged so a rerun lines up with the frozen
#: records rather than producing a second, differently-named history.
BASELINE_BACKEND = "textonly"

#: Glyphs whose vertical centres sit within this fraction of their font size
#: are treated as sharing a baseline.
_LINE_TOLERANCE_RATIO = 0.4

#: Fallback word-break threshold, as a multiple of the font size, used only
#: where the text layer supplies no space character. Set well above a real
#: space's width because narrow glyphs leave deceptively wide ink gaps: the
#: advance of "1" is nearly twice its ink width. Column gaps in tables run to
#: several ems, so this still separates them.
_WORD_GAP_RATIO = 0.45

#: A gap wider than this multiple of the line height starts a new paragraph.
_PARAGRAPH_GAP_RATIO = 0.75

#: A line must exceed the page's body font size by this ratio to be read as a
#: heading. Deliberately conservative: a false heading corrupts the section
#: tree for every block beneath it, which is worse than a missed one.
_HEADING_SIZE_RATIO = 1.15

#: Fraction of page height within which a short line counts as furniture.
_FURNITURE_BAND = 0.07

#: Furniture must be no longer than this; a full sentence low on the page is
#: body text, not a running footer.
_FURNITURE_MAX_CHARS = 60


@dataclass
class _Line:
    """Glyphs sharing a baseline, in reading order."""

    chars: list[TextChar]

    @property
    def text(self) -> str:
        """Reassemble the line in reading order.

        Spaces already present in the text layer are kept rather than dropped
        and re-derived from glyph gaps. Re-deriving them is what splits
        "2019" into "201 9": a narrow glyph like ``1`` has a much smaller ink
        box than its advance width, so the gap after it looks like a word
        break. A synthetic space is added only where the text layer supplies
        none and the gap is genuinely wide.
        """
        ordered = sorted(self.chars, key=lambda c: c.x0)
        out: list[str] = []
        previous: TextChar | None = None
        for char in ordered:
            if previous is not None and char.char != " " and previous.char != " ":
                gap = char.x0 - previous.x1
                if gap > max(previous.font_size, 1.0) * _WORD_GAP_RATIO:
                    out.append(" ")
            out.append(char.char)
            previous = char
        return "".join(out).strip()

    @property
    def _visible(self) -> list[TextChar]:
        """Inked glyphs only.

        A space's box is zero-height and sits on the baseline, so including
        spaces would drag a line's bounding box down to the baseline and
        distort every geometric comparison built on it.
        """
        return [c for c in self.chars if c.char.strip()] or self.chars

    @property
    def font_size(self) -> float:
        """The line's dominant font size.

        The mode rather than the mean, so a single drop-cap or a superscript
        footnote marker cannot shift a body line into heading territory.
        """
        sizes = [c.font_size for c in self._visible]
        if not sizes:
            return 0.0
        counts: dict[float, int] = {}
        for size in sizes:
            counts[size] = counts.get(size, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def x0(self) -> float:
        return min(c.x0 for c in self._visible)

    @property
    def y0(self) -> float:
        return min(c.y0 for c in self._visible)

    @property
    def x1(self) -> float:
        return max(c.x1 for c in self._visible)

    @property
    def y1(self) -> float:
        return max(c.y1 for c in self._visible)


def _group_into_lines(chars: list[TextChar]) -> list[_Line]:
    """Cluster glyphs into lines by vertical centre.

    Spaces are carried along rather than discarded, so a line can reassemble
    its own words from the text layer instead of guessing at them.
    Clustering is driven by inked glyphs, since a space's degenerate box
    would otherwise decide which line it joins.
    """
    inked = [c for c in chars if c.char.strip()]
    if not inked:
        return []

    ordered = sorted(inked, key=lambda c: ((c.y0 + c.y1) / 2, c.x0))
    groups: list[list[TextChar]] = [[ordered[0]]]
    centre = (ordered[0].y0 + ordered[0].y1) / 2

    for char in ordered[1:]:
        char_centre = (char.y0 + char.y1) / 2
        tolerance = max(char.font_size, 1.0) * _LINE_TOLERANCE_RATIO
        if abs(char_centre - centre) <= tolerance:
            groups[-1].append(char)
        else:
            groups.append([char])
            centre = char_centre

    # Attach each space to the line whose vertical band contains it, so word
    # separation survives into the assembled text.
    spaces = [c for c in chars if c.char == " "]
    lines = [_Line(group) for group in groups if group]
    for space in spaces:
        for line in lines:
            if line.y0 - 1.0 <= space.y0 <= line.y1 + 1.0 and line.x0 <= space.x0 <= line.x1:
                line.chars.append(space)
                break

    return lines


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _body_font_size(lines: list[_Line]) -> float:
    """The page's body text size, weighted by how much text is set in it.

    Weighting by character count rather than by line count is what stops a
    page of five headings and one paragraph from deciding that headings are
    the body size.
    """
    weighted: list[float] = []
    for line in lines:
        weighted.extend([line.font_size] * max(len(line.text), 1))
    return _median(weighted)


def _classify(line: _Line, *, body_size: float, page_height: float) -> BlockType:
    """Assign a block type from position and type size."""
    text = line.text.strip()
    centre_y = (line.y0 + line.y1) / 2
    band = page_height * _FURNITURE_BAND

    if len(text) <= _FURNITURE_MAX_CHARS:
        if centre_y <= band:
            return BlockType.PAGE_HEADER
        if centre_y >= page_height - band:
            return BlockType.PAGE_FOOTER

    if body_size > 0 and line.font_size >= body_size * _HEADING_SIZE_RATIO:
        return BlockType.SECTION_HEADER

    return BlockType.TEXT


def _to_bbox(line: _Line) -> RawBBox:
    return RawBBox(
        x0=round(line.x0, 2),
        y0=round(line.y0, 2),
        x1=round(line.x1, 2),
        y1=round(line.y1, 2),
    )


def _build_blocks(
    lines: list[_Line], *, body_size: float, page_height: float
) -> list[RawBlock]:
    """Turn classified lines into blocks, merging consecutive body lines."""
    blocks: list[RawBlock] = []
    buffer: list[_Line] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(line.text for line in buffer if line.text).strip()
        if text:
            blocks.append(
                RawBlock(
                    type=BlockType.TEXT,
                    text=text,
                    bbox=RawBBox(
                        x0=round(min(line.x0 for line in buffer), 2),
                        y0=round(min(line.y0 for line in buffer), 2),
                        x1=round(max(line.x1 for line in buffer), 2),
                        y1=round(max(line.y1 for line in buffer), 2),
                    ),
                    font_size=round(_median([line.font_size for line in buffer]), 2),
                )
            )
        buffer.clear()

    previous: _Line | None = None
    for line in lines:
        if not line.text.strip():
            continue

        kind = _classify(line, body_size=body_size, page_height=page_height)

        if kind is BlockType.TEXT:
            if buffer and previous is not None:
                gap = line.y0 - previous.y1
                if gap > max(previous.font_size, 1.0) * _PARAGRAPH_GAP_RATIO:
                    flush()
            buffer.append(line)
        else:
            flush()
            blocks.append(
                RawBlock(
                    type=kind,
                    text=line.text,
                    bbox=_to_bbox(line),
                    font_size=round(line.font_size, 2),
                    is_bold=kind is BlockType.SECTION_HEADER,
                )
            )
        previous = line

    flush()
    return blocks


class TextOnlyBaseline:
    """Reads the PDF text layer. No model, no GPU, no network."""

    name = BASELINE_BACKEND

    def __init__(self) -> None:
        self._ready = True

    @property
    def version(self) -> str:
        try:
            from pypdfium2.version import PDFIUM_INFO, PYPDFIUM_INFO

            return f"pypdfium2 {PYPDFIUM_INFO.version} / pdfium {PDFIUM_INFO.version}"
        except Exception:  # pragma: no cover - version reporting is best effort
            return "pypdfium2 unknown"

    @property
    def model_versions(self) -> dict[str, str]:
        """No models. Reported as empty rather than omitted, so a consumer
        can tell this parse used none."""
        return {}

    def is_ready(self) -> bool:
        return self._ready

    def warm(self) -> None:
        """Nothing to load; present so the Protocol is satisfied uniformly."""
        return None

    def parse(self, data: bytes, *, force_ocr: bool = False) -> RawParse:
        pages: list[RawPage] = []
        warnings: list[tuple[str, str, int | None]] = []

        with PdfDocument(data) as doc:
            for page_number in range(1, doc.page_count + 1):
                geometry = doc.geometry(page_number)

                def failed_page(code: str, message: str) -> None:
                    warnings.append((code, message, page_number))
                    pages.append(
                        RawPage(
                            page_number=page_number,
                            width=geometry.width,
                            height=geometry.height,
                            rotation=geometry.rotation,
                            extraction_source=ExtractionSource.FAILED,
                        )
                    )

                try:
                    chars = doc.text_chars(page_number)
                except Exception as exc:
                    # One unreadable page must never cost the caller the whole
                    # document — the stance inherited from thorn-nlp.
                    failed_page(
                        "PAGE_EXTRACTION_FAILED",
                        f"Page {page_number} could not be read: {exc}",
                    )
                    continue

                if not chars:
                    # No text layer. A deep-learning backend would rasterise
                    # and OCR here; this one reports the gap rather than
                    # returning a silently empty page.
                    failed_page(
                        "SCANNED_PAGE_OCR_FALLBACK",
                        f"Page {page_number} has no text layer and the "
                        f"text-layer baseline cannot OCR. Use the docling "
                        f"backend.",
                    )
                    continue

                lines = _group_into_lines(chars)
                body_size = _body_font_size(lines)
                blocks = _build_blocks(
                    lines, body_size=body_size, page_height=geometry.height
                )

                pages.append(
                    RawPage(
                        page_number=page_number,
                        width=geometry.width,
                        height=geometry.height,
                        rotation=geometry.rotation,
                        extraction_source=ExtractionSource.TEXT_LAYER,
                        blocks=tuple(blocks),
                    )
                )

        return RawParse(pages=tuple(pages), warnings=tuple(warnings))


def register() -> None:
    """Make the historical baseline selectable by name.

    Deliberately opt-in and never called by ``app``: importing this module must
    not quietly re-add a backend the service no longer supports. Evaluation
    tools and the test suite call it; the service does not.
    """
    from app.backends import registry

    registry.register(BASELINE_BACKEND, _build)


def _build() -> LayoutBackend:
    return TextOnlyBaseline()


__all__ = ["BASELINE_BACKEND", "TextOnlyBaseline", "register"]
