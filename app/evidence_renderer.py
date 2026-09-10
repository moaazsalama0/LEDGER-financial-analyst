"""
evidence_renderer.py
=====================
Bonus feature: "Bounding-box highlighting of the cited evidence on the
source PDF page."

The Strict Answer Schema's evidence objects only carry
{document_id, page, section} — no bounding box travels through
agent-service -> answer-validator-api -> orchestrator-api. Rather than
skip the bonus for that reason, this renders the ACTUAL cited page from
the original PDF (cached from the upload the user made in this session)
and uses PyMuPDF's own text search on that page to recover a real
bounding box for the evidence, which it highlights in yellow before
turning the page into an image the Gradio UI can show. If the exact
answer text doesn't produce a match, it falls back to highlighting the
`section` heading instead, so the user still sees *where* on the page to
look — and if the PDF for that document was never uploaded in this
session, it says so plainly instead of pretending to have an image.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

HIGHLIGHT_COLOR = (1, 0.85, 0.2)  # soft yellow, RGB 0..1 for PyMuPDF


@dataclass
class RenderedEvidence:
    ok: bool
    image: Optional[Image.Image] = None
    matched_text: Optional[str] = None
    note: str = ""
    page_count: Optional[int] = None


class PDFEvidenceRenderer:
    """Turns (pdf_bytes, page_number, candidate highlight strings) into a highlighted page image."""

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    @staticmethod
    def _clean_candidates(*values: object) -> list[str]:
        """
        Builds a short, ordered list of strings worth searching for on the
        page, from most-specific (the actual answer value) to least
        (the section heading). Strips currency/percent symbols and commas
        since the page's raw text often differs from the formatted answer
        (e.g. "$142.5M" vs "142.5").
        """
        candidates: list[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                candidates.extend(str(v) for v in value if v not in (None, ""))
                continue
            candidates.append(str(value))

        cleaned: list[str] = []
        for c in candidates:
            c = c.strip()
            if not c:
                continue
            cleaned.append(c)
            stripped = c.replace("$", "").replace("%", "").replace(",", "").strip()
            if stripped and stripped != c:
                cleaned.append(stripped)
        return cleaned

    def render(
        self,
        pdf_bytes: bytes,
        page_number: int,
        answer_value: object = None,
        section: Optional[str] = None,
    ) -> RenderedEvidence:
        if not pdf_bytes:
            return RenderedEvidence(ok=False, note="Original PDF for this document isn't cached in this session.")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            return RenderedEvidence(ok=False, note=f"Could not open the PDF for preview: {exc}")

        try:
            page_count = doc.page_count
            index = (page_number or 1) - 1  # evidence pages are 1-indexed
            if index < 0 or index >= page_count:
                return RenderedEvidence(
                    ok=False,
                    note=f"Page {page_number} is out of range for this document ({page_count} pages).",
                    page_count=page_count,
                )

            page = doc[index]
            matched_text: Optional[str] = None

            # Try the most specific candidates first (the answer itself),
            # then fall back to the section heading.
            for candidate in self._clean_candidates(answer_value):
                rects = page.search_for(candidate)
                if rects:
                    for r in rects:
                        page.add_highlight_annot(r).set_colors(stroke=HIGHLIGHT_COLOR)
                    matched_text = candidate
                    break

            if matched_text is None and section:
                for candidate in self._clean_candidates(section):
                    rects = page.search_for(candidate)
                    if rects:
                        for r in rects:
                            page.add_highlight_annot(r).set_colors(stroke=HIGHLIGHT_COLOR)
                        matched_text = candidate
                        break

            zoom = self.dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            image = Image.open(BytesIO(pixmap.tobytes("png")))

            note = (
                f"Highlighted match: \u201c{matched_text}\u201d on page {page_number}."
                if matched_text
                else f"Showing page {page_number} — no exact text match found to highlight."
            )
            return RenderedEvidence(ok=True, image=image, matched_text=matched_text, note=note, page_count=page_count)
        finally:
            doc.close()
