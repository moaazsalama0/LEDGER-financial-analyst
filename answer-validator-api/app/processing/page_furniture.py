"""Finding the running headers and footers a layout model called headings.

A layout model classifies each page on its own, so it has no way to know that
the same line sits at the top of nine consecutive pages. It sees a short bold
line in a heading position and says "section header" — which is locally
reasonable and globally wrong.

The cost is not cosmetic. A spurious heading opens a section, and every block
after it inherits that section in its citation path. One misread running
footer turns a real citation into

    Notes to Financial Statements > 12. Income Taxes > FINAL PROJECT

which is the string that reaches an answer's evidence field. Repetition is the
signal no single page carries, and it is the one this module supplies.

Adapted from thorn-nlp's ``layout/header_footer_detector.py``, which existed
for the same reason and is language-agnostic: it compares positions and
repetition, never words.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.backends.base import RawBlock, RawPage
from ledger_doc_contract.v1.enums import BlockType

#: Fraction of page height at the top and bottom within which a repeated line
#: is a candidate. Wider than the band a single-page heuristic dares use,
#: because a decision corroborated across pages can afford to look further.
_BAND = 0.12

#: A repeated line must appear on at least this many pages. Two is the floor
#: at which "repeated" means anything at all.
_MIN_PAGES = 2

#: ...and on at least this share of the document. A line on 2 pages of 40 is a
#: coincidence; on 2 pages of 4 it is the running header.
_MIN_SHARE = 0.25

#: Longer than this and it is a paragraph that happens to sit near the margin,
#: not a running header.
_MAX_CHARS = 120

_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Reduce a line to what stays the same from page to page.

    Digits collapse to a placeholder because the whole point of a page number
    is that it changes: "Page 7" and "Page 8" are one running footer, and
    comparing them literally would find no repetition at all.
    """
    return _SPACE.sub(" ", _DIGITS.sub("#", text)).strip().casefold()


def _band(block: RawBlock, page_height: float) -> str | None:
    """Which margin band a block sits in, if either."""
    if block.bbox is None or page_height <= 0:
        return None
    centre = (block.bbox.y0 + block.bbox.y1) / 2
    if centre <= page_height * _BAND:
        return "top"
    if centre >= page_height * (1 - _BAND):
        return "bottom"
    return None


def find_furniture(pages: list[RawPage]) -> set[tuple[str, str]]:
    """Return the ``(band, normalised text)`` pairs that repeat across pages.

    Counting distinct *pages* rather than occurrences matters: a line printed
    twice on one page is a layout quirk, not a running header, and must not
    reach the threshold on its own.
    """
    if len(pages) < _MIN_PAGES:
        return set()

    seen: dict[tuple[str, str], set[int]] = defaultdict(set)

    for page in pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text or len(text) > _MAX_CHARS:
                continue
            band = _band(block, page.height)
            if band is None:
                continue
            seen[(band, normalise(text))].add(page.page_number)

    threshold = max(_MIN_PAGES, int(len(pages) * _MIN_SHARE + 0.999))
    return {key for key, page_numbers in seen.items() if len(page_numbers) >= threshold}


def reclassify(pages: list[RawPage]) -> list[RawPage]:
    """Relabel repeated margin lines as page furniture.

    Furniture is still emitted — a consumer that wants to rebuild the whole
    page can — but it is excluded from the reading order and, because it is no
    longer a heading, it stops opening sections that never existed.

    Nothing is deleted and nothing else is touched: a block the layout model
    got right keeps the label the model gave it.
    """
    furniture = find_furniture(pages)
    if not furniture:
        return pages

    updated: list[RawPage] = []
    for page in pages:
        blocks: list[RawBlock] = []
        changed = False
        for block in page.blocks:
            band = _band(block, page.height)
            key = (band, normalise(block.text.strip())) if band else None

            if (
                key in furniture
                and not block.type.is_page_furniture
                and block.table is None
            ):
                blocks.append(
                    RawBlock(
                        type=(
                            BlockType.PAGE_HEADER
                            if band == "top"
                            else BlockType.PAGE_FOOTER
                        ),
                        text=block.text,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        font_size=block.font_size,
                        is_bold=block.is_bold,
                    )
                )
                changed = True
            else:
                blocks.append(block)

        updated.append(
            RawPage(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                rotation=page.rotation,
                extraction_source=page.extraction_source,
                ocr_confidence=page.ocr_confidence,
                blocks=tuple(blocks),
            )
            if changed
            else page
        )

    return updated


__all__ = ["find_furniture", "normalise", "reclassify"]
