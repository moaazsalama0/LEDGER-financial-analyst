"""Reading-order reconstruction for a page of blocks.

Adapted from thorn-nlp ``services/document_intelligence/reading_order/order.py``.
That module was written for right-to-left Arabic legal pages but had already
been parameterised with an ``rtl`` flag, so the column traversal transfers
directly by passing ``rtl=False``.

What was dropped: the Arabic-legal deferral rules, which pushed footnotes and
margin notes to the end of the page and sorted chapter/article headings ahead
of body text at the same y. Financial reports do not have that hierarchy, and
applying it here would reorder table captions away from their tables.

What was kept: dropping page furniture from the order so running headers and
page numbers never leak into a retrieval chunk, and bucketing blocks into
columns by horizontal centre before sorting vertically, which is what stops a
two-column page from being read straight across.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.backends.base import RawBlock

#: A column split is only believed when each side holds at least this share of
#: the page's blocks. Without it, a single wide title plus a narrow sidebar
#: reads as two columns and the page order is scrambled.
_MIN_COLUMN_SHARE = 0.25

#: Blocks must be narrower than this fraction of the page to be column
#: candidates at all; a full-width heading spans both columns.
_MAX_COLUMN_WIDTH_SHARE = 0.6


@dataclass(frozen=True)
class OrderedBlock:
    """A block paired with its position in the page's reading order."""

    block: RawBlock
    order_index: int


def _centre_x(block: RawBlock) -> float:
    if block.bbox is None:
        return 0.0
    return (block.bbox.x0 + block.bbox.x1) / 2


def _top(block: RawBlock) -> float:
    return block.bbox.y0 if block.bbox is not None else 0.0


def _width(block: RawBlock) -> float:
    if block.bbox is None:
        return 0.0
    return block.bbox.x1 - block.bbox.x0


def detect_column_count(blocks: list[RawBlock], *, page_width: float) -> int:
    """Return 1 or 2, deciding whether the page is laid out in two columns.

    Only the two cases that occur in financial reports are distinguished.
    Detecting more columns than exist is far more damaging than missing one:
    a wrong split interleaves unrelated text, while a missed split merely
    keeps the page's natural top-to-bottom order.
    """
    if page_width <= 0:
        return 1

    candidates = [
        b
        for b in blocks
        if b.bbox is not None and _width(b) <= page_width * _MAX_COLUMN_WIDTH_SHARE
    ]
    if len(candidates) < 4:
        return 1

    midpoint = page_width / 2
    left = [b for b in candidates if _centre_x(b) < midpoint]
    right = [b for b in candidates if _centre_x(b) >= midpoint]

    total = len(candidates)
    if len(left) < total * _MIN_COLUMN_SHARE or len(right) < total * _MIN_COLUMN_SHARE:
        return 1

    # Two columns only if the groups genuinely separate: the rightmost edge of
    # the left group must sit left of the leftmost edge of the right group.
    left_edge = max(b.bbox.x1 for b in left if b.bbox is not None)
    right_edge = min(b.bbox.x0 for b in right if b.bbox is not None)
    return 2 if left_edge <= right_edge else 1


def order_blocks(
    blocks: list[RawBlock],
    *,
    page_width: float,
    rtl: bool = False,
) -> list[OrderedBlock]:
    """Return blocks in reading order, page furniture excluded.

    Furniture is dropped from the order but the blocks themselves are still
    returned to the caller elsewhere, so a consumer can reconstruct the whole
    page while a chunker following ``reading_order`` never sees a running
    header.
    """
    body = [b for b in blocks if not b.type.is_page_furniture]
    if not body:
        return []

    column_count = detect_column_count(body, page_width=page_width)

    if column_count == 1:
        ordered = sorted(body, key=lambda b: (_top(b), _centre_x(b)))
    else:
        midpoint = page_width / 2

        def column_of(block: RawBlock) -> int:
            # A block wider than half the page spans the columns and belongs
            # with whatever it sits above, so it sorts into the first column.
            if _width(block) > page_width * _MAX_COLUMN_WIDTH_SHARE:
                return 0
            index = 0 if _centre_x(block) < midpoint else 1
            return (1 - index) if rtl else index

        ordered = sorted(body, key=lambda b: (column_of(b), _top(b), _centre_x(b)))

    return [OrderedBlock(block=b, order_index=i) for i, b in enumerate(ordered)]


__all__ = ["OrderedBlock", "detect_column_count", "order_blocks"]
