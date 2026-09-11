"""Heading levels and the document's section tree.

A layout model tells us *that* a block is a heading; none of them report how
deep it sits. Without a level there is no hierarchy, and without a hierarchy
there is no section path — which is exactly the string an answer citation has
to carry. So level assignment happens here, from typography and numbering.

The section tree is then a stack machine over those levels, carried across
page boundaries: a note that starts on page 2 and continues onto page 3 is one
section, not two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.backends.base import RawBlock
from ledger_doc_contract.v1.enums import BlockType
from ledger_doc_contract.v1.models import Section

#: Two font sizes within this many points are treated as the same style. PDF
#: text layers report sizes that wobble slightly across a line.
_SIZE_TOLERANCE_PT = 0.6

#: Numbering patterns, most specific first. The depth of a dotted number is
#: strong evidence of nesting and outranks font size when present, because
#: "12.3.1" is unambiguous where a one-point size difference is not.
_NUMBERING_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)[.)]?\s+"), 3),
    (re.compile(r"^\s*(\d+)\.(\d+)[.)]?\s+"), 2),
    (re.compile(r"^\s*(\d+)[.)]\s+"), 1),
    (re.compile(r"^\s*\(?([a-z])\)\s+", re.IGNORECASE), 3),
    (re.compile(r"^\s*(?:PART|ITEM|NOTE|SECTION)\s+[IVXLC\d]+", re.IGNORECASE), 1),
)


@dataclass(frozen=True)
class HeadingCandidate:
    """A heading block with the level assigned to it."""

    block: RawBlock
    page_number: int
    block_id: str
    level: int


def numbering_depth(text: str) -> int | None:
    """Return the nesting depth implied by a heading's numbering, if any.

    ``"12. Income Taxes"`` gives 1, ``"12.3 Deferred"`` gives 2. Returns
    ``None`` when the heading is not numbered, which is the common case and
    falls through to the font-size ranking.
    """
    for pattern, depth in _NUMBERING_PATTERNS:
        if pattern.match(text):
            return depth
    return None


def _size_of(block: RawBlock) -> float | None:
    """The measured type size, or ``None`` when nothing could be measured.

    Kept nullable rather than defaulted to zero. Zero would be a *size*, and
    it would sort below every real one — so a heading on a scanned page in an
    otherwise born-digital document would rank as the smallest type in the
    document and sink to the deepest level, which is the opposite of what an
    unreadable page warrants.
    """
    return block.font_size


def _rank_sizes(sizes: list[float]) -> dict[float, int]:
    """Map each font size to a 1-based rank, largest first.

    Sizes are grouped by proximity rather than snapped to a fixed grid.
    Snapping looks equivalent but is not: 14.0 and 14.2 straddle a grid
    boundary and would be ranked as two distinct heading levels, even though
    they are the same style with the wobble a PDF text layer reports along a
    line.
    """
    ordered = sorted(set(sizes), reverse=True)
    ranks: dict[float, int] = {}
    rank = 0
    representative: float | None = None

    for size in ordered:
        if representative is None or (representative - size) > _SIZE_TOLERANCE_PT:
            rank += 1
            representative = size
        ranks[size] = rank

    return ranks


def assign_levels(
    heading_blocks: list[tuple[RawBlock, int, str]],
) -> list[HeadingCandidate]:
    """Assign a 1-based level to every heading in the document.

    Takes ``(block, page_number, block_id)`` triples in document order.

    Font sizes are ranked across the whole document rather than per page, so a
    section header on page 3 lands at the same level as an identically styled
    one on page 1. Explicit numbering, where present, overrides the ranking.
    """
    if not heading_blocks:
        return []

    # Font size sets the level. Ranking across the whole document rather than
    # per page is what makes an identically styled header on page 3 land at
    # the same level as one on page 1. Only measured sizes are ranked:
    # comparing a size against the absence of one is not a comparison.
    measured = [s for s in (_size_of(b) for b, _, _ in heading_blocks) if s is not None]
    rank_by_size = _rank_sizes(measured)

    candidates: list[HeadingCandidate] = []
    previous_level = 1

    for block, page_number, block_id in heading_blocks:
        size = _size_of(block)

        if block.type is BlockType.TITLE:
            # A document title is the root whatever size it was set in.
            level = 1
        elif size is None:
            # Nothing to measure — a scanned page, or a heading the text layer
            # does not cover. It inherits the depth of the last heading we
            # could measure, which says "assume this continues at the current
            # depth" rather than inventing one. Guessing deeper would bury it
            # under an unrelated parent; guessing shallower would reparent
            # everything after it. Continuing is the only choice whose blast
            # radius is limited to the heading itself.
            level = previous_level
        else:
            level = rank_by_size.get(size, 1)

            # Numbering refines the level within a style rather than replacing
            # it. "12.3 Deferred" set in the same face as "12. Income Taxes"
            # is one level deeper, but neither is promoted above a larger
            # heading just for carrying a number.
            depth = numbering_depth(block.text)
            if depth is not None:
                level += depth - 1

        previous_level = level

        candidates.append(
            HeadingCandidate(
                block=block, page_number=page_number, block_id=block_id, level=level
            )
        )

    return candidates


def build_sections(candidates: list[HeadingCandidate]) -> list[Section]:
    """Build the flat section list, with parent links, from ranked headings.

    A stack tracks the open ancestors. A heading at level N closes every open
    section at level N or deeper, then attaches to whatever remains — which is
    what makes a heading that jumps from level 3 back to level 1 reparent
    correctly rather than nesting under its predecessor.
    """
    sections: list[Section] = []
    stack: list[Section] = []

    for index, candidate in enumerate(candidates):
        title = candidate.block.text.strip()
        if not title:
            continue

        while stack and stack[-1].level >= candidate.level:
            stack.pop()

        parent = stack[-1] if stack else None
        section = Section(
            section_id=f"s{index:03d}",
            title=title,
            level=candidate.level,
            parent_section_id=parent.section_id if parent else None,
            path=(parent.path if parent else ()) + (title,),
            heading_block_id=candidate.block_id,
            page_start=candidate.page_number,
            page_end=candidate.page_number,
        )
        sections.append(section)
        stack.append(section)

    return sections


def extend_section_pages(
    sections: list[Section], last_page_by_section: dict[str, int]
) -> list[Section]:
    """Widen each section's page range to cover the blocks assigned to it.

    A section's ``page_end`` starts as the page its heading sits on; this
    pushes it out to the last page holding its content, so a note spanning
    pages 2 to 3 reports both. Ancestors are widened too, since a parent
    necessarily spans its children.
    """
    if not sections:
        return sections

    by_id = {s.section_id: s for s in sections}
    ends = {s.section_id: s.page_end for s in sections}

    for section_id, page in last_page_by_section.items():
        current = by_id.get(section_id)
        while current is not None:
            ends[current.section_id] = max(ends[current.section_id], page)
            current = (
                by_id.get(current.parent_section_id)
                if current.parent_section_id
                else None
            )

    return [s.model_copy(update={"page_end": ends[s.section_id]}) for s in sections]


__all__ = [
    "HeadingCandidate",
    "assign_levels",
    "build_sections",
    "extend_section_pages",
    "numbering_depth",
]
