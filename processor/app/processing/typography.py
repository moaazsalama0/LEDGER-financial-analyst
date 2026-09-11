"""Measuring the type a layout block was set in.

A layout model reports *that* a region is a section header; none of them
report the size or weight it was set in. But level assignment needs exactly
that, because a document's heading hierarchy is expressed typographically and
nowhere else. So the type is measured here, from the glyphs of the PDF's own
text layer that fall inside the block the model found.

This is the join between the two halves of the pipeline: the deep-learning
model decides *what* each region is, and the text layer says *how it looks*.
Neither alone is enough — the model has no typography, and typography alone
cannot tell a heading from an emphasised sentence.

Pages with no text layer yield nothing, and the probe says so by returning
``None`` rather than a fabricated default. A heading with no measured size
falls back to numbering depth, which is the honest outcome for a scanned page.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.ingestion.render import TextChar

#: Height of one bucket in the vertical index, in points. Chosen to be a bit
#: larger than a typical body line so a block's glyphs land in few buckets.
_BUCKET_PT = 12.0

#: A glyph counts as belonging to a block when its centre is inside the box,
#: grown by this many points. Layout boxes are drawn to the ink, and a glyph's
#: box includes its descender, so exact containment loses the last line.
_MARGIN_PT = 1.5

#: Fraction of a block's glyphs that must be bold before the block is called
#: bold. A run-in bold lead-in on a body paragraph is common and must not
#: promote the paragraph.
_BOLD_SHARE = 0.6


@dataclass(frozen=True)
class Typography:
    """The type a block was set in, as measured from the text layer."""

    font_size: float | None
    is_bold: bool
    char_count: int

    @property
    def measured(self) -> bool:
        """False when no glyph of the text layer fell inside the block."""
        return self.char_count > 0


#: What the probe reports for a block it could not measure.
UNMEASURED = Typography(font_size=None, is_bold=False, char_count=0)


class TypographyIndex:
    """A page's glyphs, indexed by vertical position for repeated lookup.

    A page holds a few thousand glyphs and a few dozen blocks. Testing every
    glyph against every block is quadratic enough to show up on a
    2,758-document corpus run, so glyphs are bucketed by their vertical centre
    and a lookup touches only the bands the block actually spans.
    """

    def __init__(self, chars: list[TextChar]) -> None:
        self._buckets: dict[int, list[TextChar]] = defaultdict(list)
        for char in chars:
            if not char.char.strip():
                # A space has a zero-height box on the baseline. It carries no
                # type information and would distort the vertical index.
                continue
            centre = (char.y0 + char.y1) / 2
            self._buckets[int(centre // _BUCKET_PT)].append(char)

    def __len__(self) -> int:
        return sum(len(chars) for chars in self._buckets.values())

    def probe(self, bbox: object | None) -> Typography:
        """Measure the type inside ``bbox``.

        Takes anything with ``x0``/``y0``/``x1``/``y1`` so the probe does not
        depend on which backend's box type it is handed.
        """
        if bbox is None:
            return UNMEASURED

        x0 = float(bbox.x0) - _MARGIN_PT  # type: ignore[attr-defined]
        y0 = float(bbox.y0) - _MARGIN_PT  # type: ignore[attr-defined]
        x1 = float(bbox.x1) + _MARGIN_PT  # type: ignore[attr-defined]
        y1 = float(bbox.y1) + _MARGIN_PT  # type: ignore[attr-defined]

        first = int(y0 // _BUCKET_PT)
        last = int(y1 // _BUCKET_PT)

        sizes: dict[float, int] = defaultdict(int)
        bold = 0
        total = 0

        for bucket in range(first, last + 1):
            for char in self._buckets.get(bucket, ()):
                cx = (char.x0 + char.x1) / 2
                cy = (char.y0 + char.y1) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    sizes[char.font_size] += 1
                    total += 1
                    if char.is_bold:
                        bold += 1

        if not total:
            return UNMEASURED

        # The mode, not the mean: a superscript footnote marker or a single
        # large drop-cap must not shift a block's measured size. Ties break
        # toward the larger size, which is the one a reader notices.
        size = max(sizes.items(), key=lambda kv: (kv[1], kv[0]))[0]

        return Typography(
            font_size=round(size, 2),
            is_bold=bold >= total * _BOLD_SHARE,
            char_count=total,
        )


__all__ = ["UNMEASURED", "Typography", "TypographyIndex"]
