"""Tests for the typography probe.

The probe is what supplies heading levels when the layout model does not.
Docling says a region is a section header; without a measured type size every
header in the document lands at the same level and the section tree flattens
into a list — taking the citation paths down with it.
"""

from __future__ import annotations

from app.backends.base import RawBBox
from app.ingestion.render import TextChar
from app.processing.typography import TypographyIndex


def chars(
    text: str,
    *,
    x0: float = 72.0,
    y0: float = 100.0,
    size: float = 10.0,
    weight: int = 400,
    advance: float = 6.0,
) -> list[TextChar]:
    """Lay out a string on one line, one box per glyph."""
    out: list[TextChar] = []
    for index, char in enumerate(text):
        left = x0 + index * advance
        out.append(
            TextChar(
                char=char,
                x0=left,
                y0=y0,
                x1=left + advance,
                y1=y0 + size,
                font_size=size,
                font_weight=weight,
            )
        )
    return out


def box(x0: float, y0: float, x1: float, y1: float) -> RawBBox:
    return RawBBox(x0=x0, y0=y0, x1=x1, y1=y1)


class TestMeasurement:
    def test_reports_the_size_inside_the_box(self) -> None:
        index = TypographyIndex(chars("Income Taxes", size=14.0))
        assert index.probe(box(70, 98, 200, 116)).font_size == 14.0

    def test_a_box_with_no_glyphs_is_unmeasured(self) -> None:
        """A scanned page has no text layer. Saying so is what lets heading
        levels fall back to numbering instead of trusting a made-up size."""
        index = TypographyIndex(chars("body", y0=100.0))
        measured = index.probe(box(70, 400, 200, 420))
        assert measured.measured is False
        assert measured.font_size is None

    def test_no_box_is_unmeasured(self) -> None:
        assert TypographyIndex(chars("body")).probe(None).measured is False

    def test_two_blocks_on_one_page_are_measured_separately(self) -> None:
        index = TypographyIndex(
            chars("Heading", y0=100.0, size=18.0) + chars("body text", y0=200.0, size=10.0)
        )
        assert index.probe(box(70, 98, 300, 120)).font_size == 18.0
        assert index.probe(box(70, 198, 300, 212)).font_size == 10.0

    def test_the_mode_wins_not_the_mean(self) -> None:
        """A superscript footnote marker beside a body line must not lift the
        line's measured size, or every footnoted paragraph reads as a heading."""
        line = chars("A long body sentence", size=10.0)
        line += chars("1", x0=200.0, size=6.0)
        assert TypographyIndex(line).probe(box(70, 98, 300, 112)).font_size == 10.0

    def test_glyphs_outside_the_box_are_not_counted(self) -> None:
        index = TypographyIndex(
            chars("inside", x0=72.0, size=12.0) + chars("outside", x0=400.0, size=24.0)
        )
        assert index.probe(box(70, 98, 130, 114)).font_size == 12.0


class TestWeight:
    def test_a_bold_run_is_reported_bold(self) -> None:
        """Financial notes set many sub-headings bold at body size, where the
        size alone says nothing at all."""
        index = TypographyIndex(chars("Total assets", weight=700))
        assert index.probe(box(70, 98, 200, 112)).is_bold is True

    def test_regular_text_is_not_bold(self) -> None:
        index = TypographyIndex(chars("regular", weight=400))
        assert index.probe(box(70, 98, 200, 112)).is_bold is False

    def test_a_bold_lead_in_does_not_make_the_paragraph_bold(self) -> None:
        """'Note 12.' set bold at the head of a paragraph is a lead-in, not a
        heading, and must not promote the paragraph that follows it."""
        line = chars("Note 12.", x0=72.0, weight=700)
        line += chars(
            "The Company files income tax returns in many jurisdictions.",
            x0=130.0,
            weight=400,
        )
        assert TypographyIndex(line).probe(box(70, 98, 500, 112)).is_bold is False

    def test_an_unreported_weight_is_not_bold(self) -> None:
        """pdfium returns -1 for an unknown weight, normalised to 0. Zero must
        read as 'not reported', never as an ultra-light face."""
        index = TypographyIndex(chars("unknown", weight=0))
        assert index.probe(box(70, 98, 200, 112)).is_bold is False


class TestIndexing:
    def test_spaces_are_excluded_from_the_index(self) -> None:
        """A space has a zero-height box on the baseline and carries no type
        information, so counting it would distort the vertical index."""
        index = TypographyIndex(chars("a b", size=10.0))
        assert len(index) == 2

    def test_an_empty_page_probes_cleanly(self) -> None:
        assert TypographyIndex([]).probe(box(0, 0, 100, 100)).measured is False

    def test_a_block_spanning_several_buckets_is_measured_whole(self) -> None:
        """The index buckets glyphs by height; a multi-line paragraph crosses
        bucket boundaries and must still be measured as one block."""
        page: list[TextChar] = []
        for line in range(8):
            page += chars("body line", y0=100.0 + line * 14.0, size=10.0)
        measured = TypographyIndex(page).probe(box(70, 98, 300, 214))
        assert measured.font_size == 10.0
        assert measured.char_count == 8 * len("body line".replace(" ", ""))
