"""Tests for running-header and footer detection.

The failure this prevents is specific and expensive. A layout model reads each
page alone, sees a short line in a heading position, and calls it a section
header. That opens a section, and every block after it inherits the section in
its citation path — so a misread running footer ends up inside an answer's
evidence field. Repetition across pages is the only signal that catches it.
"""

from __future__ import annotations

from app.backends.base import RawBBox, RawBlock, RawPage
from app.processing.page_furniture import find_furniture, normalise, reclassify
from ledger_doc_contract.v1.enums import BlockType

WIDTH, HEIGHT = 612.0, 792.0


def block(
    text: str,
    y: float,
    *,
    kind: BlockType = BlockType.SECTION_HEADER,
    x0: float = 72.0,
    x1: float = 540.0,
) -> RawBlock:
    return RawBlock(
        type=kind, text=text, bbox=RawBBox(x0=x0, y0=y, x1=x1, y1=y + 12.0)
    )


def page(number: int, blocks: list[RawBlock]) -> RawPage:
    return RawPage(
        page_number=number, width=WIDTH, height=HEIGHT, blocks=tuple(blocks)
    )


def report(count: int, *, header: str | None = None, footer: str | None = None):
    """A document with an optional running header and footer on every page."""
    pages = []
    for number in range(1, count + 1):
        blocks = []
        if header:
            blocks.append(block(header, 20.0))
        blocks.append(block(f"Body of page {number}", 300.0, kind=BlockType.TEXT))
        if footer:
            blocks.append(block(footer, 760.0))
        pages.append(page(number, blocks))
    return pages


def types_of(pages: list[RawPage], text: str) -> set[BlockType]:
    return {b.type for p in pages for b in p.blocks if b.text == text}


class TestNormalisation:
    def test_page_numbers_collapse_so_a_footer_looks_repeated(self) -> None:
        """The whole point of a page number is that it changes. Comparing
        'Page 7' to 'Page 8' literally finds no repetition at all."""
        assert normalise("Page 7") == normalise("Page 8")

    def test_case_and_spacing_do_not_split_a_repeat(self) -> None:
        assert normalise("FINAL  PROJECT") == normalise("Final Project")

    def test_different_lines_stay_different(self) -> None:
        assert normalise("Income Taxes") != normalise("Revenue")


class TestDetection:
    def test_a_line_repeated_in_the_top_band_is_found(self) -> None:
        found = find_furniture(report(4, header="Annual Report 2019"))
        assert ("top", normalise("Annual Report 2019")) in found

    def test_a_line_repeated_in_the_bottom_band_is_found(self) -> None:
        found = find_furniture(report(4, footer="FINAL PROJECT"))
        assert ("bottom", normalise("FINAL PROJECT")) in found

    def test_body_text_is_never_furniture(self) -> None:
        """It repeats nowhere near a margin, so position alone excludes it."""
        pages = [
            page(n, [block("Recurring body line", 400.0, kind=BlockType.TEXT)])
            for n in range(1, 5)
        ]
        assert find_furniture(pages) == set()

    def test_one_occurrence_is_not_a_repeat(self) -> None:
        pages = report(4)
        pages[0] = page(1, [block("12. Income Taxes", 20.0)])
        assert find_furniture(pages) == set()

    def test_a_line_twice_on_one_page_is_not_a_running_header(self) -> None:
        """A layout quirk, not a repeat. Counting occurrences instead of pages
        would let a single page promote itself over the threshold."""
        pages = [page(1, [block("Duplicated", 20.0), block("Duplicated", 30.0)])]
        assert find_furniture(pages) == set()

    def test_a_rare_line_in_a_long_document_is_not_furniture(self) -> None:
        """Two pages out of forty is a coincidence; two out of four is the
        running header. The share matters, not just the count."""
        pages = report(40)
        for index in (0, 1):
            pages[index] = page(index + 1, [block("Appears Twice", 20.0)])
        assert find_furniture(pages) == set()

    def test_a_single_page_document_has_no_furniture(self) -> None:
        """Nothing can be shown to repeat, so nothing is removed. Guessing
        from one page is what the layout model already did."""
        assert find_furniture(report(1, header="Header")) == set()

    def test_repeated_boilerplate_counts_as_furniture(self) -> None:
        """A whole sentence repeated verbatim at the top of every page is a
        confidentiality notice, not prose. Length is not what disqualifies
        something — failing to repeat is."""
        text = "Confidential. Not for distribution outside the recipient organisation."
        pages = [page(n, [block(text, 20.0, kind=BlockType.TEXT)]) for n in range(1, 5)]
        assert ("top", normalise(text)) in find_furniture(pages)

    def test_a_genuinely_long_block_near_the_margin_is_left_alone(self) -> None:
        """Past a point it is a paragraph that happens to start high on the
        page, and demoting it would drop real prose out of every chunk."""
        text = (
            "Total revenue increased seven per cent in 2019, driven by "
            "subscription growth across all reportable segments, with the "
            "remainder attributable to pricing actions taken in the second half."
        )
        assert len(text) > 120
        pages = [page(n, [block(text, 20.0, kind=BlockType.TEXT)]) for n in range(1, 5)]
        assert find_furniture(pages) == set()


class TestReclassification:
    def test_a_repeated_header_stops_being_a_heading(self) -> None:
        """The defect this exists for: one misread running footer otherwise
        appends itself to every citation path beneath it."""
        pages = reclassify(report(4, header="AI TEAM TRAINING '27"))
        assert types_of(pages, "AI TEAM TRAINING '27") == {BlockType.PAGE_HEADER}

    def test_the_band_decides_header_or_footer(self) -> None:
        pages = reclassify(report(4, header="Top line", footer="FINAL PROJECT"))
        assert types_of(pages, "Top line") == {BlockType.PAGE_HEADER}
        assert types_of(pages, "FINAL PROJECT") == {BlockType.PAGE_FOOTER}

    def test_real_headings_are_left_alone(self) -> None:
        pages = report(4, header="Running header")
        pages[1] = page(
            2,
            [
                block("Running header", 20.0),
                block("12. Income Taxes", 200.0),
                block("Body", 300.0, kind=BlockType.TEXT),
            ],
        )
        result = reclassify(pages)
        assert types_of(result, "12. Income Taxes") == {BlockType.SECTION_HEADER}

    def test_nothing_is_deleted(self) -> None:
        """Furniture is demoted, not dropped. A consumer that wants to rebuild
        the whole page still can; it is only kept out of the reading order."""
        before = report(4, header="Running header")
        after = reclassify(before)
        assert [len(p.blocks) for p in after] == [len(p.blocks) for p in before]

    def test_a_document_with_no_furniture_is_returned_untouched(self) -> None:
        pages = report(4)
        assert reclassify(pages) is pages

    def test_a_table_is_never_demoted(self) -> None:
        """Demoting a table would drop a whole statement out of the reading
        order — the single most expensive block on the page to lose."""
        from app.backends.base import RawCell, RawTable

        table = RawTable(
            n_rows=1, n_cols=1, cells=(RawCell(row=0, col=0, text="1,234"),)
        )
        pages = [
            page(
                n,
                [
                    RawBlock(
                        type=BlockType.TABLE,
                        text="",
                        bbox=RawBBox(x0=72, y0=20, x1=540, y1=32),
                        table=table,
                    )
                ],
            )
            for n in range(1, 5)
        ]
        assert all(
            b.type is BlockType.TABLE for p in reclassify(pages) for b in p.blocks
        )


class TestThroughAssembly:
    def test_a_running_footer_never_reaches_a_citation_path(self) -> None:
        """End to end: the section path is the string that lands in an
        answer's evidence field, and this is what keeps a page footer out
        of it."""
        from app.core.settings import Settings
        from app.processing.assemble import assemble
        from app.backends.base import RawParse

        pages = []
        for number in range(1, 5):
            pages.append(
                page(
                    number,
                    [
                        RawBlock(
                            type=BlockType.SECTION_HEADER,
                            text="12. Income Taxes",
                            bbox=RawBBox(x0=72, y0=200, x1=540, y1=216),
                            font_size=14.0,
                        ),
                        RawBlock(
                            type=BlockType.TEXT,
                            text=f"Body of page {number}",
                            bbox=RawBBox(x0=72, y0=300, x1=540, y1=316),
                            font_size=10.0,
                        ),
                        block("FINAL PROJECT", 760.0),
                    ],
                )
            )

        document = assemble(
            RawParse(pages=tuple(pages)),
            document_id="doc_test",
            filename="t.pdf",
            content_sha256="a" * 64,
            backend_name="fake",
            backend_version="0",
            model_versions={},
            duration_ms=0,
            settings=Settings(),
        )

        titles = {s.title for s in document.sections}
        assert "FINAL PROJECT" not in titles
        assert "12. Income Taxes" in titles

        for block_ in document.iter_blocks():
            path = document.section_path(block_.section_id)
            assert "FINAL PROJECT" not in path
