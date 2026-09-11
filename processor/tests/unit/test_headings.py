"""Tests for heading levels and section-tree construction.

A wrong level does not just mislabel one heading — it reparents everything
beneath it, so every block under it gets the wrong section and every citation
built from those blocks points at the wrong place.
"""

from __future__ import annotations

from app.backends.base import RawBlock
from app.processing.headings import (
    assign_levels,
    build_sections,
    extend_section_pages,
    numbering_depth,
)
from ledger_doc_contract.v1.enums import BlockType


def heading(text: str, size: float, *, kind: BlockType = BlockType.SECTION_HEADER):
    return RawBlock(type=kind, text=text, font_size=size)


def inputs(*specs: tuple[RawBlock, int]) -> list[tuple[RawBlock, int, str]]:
    return [(block, page, f"p{page:04d}_b{i:03d}") for i, (block, page) in enumerate(specs)]


class TestNumberingDepth:
    def test_single_number(self) -> None:
        assert numbering_depth("12. Income Taxes") == 1

    def test_two_levels(self) -> None:
        assert numbering_depth("12.3 Deferred tax assets") == 2

    def test_three_levels(self) -> None:
        assert numbering_depth("12.3.1 Valuation allowance") == 3

    def test_lettered_item(self) -> None:
        assert numbering_depth("(a) Basis of presentation") == 3

    def test_named_part(self) -> None:
        assert numbering_depth("PART II") == 1
        assert numbering_depth("Item 7A") == 1

    def test_unnumbered_heading(self) -> None:
        assert numbering_depth("Consolidated Statements of Income") is None

    def test_a_number_that_is_not_a_heading_number(self) -> None:
        # A year opening a heading is not a numbering scheme.
        assert numbering_depth("2019 Highlights") is None


class TestLevelAssignment:
    def test_font_size_sets_the_level(self) -> None:
        levels = assign_levels(
            inputs(
                (heading("Big", 20.0), 1),
                (heading("Medium", 14.0), 1),
                (heading("Small", 11.0), 1),
            )
        )
        assert [c.level for c in levels] == [1, 2, 3]

    def test_same_size_gets_the_same_level_across_pages(self) -> None:
        levels = assign_levels(
            inputs(
                (heading("Big", 20.0), 1),
                (heading("Later section", 14.0), 3),
                (heading("Earlier section", 14.0), 1),
            )
        )
        assert levels[1].level == levels[2].level

    def test_near_identical_sizes_cluster_into_one_level(self) -> None:
        """PDF text layers report sizes that wobble slightly along a line."""
        levels = assign_levels(
            inputs((heading("A", 14.0), 1), (heading("B", 14.2), 1))
        )
        assert levels[0].level == levels[1].level

    def test_a_title_is_the_root_whatever_its_size(self) -> None:
        levels = assign_levels(
            inputs(
                (heading("Doc title", 11.0, kind=BlockType.TITLE), 1),
                (heading("Bigger header", 20.0), 1),
            )
        )
        assert levels[0].level == 1

    def test_numbering_refines_within_a_size_rather_than_overriding_it(self) -> None:
        """A numbered heading must not be promoted above a larger unnumbered
        one just for carrying a number."""
        levels = assign_levels(
            inputs(
                (heading("Annual Report 2019", 20.0), 1),
                (heading("Consolidated Statements", 14.0), 1),
                (heading("12. Income Taxes", 14.0), 2),
                (heading("12.3 Deferred", 14.0), 2),
            )
        )
        assert [c.level for c in levels] == [1, 2, 2, 3]


class TestSectionTree:
    def test_nesting_follows_levels(self) -> None:
        sections = build_sections(
            assign_levels(
                inputs(
                    (heading("Root", 20.0), 1),
                    (heading("Child", 14.0), 1),
                    (heading("Grandchild", 11.0), 1),
                )
            )
        )
        assert sections[0].parent_section_id is None
        assert sections[1].parent_section_id == sections[0].section_id
        assert sections[2].parent_section_id == sections[1].section_id
        assert sections[2].path == ("Root", "Child", "Grandchild")

    def test_a_sibling_closes_the_previous_subtree(self) -> None:
        sections = build_sections(
            assign_levels(
                inputs(
                    (heading("Root", 20.0), 1),
                    (heading("Child", 14.0), 1),
                    (heading("Second child", 14.0), 1),
                )
            )
        )
        assert sections[2].parent_section_id == sections[0].section_id
        assert sections[2].path == ("Root", "Second child")

    def test_jumping_back_to_the_top_reparents_correctly(self) -> None:
        sections = build_sections(
            assign_levels(
                inputs(
                    (heading("Root A", 20.0), 1),
                    (heading("Deep", 11.0), 1),
                    (heading("Root B", 20.0), 2),
                )
            )
        )
        assert sections[2].parent_section_id is None
        assert sections[2].path == ("Root B",)

    def test_empty_input_yields_no_sections(self) -> None:
        assert build_sections([]) == []

    def test_blank_headings_are_skipped(self) -> None:
        sections = build_sections(assign_levels(inputs((heading("   ", 20.0), 1))))
        assert sections == []


class TestPageRanges:
    def test_a_section_spans_to_its_last_block(self) -> None:
        sections = build_sections(
            assign_levels(inputs((heading("Note 12", 14.0), 2)))
        )
        assert sections[0].page_start == 2
        assert sections[0].page_end == 2

        widened = extend_section_pages(sections, {sections[0].section_id: 4})
        assert widened[0].page_start == 2
        assert widened[0].page_end == 4

    def test_a_parent_spans_its_children(self) -> None:
        sections = build_sections(
            assign_levels(
                inputs((heading("Root", 20.0), 1), (heading("Child", 14.0), 2))
            )
        )
        widened = extend_section_pages(sections, {sections[1].section_id: 5})
        by_id = {s.section_id: s for s in widened}
        assert by_id[sections[0].section_id].page_end == 5

    def test_no_sections_is_handled(self) -> None:
        assert extend_section_pages([], {}) == []


class TestUnmeasuredHeadings:
    """A heading the text layer does not cover — a scanned page in an
    otherwise born-digital document, which is the common mixed case.
    """

    def test_an_unmeasured_heading_does_not_sink_below_every_measured_one(
        self,
    ) -> None:
        """Treating "no size" as size zero ranks it as the smallest type in
        the document, so the cover title of a scanned first page lands deeper
        than the body headings that follow it — exactly backwards."""
        blocks = [
            (RawBlock(type=BlockType.SECTION_HEADER, text="Scanned Cover Title"), 1, "b0"),
            (RawBlock(type=BlockType.SECTION_HEADER, text="Overview", font_size=18.0), 2, "b1"),
            (RawBlock(type=BlockType.SECTION_HEADER, text="Detail", font_size=11.0), 2, "b2"),
        ]
        levels = [c.level for c in assign_levels(blocks)]
        assert levels[0] <= max(levels[1:])

    def test_an_unmeasured_heading_continues_at_the_current_depth(self) -> None:
        """Not deeper, which would bury it under an unrelated parent; not
        shallower, which would reparent everything after it. Continuing is the
        only guess whose blast radius stops at the heading itself."""
        blocks = [
            (RawBlock(type=BlockType.SECTION_HEADER, text="Big", font_size=18.0), 1, "b0"),
            (RawBlock(type=BlockType.SECTION_HEADER, text="Small", font_size=11.0), 1, "b1"),
            (RawBlock(type=BlockType.SECTION_HEADER, text="Unmeasured"), 2, "b2"),
        ]
        levels = [c.level for c in assign_levels(blocks)]
        assert levels[2] == levels[1]

    def test_a_leading_unmeasured_heading_starts_at_the_top(self) -> None:
        blocks = [
            (RawBlock(type=BlockType.SECTION_HEADER, text="Scanned Title"), 1, "b0"),
        ]
        assert assign_levels(blocks)[0].level == 1

    def test_a_fully_unmeasured_document_is_flat_not_arbitrary(self) -> None:
        """Every page scanned: nothing is measurable, so no hierarchy can be
        claimed and none is invented."""
        blocks = [
            (RawBlock(type=BlockType.SECTION_HEADER, text=f"H{i}"), 1, f"b{i}")
            for i in range(4)
        ]
        assert {c.level for c in assign_levels(blocks)} == {1}

    def test_measured_headings_are_unaffected(self) -> None:
        blocks = [
            (RawBlock(type=BlockType.SECTION_HEADER, text="Big", font_size=18.0), 1, "b0"),
            (RawBlock(type=BlockType.SECTION_HEADER, text="Small", font_size=11.0), 1, "b1"),
        ]
        levels = [c.level for c in assign_levels(blocks)]
        assert levels == [1, 2]
