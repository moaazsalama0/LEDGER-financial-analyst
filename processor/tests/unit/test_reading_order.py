"""Tests for reading-order reconstruction.

Wrong order splices unrelated sentences together, and a chunk built from the
result reads as nonsense while still carrying a confident citation.
"""

from __future__ import annotations

from app.backends.base import RawBBox, RawBlock
from app.processing.reading_order import detect_column_count, order_blocks
from ledger_doc_contract.v1.enums import BlockType

PAGE_WIDTH = 612.0


def block(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    kind: BlockType = BlockType.TEXT,
) -> RawBlock:
    return RawBlock(type=kind, text=text, bbox=RawBBox(x0=x0, y0=y0, x1=x1, y1=y1))


def texts(ordered) -> list[str]:
    return [entry.block.text for entry in ordered]


class TestSingleColumn:
    def test_sorts_top_to_bottom(self) -> None:
        blocks = [
            block("third", 72, 300, 540, 320),
            block("first", 72, 100, 540, 120),
            block("second", 72, 200, 540, 220),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH)) == [
            "first",
            "second",
            "third",
        ]

    def test_left_to_right_within_a_row(self) -> None:
        blocks = [
            block("right", 300, 100, 400, 120),
            block("left", 72, 100, 200, 120),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH)) == ["left", "right"]

    def test_order_index_is_sequential(self) -> None:
        blocks = [block(f"b{i}", 72, i * 20.0, 540, i * 20.0 + 10) for i in range(5)]
        ordered = order_blocks(blocks, page_width=PAGE_WIDTH)
        assert [e.order_index for e in ordered] == [0, 1, 2, 3, 4]


class TestPageFurniture:
    def test_furniture_is_dropped_from_the_order(self) -> None:
        blocks = [
            block("running header", 72, 20, 540, 32, kind=BlockType.PAGE_HEADER),
            block("body", 72, 100, 540, 120),
            block("page 7", 290, 760, 330, 772, kind=BlockType.PAGE_FOOTER),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH)) == ["body"]

    def test_a_page_of_only_furniture_orders_nothing(self) -> None:
        blocks = [block("page 7", 290, 760, 330, 772, kind=BlockType.PAGE_FOOTER)]
        assert order_blocks(blocks, page_width=PAGE_WIDTH) == []


class TestColumnDetection:
    def test_a_plain_page_is_one_column(self) -> None:
        blocks = [block(f"b{i}", 72, i * 20.0, 540, i * 20.0 + 10) for i in range(6)]
        assert detect_column_count(blocks, page_width=PAGE_WIDTH) == 1

    def test_two_separated_groups_are_two_columns(self) -> None:
        left = [block(f"L{i}", 72, i * 20.0, 290, i * 20.0 + 10) for i in range(4)]
        right = [block(f"R{i}", 320, i * 20.0, 540, i * 20.0 + 10) for i in range(4)]
        assert detect_column_count(left + right, page_width=PAGE_WIDTH) == 2

    def test_overlapping_groups_are_not_columns(self) -> None:
        """Blocks that overlap horizontally are one column with varied
        indentation, not two columns."""
        left = [block(f"L{i}", 72, i * 20.0, 400, i * 20.0 + 10) for i in range(4)]
        right = [block(f"R{i}", 350, i * 20.0, 540, i * 20.0 + 10) for i in range(4)]
        assert detect_column_count(left + right, page_width=PAGE_WIDTH) == 1

    def test_too_few_blocks_is_one_column(self) -> None:
        """A split claimed from two blocks is a guess, and a wrong split is
        far more damaging than a missed one."""
        blocks = [block("L", 72, 0, 290, 10), block("R", 320, 0, 540, 10)]
        assert detect_column_count(blocks, page_width=PAGE_WIDTH) == 1

    def test_a_lopsided_page_is_one_column(self) -> None:
        left = [block(f"L{i}", 72, i * 20.0, 290, i * 20.0 + 10) for i in range(8)]
        right = [block("R", 320, 0, 540, 10)]
        assert detect_column_count(left + right, page_width=PAGE_WIDTH) == 1

    def test_zero_width_page_is_handled(self) -> None:
        assert detect_column_count([block("a", 0, 0, 1, 1)], page_width=0.0) == 1


class TestTwoColumnOrdering:
    def test_left_column_is_read_before_right(self) -> None:
        blocks = [
            block("R1", 320, 100, 540, 120),
            block("L1", 72, 100, 290, 120),
            block("R2", 320, 200, 540, 220),
            block("L2", 72, 200, 290, 220),
            block("L3", 72, 300, 290, 320),
            block("R3", 320, 300, 540, 320),
            block("L4", 72, 400, 290, 420),
            block("R4", 320, 400, 540, 420),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH)) == [
            "L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4",
        ]

    def test_rtl_flag_reverses_the_column_traversal(self) -> None:
        """The flag the thorn-nlp module already carried. LEDGER never sets
        it, but keeping it is what made that module reusable here at all."""
        blocks = [
            block("L1", 72, 100, 290, 120),
            block("R1", 320, 100, 540, 120),
            block("L2", 72, 200, 290, 220),
            block("R2", 320, 200, 540, 220),
            block("L3", 72, 300, 290, 320),
            block("R3", 320, 300, 540, 320),
            block("L4", 72, 400, 290, 420),
            block("R4", 320, 400, 540, 420),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH, rtl=True))[0] == "R1"

    def test_a_full_width_heading_stays_ahead_of_both_columns(self) -> None:
        blocks = [
            block("L1", 72, 100, 290, 120),
            block("R1", 320, 100, 540, 120),
            block("L2", 72, 200, 290, 220),
            block("R2", 320, 200, 540, 220),
            block("L3", 72, 300, 290, 320),
            block("R3", 320, 300, 540, 320),
            block("L4", 72, 400, 290, 420),
            block("R4", 320, 400, 540, 420),
            block("Wide heading", 72, 50, 540, 70, kind=BlockType.SECTION_HEADER),
        ]
        assert texts(order_blocks(blocks, page_width=PAGE_WIDTH))[0] == "Wide heading"


class TestDegenerateInput:
    def test_blocks_without_boxes_do_not_crash(self) -> None:
        blocks = [
            RawBlock(type=BlockType.TEXT, text="no box"),
            block("boxed", 72, 100, 540, 120),
        ]
        assert len(order_blocks(blocks, page_width=PAGE_WIDTH)) == 2

    def test_empty_page(self) -> None:
        assert order_blocks([], page_width=PAGE_WIDTH) == []
