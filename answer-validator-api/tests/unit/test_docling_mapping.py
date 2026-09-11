"""Tests for the Docling → contract mapping, with no model and no docling.

The mapping functions take duck-typed arguments on purpose. That is what lets
the part of the backend most likely to be wrong — label translation, the
coordinate flip, span arithmetic, header detection — be tested on CPU in
milliseconds, while the part that needs a GPU is exercised separately and
marked heavy.

The stakes are concrete. A missed coordinate flip puts every citation
highlight on the mirror image of the right paragraph, and no unit test of the
model itself would catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.backends.docling_backend import (
    header_extent,
    is_bottom_left,
    map_label,
    to_raw_bbox,
    to_raw_table,
)
from ledger_doc_contract.v1.enums import BlockType

PAGE_HEIGHT = 792.0


@dataclass
class FakeBBox:
    """Shaped like ``docling_core.types.doc.BoundingBox``."""

    l: float  # noqa: E741 - the attribute name docling uses
    t: float
    r: float
    b: float
    coord_origin: str | None = "BOTTOMLEFT"


@dataclass
class FakeLabel:
    value: str


@dataclass
class FakeCell:
    """Shaped like ``docling_core.types.doc.document.TableCell``."""

    start_row_offset_idx: int
    start_col_offset_idx: int
    text: str
    end_row_offset_idx: int | None = None
    end_col_offset_idx: int | None = None
    column_header: bool = False
    row_header: bool = False
    bbox: FakeBBox | None = None

    def __post_init__(self) -> None:
        if self.end_row_offset_idx is None:
            self.end_row_offset_idx = self.start_row_offset_idx + 1
        if self.end_col_offset_idx is None:
            self.end_col_offset_idx = self.start_col_offset_idx + 1


@dataclass
class FakeTableData:
    num_rows: int
    num_cols: int
    table_cells: list[FakeCell] = field(default_factory=list)


def statement_cells() -> list[FakeCell]:
    """A three-column statement: a header row and a label column."""
    rows = [
        ["", "2019", "2018"],
        ["Current federal", "1,234", "(567)"],
        ["Deferred", "890", "—"],
    ]
    cells: list[FakeCell] = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cells.append(
                FakeCell(
                    start_row_offset_idx=r,
                    start_col_offset_idx=c,
                    text=text,
                    column_header=(r == 0),
                    row_header=(c == 0),
                )
            )
    return cells


class TestLabelMapping:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("title", BlockType.TITLE),
            ("section_header", BlockType.SECTION_HEADER),
            ("text", BlockType.TEXT),
            ("list_item", BlockType.LIST_ITEM),
            ("table", BlockType.TABLE),
            ("caption", BlockType.CAPTION),
            ("footnote", BlockType.FOOTNOTE),
            ("picture", BlockType.FIGURE),
            ("formula", BlockType.FORMULA),
            ("page_header", BlockType.PAGE_HEADER),
            ("page_footer", BlockType.PAGE_FOOTER),
        ],
    )
    def test_every_label_we_care_about_maps(
        self, label: str, expected: BlockType
    ) -> None:
        assert map_label(FakeLabel(label)) is expected

    def test_a_bare_string_label_maps_too(self) -> None:
        """Docling hands over an enum; a dict-shaped document hands over a
        string. Both have to work."""
        assert map_label("section_header") is BlockType.SECTION_HEADER

    def test_an_unknown_label_becomes_text_not_a_dropped_block(self) -> None:
        """A docling release that adds a label should cost a slightly wrong
        content_type, never the paragraph. A silently missing paragraph is an
        unanswerable question."""
        assert map_label(FakeLabel("some_future_label")) is BlockType.TEXT

    def test_a_missing_label_becomes_text(self) -> None:
        assert map_label(None) is BlockType.TEXT

    def test_page_furniture_is_recognised_as_such(self) -> None:
        """The whole point of the layout model over a position heuristic:
        running headers are labelled, not guessed at from a band."""
        assert map_label(FakeLabel("page_header")).is_page_furniture is True
        assert map_label(FakeLabel("page_footer")).is_page_furniture is True

    def test_headings_are_recognised_as_such(self) -> None:
        assert map_label(FakeLabel("title")).is_heading is True
        assert map_label(FakeLabel("section_header")).is_heading is True


class TestGeometry:
    def test_a_bottom_left_box_is_flipped(self) -> None:
        """PDF space counts up from the bottom; the contract counts down from
        the top. Miss this and every highlight lands on the page's mirror."""
        raw = to_raw_bbox(
            FakeBBox(l=72, t=700, r=540, b=680), page_height=PAGE_HEIGHT
        )
        assert raw is not None
        assert raw.y0 == pytest.approx(92.0)  # 792 - 700
        assert raw.y1 == pytest.approx(112.0)  # 792 - 680
        assert (raw.x0, raw.x1) == (72.0, 540.0)

    def test_a_top_left_box_is_left_alone(self) -> None:
        raw = to_raw_bbox(
            FakeBBox(l=72, t=92, r=540, b=112, coord_origin="TOPLEFT"),
            page_height=PAGE_HEIGHT,
        )
        assert raw is not None
        assert (raw.y0, raw.y1) == (92.0, 112.0)

    def test_the_box_always_comes_back_ordered(self) -> None:
        """y0 < y1 and x0 < x1 whatever order the source reported, so no
        consumer has to defend against a negative-height box."""
        raw = to_raw_bbox(
            FakeBBox(l=540, t=680, r=72, b=700), page_height=PAGE_HEIGHT
        )
        assert raw is not None
        assert raw.x0 < raw.x1 and raw.y0 < raw.y1

    def test_a_scale_is_applied_after_the_flip(self) -> None:
        """The flip happens in the source's space, the scale converts to PDF
        points. Doing them the other way round mislocates every box."""
        raw = to_raw_bbox(
            FakeBBox(l=36, t=350, r=270, b=340),
            page_height=396.0,
            scale_x=2.0,
            scale_y=2.0,
        )
        assert raw is not None
        assert raw.y0 == pytest.approx(92.0)  # (396 - 350) * 2
        assert raw.x0 == pytest.approx(72.0)

    def test_an_origin_free_box_is_treated_as_top_left(self) -> None:
        raw = to_raw_bbox(
            FakeBBox(l=72, t=92, r=540, b=112, coord_origin=None),
            page_height=PAGE_HEIGHT,
        )
        assert raw is not None and raw.y0 == 92.0

    def test_a_missing_box_is_none_not_a_crash(self) -> None:
        assert to_raw_bbox(None, page_height=PAGE_HEIGHT) is None

    def test_an_unusable_box_is_none_not_a_crash(self) -> None:
        """A block we cannot place is worth losing the geometry of; it is not
        worth losing the document over."""
        assert to_raw_bbox(object(), page_height=PAGE_HEIGHT) is None

    def test_origin_detection(self) -> None:
        assert is_bottom_left(FakeBBox(0, 0, 1, 1, "BOTTOMLEFT")) is True
        assert is_bottom_left(FakeBBox(0, 0, 1, 1, "TOPLEFT")) is False
        assert is_bottom_left(FakeBBox(0, 0, 1, 1, None)) is False


class TestHeaderExtent:
    def test_leading_header_row_and_column_are_found(self) -> None:
        rows, cols = header_extent(statement_cells(), n_rows=3, n_cols=3)
        assert (rows, cols) == (1, 1)

    def test_two_header_rows_are_both_counted(self) -> None:
        cells = [
            FakeCell(0, c, "Year ended", column_header=True) for c in range(2)
        ] + [FakeCell(1, c, "2019", column_header=True) for c in range(2)]
        cells += [FakeCell(2, c, "1,234") for c in range(2)]
        assert header_extent(cells, n_rows=3, n_cols=2)[0] == 2

    def test_counting_stops_at_the_first_non_header_row(self) -> None:
        """TableFormer sometimes flags a stray body cell as a header. Taking
        the leading run rather than the total keeps one such cell from
        declaring half the statement to be headings."""
        cells = [FakeCell(0, c, "h", column_header=True) for c in range(2)]
        cells += [FakeCell(1, c, "body") for c in range(2)]
        cells += [FakeCell(2, 0, "stray", column_header=True), FakeCell(2, 1, "x")]
        assert header_extent(cells, n_rows=3, n_cols=2)[0] == 1

    def test_a_partly_flagged_row_is_not_a_header_row(self) -> None:
        cells = [
            FakeCell(0, 0, "a", column_header=True),
            FakeCell(0, 1, "b", column_header=False),
        ]
        assert header_extent(cells, n_rows=1, n_cols=2)[0] == 0

    def test_a_table_with_no_flags_reports_none(self) -> None:
        cells = [FakeCell(r, c, "x") for r in range(2) for c in range(2)]
        assert header_extent(cells, n_rows=2, n_cols=2) == (0, 0)


class TestTableConversion:
    def test_the_grid_survives_conversion(self) -> None:
        table = to_raw_table(
            FakeTableData(3, 3, statement_cells()), page_height=PAGE_HEIGHT
        )
        assert (table.n_rows, table.n_cols) == (3, 3)
        assert len(table.cells) == 9
        assert (table.header_rows, table.header_cols) == (1, 1)

    def test_cells_land_at_their_offsets(self) -> None:
        table = to_raw_table(
            FakeTableData(3, 3, statement_cells()), page_height=PAGE_HEIGHT
        )
        cell = next(c for c in table.cells if c.row == 1 and c.col == 2)
        assert cell.text == "(567)", "the printed string reaches the finance parser"

    def test_spans_come_from_the_offsets(self) -> None:
        """A merged 'Year ended December 31' over three year columns has to
        keep its width, or the year labels shift by two columns."""
        cells = [
            FakeCell(
                0, 1, "Year ended December 31", end_col_offset_idx=4, column_header=True
            )
        ]
        table = to_raw_table(FakeTableData(1, 4, cells), page_height=PAGE_HEIGHT)
        assert table.cells[0].col_span == 3

    def test_a_degenerate_span_is_corrected_to_one(self) -> None:
        """A zero-width cell would erase a column from the grid."""
        cells = [FakeCell(0, 0, "x", end_col_offset_idx=0, end_row_offset_idx=0)]
        table = to_raw_table(FakeTableData(1, 1, cells), page_height=PAGE_HEIGHT)
        assert (table.cells[0].row_span, table.cells[0].col_span) == (1, 1)

    def test_dimensions_are_derived_when_not_reported(self) -> None:
        table = to_raw_table(
            FakeTableData(0, 0, statement_cells()), page_height=PAGE_HEIGHT
        )
        assert (table.n_rows, table.n_cols) == (3, 3)

    def test_cells_are_emitted_in_a_stable_order(self) -> None:
        """The contract promises a byte-identical body for the same PDF. A
        cell list in whatever order the model happened to emit would break
        that quietly, and only for some documents."""
        shuffled = list(reversed(statement_cells()))
        table = to_raw_table(FakeTableData(3, 3, shuffled), page_height=PAGE_HEIGHT)
        assert [(c.row, c.col) for c in table.cells] == sorted(
            (c.row, c.col) for c in table.cells
        )

    def test_cell_boxes_are_flipped_like_any_other(self) -> None:
        cells = [FakeCell(0, 0, "x", bbox=FakeBBox(l=72, t=700, r=140, b=690))]
        table = to_raw_table(FakeTableData(1, 1, cells), page_height=PAGE_HEIGHT)
        assert table.cells[0].bbox is not None
        assert table.cells[0].bbox.y0 == pytest.approx(92.0)

    def test_a_caption_is_carried_through(self) -> None:
        """The caption is where '(in millions)' lives, and that multiplier is
        the difference between 1,234 and 1.234 billion."""
        table = to_raw_table(
            FakeTableData(3, 3, statement_cells()),
            caption="(in millions)",
            page_height=PAGE_HEIGHT,
        )
        assert table.caption == "(in millions)"

    def test_an_empty_caption_is_none(self) -> None:
        table = to_raw_table(
            FakeTableData(1, 1, [FakeCell(0, 0, "x")]),
            caption="   ",
            page_height=PAGE_HEIGHT,
        )
        assert table.caption is None

    def test_an_empty_table_does_not_crash(self) -> None:
        table = to_raw_table(FakeTableData(0, 0, []), page_height=PAGE_HEIGHT)
        assert (table.n_rows, table.n_cols, table.cells) == (0, 0, ())


class TestEndToEndThroughAssembly:
    def test_a_converted_table_parses_its_figures(self) -> None:
        """The point of the whole mapping: what Docling produced reaches the
        finance parser and comes out as signed, scaled numbers."""
        from app.processing.tables import build_table

        raw = to_raw_table(
            FakeTableData(3, 3, statement_cells()),
            caption="(in millions)",
            page_height=PAGE_HEIGHT,
        )
        table = build_table(
            raw,
            table_id="t000",
            block_id="p0001_b000",
            page_number=1,
            section_id=None,
            bbox=None,
        )

        assert table.units.scale == pytest.approx(1e6)

        negative = table.cell_at(1, 2)
        assert negative is not None and negative.value is not None
        assert negative.value.num == pytest.approx(-567.0)
        assert negative.value.scaled == pytest.approx(-567e6)

        dash = table.cell_at(2, 2)
        assert dash is not None and dash.value is not None
        assert dash.value.is_dash is True and dash.value.num is None

        assert table.column_headers == ("", "2019", "2018")
        assert table.row_headers == ("Current federal", "Deferred")
