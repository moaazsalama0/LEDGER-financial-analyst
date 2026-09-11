"""Tests for table assembly.

Covers the three things that happen here and nowhere else: cells become
parsed numbers, the caption's scale reaches every cell, and header labels are
flattened so a caller can do a direct (row, column) lookup.
"""

from __future__ import annotations

import pytest

from app.backends.base import RawCell, RawTable
from app.processing.tables import build_table, to_markdown
from ledger_doc_contract.v1.enums import NumericUnit


def income_tax_table(caption: str | None = "(in millions)") -> RawTable:
    """A small statement table with a header row and a header column."""
    rows = [
        ["", "2019", "2018"],
        ["Current federal", "1,234", "(567)"],
        ["Deferred", "890", "—"],
    ]
    cells: list[RawCell] = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cells.append(
                RawCell(row=r, col=c, text=text, is_header=(r == 0 or c == 0))
            )
    return RawTable(
        n_rows=3,
        n_cols=3,
        cells=tuple(cells),
        header_rows=1,
        header_cols=1,
        caption=caption,
        confidence=0.91,
    )


def build(raw: RawTable, **kwargs):
    defaults = dict(
        table_id="t000",
        block_id="p0002_b007",
        page_number=2,
        section_id="s002",
        bbox=None,
    )
    defaults.update(kwargs)
    return build_table(raw, **defaults)


class TestCellParsing:
    def test_numeric_cells_are_parsed(self) -> None:
        table = build(income_tax_table())
        cell = table.cell_at(1, 1)
        assert cell is not None
        assert cell.text == "1,234", "the printed string survives"
        assert cell.value is not None
        assert cell.value.num == pytest.approx(1234.0)

    def test_parenthesised_negative_keeps_its_sign(self) -> None:
        table = build(income_tax_table())
        cell = table.cell_at(1, 2)
        assert cell is not None and cell.value is not None
        assert cell.value.num == pytest.approx(-567.0)
        assert cell.value.is_negative is True

    def test_dash_cell_is_nil_not_zero(self) -> None:
        table = build(income_tax_table())
        cell = table.cell_at(2, 2)
        assert cell is not None and cell.value is not None
        assert cell.value.is_dash is True
        assert cell.value.num is None

    def test_header_cells_are_not_parsed_as_figures(self) -> None:
        """A year column header is a label. Parsing '2019' as a measurement
        would put a spurious number into the calculator's reach."""
        table = build(income_tax_table())
        header = table.cell_at(0, 1)
        assert header is not None
        assert header.is_header is True
        assert header.value is None

    def test_row_label_is_not_numeric(self) -> None:
        table = build(income_tax_table())
        label = table.cell_at(1, 0)
        assert label is not None
        assert label.value is None

    def test_cell_ids_are_derived_from_position(self) -> None:
        table = build(income_tax_table())
        assert table.cell_at(1, 1).cell_id == "t000_r1_c1"


class TestScale:
    def test_caption_scale_reaches_every_cell(self) -> None:
        table = build(income_tax_table("(in millions)"))
        assert table.units.scale == pytest.approx(1e6)

        cell = table.cell_at(1, 1)
        assert cell.value.num == pytest.approx(1234.0), "as printed"
        assert cell.value.scaled == pytest.approx(1.234e9), "with the multiplier"

    def test_scale_is_read_from_nearby_text_when_there_is_no_caption(self) -> None:
        """Reports often print '(in millions)' as its own line above the
        table rather than as a caption."""
        table = build(income_tax_table(None), nearby_text="(in thousands)")
        assert table.units.scale == pytest.approx(1e3)

    def test_no_scale_declared_leaves_values_as_printed(self) -> None:
        table = build(income_tax_table(None))
        assert table.units.scale == pytest.approx(1.0)
        cell = table.cell_at(1, 1)
        assert cell.value.scaled == pytest.approx(cell.value.num)

    def test_currency_from_the_caption_applies_to_cells(self) -> None:
        table = build(income_tax_table("(in millions of $)"))
        assert table.units.currency == "USD"
        cell = table.cell_at(1, 1)
        assert cell.value.currency == "USD"
        assert cell.value.unit is NumericUnit.CURRENCY


class TestHeaderLabels:
    def test_column_headers_are_flattened(self) -> None:
        table = build(income_tax_table())
        assert table.column_headers == ("", "2019", "2018")

    def test_row_headers_skip_the_header_rows(self) -> None:
        table = build(income_tax_table())
        assert table.row_headers == ("Current federal", "Deferred")

    def test_labels_enable_a_direct_lookup(self) -> None:
        """The lookup search_tables() needs: (row label, column label) -> cell."""
        table = build(income_tax_table())
        col = table.column_headers.index("2019")
        row = table.row_headers.index("Deferred") + table.header_rows
        cell = table.cell_at(row, col)
        assert cell is not None
        assert cell.value.num == pytest.approx(890.0)

    def test_multi_row_headers_are_joined(self) -> None:
        raw = RawTable(
            n_rows=2,
            n_cols=2,
            header_rows=2,
            header_cols=0,
            cells=(
                RawCell(row=0, col=0, text="", is_header=True),
                RawCell(row=0, col=1, text="Year ended", is_header=True),
                RawCell(row=1, col=0, text="", is_header=True),
                RawCell(row=1, col=1, text="2019", is_header=True),
            ),
        )
        assert build(raw).column_headers[1] == "Year ended 2019"


class TestSpans:
    def test_cell_at_resolves_through_a_column_span(self) -> None:
        raw = RawTable(
            n_rows=1,
            n_cols=3,
            cells=(RawCell(row=0, col=0, text="spans two", col_span=2),),
        )
        table = build(raw)
        assert table.cell_at(0, 0) is table.cell_at(0, 1)
        assert table.cell_at(0, 2) is None

    def test_cell_at_resolves_through_a_row_span(self) -> None:
        raw = RawTable(
            n_rows=3,
            n_cols=1,
            cells=(RawCell(row=0, col=0, text="tall", row_span=2),),
        )
        table = build(raw)
        assert table.cell_at(1, 0) is table.cell_at(0, 0)
        assert table.cell_at(2, 0) is None


class TestMarkdown:
    def test_renders_a_header_and_separator(self) -> None:
        table = build(income_tax_table())
        lines = table.markdown.splitlines()
        assert lines[0] == "|  | 2019 | 2018 |"
        assert lines[1] == "|---|---|---|"
        assert lines[2] == "| Current federal | 1,234 | (567) |"

    def test_pipes_in_cell_text_are_escaped(self) -> None:
        grid = [["a|b", "c"]]
        assert r"a\|b" in to_markdown(grid, header_rows=1)

    def test_a_table_without_header_rows_still_renders_validly(self) -> None:
        grid = [["1", "2"], ["3", "4"]]
        lines = to_markdown(grid, header_rows=0).splitlines()
        # Markdown needs a header row; an empty one avoids inventing labels
        # that were not printed on the page.
        assert lines[0] == "|  |  |"
        assert lines[1] == "|---|---|"
        assert lines[2] == "| 1 | 2 |"

    def test_empty_grid_yields_empty_markdown(self) -> None:
        assert to_markdown([], header_rows=0) == ""


class TestProvenance:
    def test_table_carries_its_location(self) -> None:
        table = build(income_tax_table())
        assert table.table_id == "t000"
        assert table.block_id == "p0002_b007"
        assert table.page_number == 2
        assert table.section_id == "s002"
        assert table.confidence == pytest.approx(0.91)
