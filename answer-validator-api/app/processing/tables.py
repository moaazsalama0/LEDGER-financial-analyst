"""Turning a backend's raw table grid into the contract's table.

Three things happen here that a layout model does not do:

* every cell is run through the finance parser, so a downstream calculator
  gets numbers rather than strings;
* the table's scale and currency are read from its caption once and applied to
  all of its cells, because reports declare magnitude in the caption and then
  print bare numbers;
* header labels are flattened into per-row and per-column arrays, so a caller
  can resolve ``(row_label, column_label)`` without reimplementing span logic.

The markdown serialisation exists because embedding a table needs a linear
string, and a hand-rolled one per consumer would drift.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.backends.base import RawTable
from app.processing.finance import parse_value
from app.processing.scale import detect_scale
from ledger_doc_contract.v1.models import BBox, Cell, Table

logger = logging.getLogger(__name__)


def _bbox(raw: object | None) -> BBox | None:
    if raw is None:
        return None
    return BBox(x0=raw.x0, y0=raw.y0, x1=raw.x1, y1=raw.y1)  # type: ignore[attr-defined]


def _grid(table: RawTable) -> list[list[str]]:
    """Expand the cell list into a dense text grid, honouring spans.

    A cell covering two columns writes its text into the first and leaves the
    covered position empty, matching how a reader sees it, rather than
    duplicating the value into both.
    """
    grid = [["" for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    for cell in table.cells:
        if 0 <= cell.row < table.n_rows and 0 <= cell.col < table.n_cols:
            grid[cell.row][cell.col] = cell.text.strip()
    return grid


def _column_headers(table: RawTable, grid: list[list[str]]) -> tuple[str, ...]:
    """Flatten the header rows into one label per column.

    With several header rows, labels are joined top to bottom so a column
    under 'Year ended December 31' / '2019' becomes a single distinguishable
    label rather than an ambiguous '2019'.
    """
    if table.n_cols == 0:
        return ()
    rows = table.header_rows or (1 if table.n_rows else 0)
    labels: list[str] = []
    for col in range(table.n_cols):
        parts = [grid[row][col] for row in range(min(rows, table.n_rows)) if grid[row][col]]
        labels.append(" ".join(parts).strip())
    return tuple(labels)


def _row_headers(table: RawTable, grid: list[list[str]]) -> tuple[str, ...]:
    """One label per body row, taken from the header columns."""
    if table.n_rows == 0:
        return ()
    cols = table.header_cols or 1
    start = table.header_rows
    labels: list[str] = []
    for row in range(start, table.n_rows):
        parts = [grid[row][col] for col in range(min(cols, table.n_cols)) if grid[row][col]]
        labels.append(" ".join(parts).strip())
    return tuple(labels)


def to_markdown(grid: list[list[str]], *, header_rows: int) -> str:
    """Render the grid as a GitHub-flavoured table, for embedding.

    Pipes inside cell text are escaped so a value containing one cannot break
    the row structure for whoever parses the markdown later.
    """
    if not grid or not grid[0]:
        return ""

    def row_to_line(cells: list[str]) -> str:
        escaped = [c.replace("|", r"\|").replace("\n", " ") for c in cells]
        return "| " + " | ".join(escaped) + " |"

    width = len(grid[0])
    lines: list[str] = []

    if header_rows > 0:
        for row in grid[:header_rows]:
            lines.append(row_to_line(row))
        lines.append("|" + "|".join(["---"] * width) + "|")
        body = grid[header_rows:]
    else:
        # Markdown requires a header row; an empty one keeps the table valid
        # without inventing labels that were not on the page.
        lines.append(row_to_line([""] * width))
        lines.append("|" + "|".join(["---"] * width) + "|")
        body = grid

    for row in body:
        lines.append(row_to_line(row))

    return "\n".join(lines)


def _header_texts(raw: RawTable, grid: list[list[str]]) -> list[str]:
    """Header cell text, plus the flattened per-column labels.

    Both are included because a magnitude can be printed in a header cell of
    its own — ``(In millions)`` — or tacked onto a spanning label that only
    appears once flattened: ``Year Ended December 31, 2019 (In millions)``.
    """
    texts = [cell.text for cell in raw.cells if cell.is_header and cell.text.strip()]
    texts.extend(label for label in _column_headers(raw, grid) if label.strip())
    return texts


def _row_stem_texts(raw: RawTable) -> list[str]:
    """Column-zero text, where a row-label stem states the units."""
    return [cell.text for cell in raw.cells if cell.col == 0 and cell.text.strip()]


def build_table(
    raw: RawTable,
    *,
    table_id: str,
    block_id: str,
    page_number: int,
    section_id: str | None,
    bbox: BBox | None,
    nearby_text: str | None = None,
    preceding_texts: Sequence[str] = (),
) -> Table:
    """Assemble the contract table, parsing every cell.

    Magnitude is resolved from four places in precedence order — caption,
    header cells, row-label stems, then the blocks above the table — because
    the Phase 2 corpus showed the caption is empty on nearly every table this
    backend produces, while headers and row stems carry the phrase in
    hundreds of cases. See :mod:`app.processing.scale`.

    ``preceding_texts`` is nearest-first. ``nearby_text`` is the older
    single-string form of the same thing and is appended after it.
    """
    grid = _grid(raw)

    preceding = [text for text in preceding_texts if text]
    if nearby_text:
        preceding.append(nearby_text)

    detection = detect_scale(
        caption=raw.caption,
        headers=_header_texts(raw, grid),
        row_stems=_row_stem_texts(raw),
        preceding=preceding,
    )
    units = detection.units

    if detection.ambiguous:
        # Deliberately not silent, and deliberately not fatal: the precedence
        # rule still returns one answer, but a table whose own headers
        # disagree about their units is worth a person's attention.
        logger.warning(
            "table %s on page %s: conflicting magnitude in %s; %s",
            table_id,
            page_number,
            detection.source.value if detection.source else "?",
            " vs ".join(
                sorted({f"{c.scale:g} ({c.label!r})" for c in detection.candidates})
            ),
        )

    cells: list[Cell] = []
    for cell in raw.cells:
        cells.append(
            Cell(
                cell_id=f"{table_id}_r{cell.row}_c{cell.col}",
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                is_header=cell.is_header,
                text=cell.text,
                bbox=_bbox(cell.bbox),
                # Header cells hold labels, not figures. Parsing them would
                # turn a year column header like "2019" into a measurement.
                value=None if cell.is_header else parse_value(cell.text, units=units),
            )
        )

    return Table(
        table_id=table_id,
        block_id=block_id,
        page_number=page_number,
        section_id=section_id,
        bbox=bbox,
        caption=raw.caption,
        caption_bbox=_bbox(raw.caption_bbox),
        n_rows=raw.n_rows,
        n_cols=raw.n_cols,
        header_rows=raw.header_rows,
        header_cols=raw.header_cols,
        column_headers=_column_headers(raw, grid),
        row_headers=_row_headers(raw, grid),
        units=units,
        cells=tuple(cells),
        markdown=to_markdown(grid, header_rows=raw.header_rows),
        confidence=raw.confidence,
    )


__all__ = ["build_table", "to_markdown"]
