"""Tests for contract assembly, driven by a fake backend.

The fake is the point as much as the assertions: it proves a backend can be
substituted with no model, no torch, and no GPU, which is both the brief's
"replaceable backend" requirement and the reason this suite runs in a second.
"""

from __future__ import annotations

import pytest

from app.backends.base import (
    LayoutBackend,
    RawBBox,
    RawBlock,
    RawCell,
    RawPage,
    RawParse,
    RawTable,
)
from app.core.settings import Settings
from app.processing.assemble import assemble
from ledger_doc_contract.v1.enums import BlockType, ExtractionSource


class FakeBackend:
    """A backend that replays a prepared parse."""

    name = "fake"

    def __init__(self, parse: RawParse) -> None:
        self._parse = parse

    @property
    def version(self) -> str:
        return "0.0.1-fake"

    @property
    def model_versions(self) -> dict[str, str]:
        return {"layout": "fake-layout", "table": "fake-table"}

    def is_ready(self) -> bool:
        return True

    def warm(self) -> None:
        return None

    def parse(self, data: bytes, *, force_ocr: bool = False) -> RawParse:
        return self._parse


def bbox(x0: float, y0: float, x1: float, y1: float) -> RawBBox:
    return RawBBox(x0=x0, y0=y0, x1=x1, y1=y1)


def statement_table() -> RawTable:
    rows = [["", "2019"], ["Revenue", "1,234"], ["Cost", "(567)"]]
    cells = [
        RawCell(row=r, col=c, text=text, is_header=(r == 0 or c == 0))
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return RawTable(
        n_rows=3,
        n_cols=2,
        cells=tuple(cells),
        header_rows=1,
        header_cols=1,
        caption="(in millions)",
        confidence=0.88,
    )


def report_parse() -> RawParse:
    return RawParse(
        pages=(
            RawPage(
                page_number=1,
                width=612.0,
                height=792.0,
                extraction_source=ExtractionSource.TEXT_LAYER,
                blocks=(
                    RawBlock(
                        type=BlockType.TITLE,
                        text="Annual Report 2019",
                        bbox=bbox(72, 60, 540, 90),
                        font_size=20.0,
                    ),
                    RawBlock(
                        type=BlockType.SECTION_HEADER,
                        text="12. Income Taxes",
                        bbox=bbox(72, 120, 540, 140),
                        font_size=14.0,
                    ),
                    RawBlock(
                        type=BlockType.TABLE,
                        text="",
                        bbox=bbox(72, 160, 540, 260),
                        table=statement_table(),
                    ),
                    RawBlock(
                        type=BlockType.PAGE_FOOTER,
                        text="Page 1",
                        bbox=bbox(290, 760, 330, 772),
                        font_size=9.0,
                    ),
                ),
            ),
        )
    )


def build(parse: RawParse):
    backend = FakeBackend(parse)
    return assemble(
        parse,
        document_id="doc_test",
        filename="test.pdf",
        content_sha256="a" * 64,
        backend_name=backend.name,
        backend_version=backend.version,
        model_versions=backend.model_versions,
        duration_ms=12,
        settings=Settings(),
    )


class TestBackendSeam:
    def test_a_fake_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeBackend(RawParse()), LayoutBackend)


class TestTableWiring:
    def test_the_table_is_hoisted_to_document_level(self) -> None:
        document = build(report_parse())
        assert len(document.tables) == 1
        assert document.tables[0].table_id == "t000"

    def test_the_block_and_the_table_are_joined_by_id(self) -> None:
        """Both access paths must reach the same table."""
        document = build(report_parse())
        table_block = next(
            b for b in document.iter_blocks() if b.type is BlockType.TABLE
        )
        assert table_block.table_id == "t000"

        table = document.table(table_block.table_id)
        assert table is not None
        assert table.block_id == table_block.block_id

    def test_the_table_inherits_the_section_it_sits_in(self) -> None:
        document = build(report_parse())
        table = document.tables[0]
        assert table.section_id is not None
        assert document.section_path(table.section_id).endswith("12. Income Taxes")

    def test_cells_are_parsed_through_assembly(self) -> None:
        document = build(report_parse())
        table = document.tables[0]
        cell = table.cell_at(2, 1)
        assert cell is not None and cell.value is not None
        assert cell.value.num == pytest.approx(-567.0)
        assert cell.value.scaled == pytest.approx(-567e6)


class TestBlockAssembly:
    def test_ids_are_positional_and_sort_with_the_document(self) -> None:
        document = build(report_parse())
        ids = [b.block_id for b in document.pages[0].blocks]
        assert ids == sorted(ids)
        assert ids[0] == "p0001_b000"

    def test_furniture_is_emitted_but_kept_out_of_the_order(self) -> None:
        document = build(report_parse())
        page = document.pages[0]
        footer = next(b for b in page.blocks if b.type is BlockType.PAGE_FOOTER)
        assert footer.block_id not in page.reading_order

    def test_headings_carry_their_level(self) -> None:
        document = build(report_parse())
        title = document.pages[0].blocks[0]
        assert title.type is BlockType.TITLE
        assert title.heading_level == 1

    def test_non_headings_have_no_level(self) -> None:
        document = build(report_parse())
        table_block = next(
            b for b in document.pages[0].blocks if b.type is BlockType.TABLE
        )
        assert table_block.heading_level is None


class TestWarnings:
    def test_a_document_with_no_headings_says_so(self) -> None:
        """Retrieval citations lose their section label, which is worth
        flagging rather than leaving the consumer to notice."""
        parse = RawParse(
            pages=(
                RawPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    blocks=(
                        RawBlock(
                            type=BlockType.TEXT,
                            text="Body only.",
                            bbox=bbox(72, 100, 540, 120),
                        ),
                    ),
                ),
            )
        )
        codes = {w.code.value for w in build(parse).processing.warnings}
        assert "NO_HEADINGS_DETECTED" in codes

    def test_low_ocr_confidence_is_flagged(self) -> None:
        parse = RawParse(
            pages=(
                RawPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    extraction_source=ExtractionSource.OCR,
                    ocr_confidence=0.42,
                    blocks=(
                        RawBlock(
                            type=BlockType.TEXT,
                            text="blurry",
                            bbox=bbox(72, 100, 540, 120),
                        ),
                    ),
                ),
            )
        )
        document = build(parse)
        assert document.pages[0].low_confidence is True
        codes = {w.code.value for w in document.processing.warnings}
        assert "LOW_OCR_CONFIDENCE" in codes

    def test_backend_warnings_are_carried_through(self) -> None:
        parse = RawParse(
            pages=(RawPage(page_number=1, width=612.0, height=792.0),),
            warnings=(("SCANNED_PAGE_OCR_FALLBACK", "no text layer", 1),),
        )
        warnings = build(parse).processing.warnings
        assert warnings[0].code.value == "SCANNED_PAGE_OCR_FALLBACK"
        assert warnings[0].page == 1

    def test_an_unknown_warning_code_is_dropped_not_fatal(self) -> None:
        parse = RawParse(
            pages=(RawPage(page_number=1, width=612.0, height=792.0),),
            warnings=(("SOMETHING_NEW", "from a future backend", 1),),
        )
        assert build(parse).processing.warnings == ()


class TestProvenance:
    def test_the_run_is_described(self) -> None:
        info = build(report_parse()).processing
        assert info.backend == "fake"
        assert info.backend_version == "0.0.1-fake"
        assert info.model_versions == {"layout": "fake-layout", "table": "fake-table"}
        assert info.duration_ms == 12
        assert info.processed_at.endswith("+00:00")
