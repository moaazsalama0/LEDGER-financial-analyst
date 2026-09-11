"""The Docling backend against real models.

Everything here needs weights on disk and takes minutes, which is why it lives
behind the ``heavy`` marker rather than in the fast suite. What it checks is
what unit tests over fakes structurally cannot: that the real library's output
lands in the contract's coordinate system, that the typography probe finds the
glyphs inside the boxes the real model drew, and that two runs over the same
bytes agree.

Point ``DOC_TEST_PDF`` at a real financial report to add the table assertions
a synthetic fixture cannot support — a hand-built PDF has no ruled lines for
TableFormer to find.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.backends.base import LayoutBackend, RawParse
from app.backends.docling_backend import DoclingBackend
from app.core.settings import Settings
from app.processing.assemble import assemble
from ledger_doc_contract.v1.enums import BlockType, ExtractionSource
from ledger_doc_contract.v1.models import ProcessedDocument
from tests.fixtures.make_pdf import simple_report_pdf

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def backend() -> DoclingBackend:
    """One warmed backend for the whole module — loading is the slow part."""
    instance = DoclingBackend()
    instance.warm()
    return instance


@pytest.fixture(scope="module")
def parse(backend: DoclingBackend) -> RawParse:
    return backend.parse(simple_report_pdf())


def build(parse: RawParse, backend: DoclingBackend) -> ProcessedDocument:
    return assemble(
        parse,
        document_id="doc_heavy",
        filename="fixture.pdf",
        content_sha256="a" * 64,
        backend_name=backend.name,
        backend_version=backend.version,
        model_versions=backend.model_versions,
        duration_ms=0,
        settings=Settings(),
    )


class TestLifecycle:
    def test_it_satisfies_the_backend_protocol(self, backend: DoclingBackend) -> None:
        assert isinstance(backend, LayoutBackend)

    def test_a_warmed_backend_reports_ready(self, backend: DoclingBackend) -> None:
        """`GET /ready` must not invite traffic into a service that would then
        spend a minute fetching weights."""
        assert backend.is_ready() is True

    def test_the_models_that_ran_are_named(self, backend: DoclingBackend) -> None:
        """A backend comparison is only meaningful if each result says what
        actually ran."""
        versions = backend.model_versions
        assert "docling-layout" in versions["layout"], (
            "the layout model must be named exactly, not described generically"
        )
        assert versions["table"].startswith("tableformer-")
        assert "docling" in backend.version


class TestPages:
    def test_every_page_of_the_pdf_is_reported(self, parse: RawParse) -> None:
        assert [p.page_number for p in parse.pages] == [1, 2]

    def test_page_geometry_comes_from_the_pdf(self, parse: RawParse) -> None:
        assert (parse.pages[0].width, parse.pages[0].height) == (612.0, 792.0)

    def test_a_born_digital_page_is_read_from_its_text_layer(
        self, parse: RawParse
    ) -> None:
        """Not a shortcut. Force-OCR'ing a page whose text layer is already
        character-exact can only add errors, and they would land in the very
        figures the answer validator checks."""
        assert parse.pages[0].extraction_source is ExtractionSource.TEXT_LAYER

    def test_ocr_confidence_is_absent_rather_than_invented(
        self, parse: RawParse
    ) -> None:
        """'Not measured' and 'measured at 1.0' are different facts, and a
        quality gate that conflates them is useless."""
        assert parse.pages[0].ocr_confidence is None

    def test_pages_carry_blocks(self, parse: RawParse) -> None:
        assert all(page.blocks for page in parse.pages)


class TestGeometry:
    def test_every_box_lies_within_its_page(self, parse: RawParse) -> None:
        for page in parse.pages:
            for block in page.blocks:
                if block.bbox is None:
                    continue
                assert 0 <= block.bbox.x0 <= block.bbox.x1 <= page.width + 1
                assert 0 <= block.bbox.y0 <= block.bbox.y1 <= page.height + 1

    def test_the_origin_is_top_left(self, parse: RawParse) -> None:
        """The title sits near the top of page 1, so in the contract's
        convention it must have a small y0. A missed flip would put it near
        792 instead, and every citation highlight on the page's mirror."""
        page = parse.pages[0]
        title = min(
            (b for b in page.blocks if b.bbox is not None and b.text),
            key=lambda b: b.bbox.y0,  # type: ignore[union-attr]
        )
        assert title.bbox is not None
        assert title.bbox.y0 < page.height / 2
        assert "Annual Report" in title.text


class TestTypography:
    def test_blocks_are_measured_from_the_text_layer(self, parse: RawParse) -> None:
        """Docling reports no type size. Without the probe finding glyphs
        inside the boxes the model drew, every heading lands at one level and
        the section tree flattens into a list."""
        measured = [
            b for p in parse.pages for b in p.blocks if b.font_size is not None
        ]
        assert measured, "no block was measured; the probe found no glyphs"

    def test_the_title_is_measured_larger_than_the_body(
        self, parse: RawParse
    ) -> None:
        sizes = {
            b.text: b.font_size
            for p in parse.pages
            for b in p.blocks
            if b.font_size is not None
        }
        title = next((s for t, s in sizes.items() if "Annual Report" in t), None)
        body = next((s for t, s in sizes.items() if "Total revenue" in t), None)
        if title is None or body is None:
            pytest.skip("the layout model split the fixture differently")
        assert title > body


class TestContractAssembly:
    def test_the_parse_assembles_into_a_valid_document(
        self, parse: RawParse, backend: DoclingBackend
    ) -> None:
        document = build(parse, backend)
        assert ProcessedDocument.model_validate(document.model_dump()) == document

    def test_the_document_reports_the_models_that_produced_it(
        self, parse: RawParse, backend: DoclingBackend
    ) -> None:
        document = build(parse, backend)
        assert document.processing.backend == "docling"
        assert document.processing.model_versions["table"].startswith("tableformer-")

    def test_headings_produce_a_section_tree(
        self, parse: RawParse, backend: DoclingBackend
    ) -> None:
        document = build(parse, backend)
        if not document.sections:
            pytest.skip("the layout model found no heading in the fixture")
        assert all(s.level >= 1 for s in document.sections)
        assert document.section_path(document.sections[0].section_id)

    def test_every_block_carries_the_metadata_a_chunk_needs(
        self, parse: RawParse, backend: DoclingBackend
    ) -> None:
        """document_id / page / section / content_type — the four fields the
        brief requires on every retrieval chunk."""
        document = build(parse, backend)
        assert document.document_id
        for block in document.iter_blocks():
            assert block.page_number >= 1
            assert isinstance(block.type, BlockType)


class TestDeterminism:
    def test_two_runs_over_the_same_bytes_agree(
        self, backend: DoclingBackend
    ) -> None:
        """The cache and the golden-file tests both rest on this."""
        data = simple_report_pdf()
        first = build(backend.parse(data), backend)
        second = build(backend.parse(data), backend)
        assert first.to_stable_dict() == second.to_stable_dict()


@pytest.fixture(scope="module")
def real_document(backend: DoclingBackend) -> ProcessedDocument:
    """A real report, parsed once for the whole module.

    Module scope rather than class scope: pytest deprecates a class-scoped
    fixture written as an instance method, and parsing a real document is far
    too slow to repeat per test.
    """
    path = os.environ.get("DOC_TEST_PDF")
    if not path or not Path(path).is_file():
        pytest.skip("set DOC_TEST_PDF to a real PDF to run these")
    return build(backend.parse(Path(path).read_bytes()), backend)


class TestRealReport:
    """Assertions a synthetic fixture cannot support.

    A hand-built PDF has no ruled lines, so TableFormer has nothing to find in
    it. These run only against a real document.
    """

    def test_tables_are_found_and_have_cells(
        self, real_document: ProcessedDocument
    ) -> None:
        assert real_document.tables, "no table was recovered"
        assert all(t.cells for t in real_document.tables)

    def test_a_table_is_reachable_from_both_access_paths(
        self, real_document: ProcessedDocument
    ) -> None:
        table = real_document.tables[0]
        block = real_document.block(table.block_id)
        assert block is not None and block.table_id == table.table_id

    def test_tables_serialise_to_markdown_for_embedding(
        self, real_document: ProcessedDocument
    ) -> None:
        assert all(t.markdown.startswith("|") for t in real_document.tables)

    def test_page_furniture_is_kept_out_of_the_reading_order(
        self, real_document: ProcessedDocument
    ) -> None:
        for page in real_document.pages:
            for block in page.blocks:
                if block.type.is_page_furniture:
                    assert block.block_id not in page.reading_order
