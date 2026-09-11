"""Guards on the published contract.

The retrieval service codes against this shape. These tests exist so that a
change which would break it fails here, loudly, rather than in another team's
integration run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ledger_doc_contract.v1 import CONTRACT_VERSION, ProcessedDocument
from tests.fixtures.make_pdf import simple_report_pdf, single_page_pdf

SCHEMA_PATH = Path("contracts/ledger_doc_contract/v1/schema.json")


def _properties(schema: dict) -> dict:
    return schema.get("properties", {})


class TestSchemaStability:
    def test_committed_schema_exists(self) -> None:
        assert SCHEMA_PATH.exists(), (
            "The committed schema is the record of what consumers rely on. "
            "Regenerate it with: python scripts/generate_schema.py"
        )

    def test_live_models_have_not_dropped_a_committed_field(self) -> None:
        """Within 1.x, fields may be added but never removed or retyped.

        Compared field by field rather than by whole-document equality so
        that an additive change passes and only a breaking one fails.
        """
        committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        live = ProcessedDocument.model_json_schema()

        committed_defs = committed.get("$defs", {})
        live_defs = live.get("$defs", {})

        missing_models = set(committed_defs) - set(live_defs)
        assert not missing_models, f"models removed from the contract: {missing_models}"

        for name, definition in committed_defs.items():
            was = set(_properties(definition))
            now = set(_properties(live_defs[name]))
            removed = was - now
            assert not removed, f"{name} lost field(s) {removed}; that breaks 1.x"

        removed_top = set(_properties(committed)) - set(_properties(live))
        assert not removed_top, f"ProcessedDocument lost field(s) {removed_top}"

    def test_contract_version_matches_the_committed_schema(self) -> None:
        committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        default = _properties(committed)["contract_version"].get("default")
        assert default == CONTRACT_VERSION

    def test_served_schema_matches_the_live_models(self, client: TestClient) -> None:
        """The endpoint is generated from the models, so it can never describe
        a different shape than the service returns."""
        served = client.get("/contract/schema").json()
        assert served == ProcessedDocument.model_json_schema()


class TestRoundTrip:
    def test_a_response_validates_back_into_the_model(
        self, client: TestClient
    ) -> None:
        """A consumer must be able to parse our output with the published
        models. This is what caught width/height being serialised."""
        body = client.post(
            "/process",
            files={"file": ("r.pdf", simple_report_pdf(), "application/pdf")},
        ).json()

        document = ProcessedDocument.model_validate(body)
        assert document.page_count == 2

    def test_round_trip_is_lossless(self, client: TestClient) -> None:
        body = client.post(
            "/process",
            files={"file": ("r.pdf", simple_report_pdf(), "application/pdf")},
        ).json()

        again = ProcessedDocument.model_validate(body).model_dump(mode="json")
        assert again == body


class TestDeterminism:
    def test_same_bytes_produce_an_identical_body(self, client: TestClient) -> None:
        """Two parses of one PDF must agree exactly, or the cache is unsafe
        and golden-file tests are meaningless."""
        data = simple_report_pdf()

        first = ProcessedDocument.model_validate(
            client.post(
                "/process", files={"file": ("r.pdf", data, "application/pdf")}
            ).json()
        )
        # A different id must not change anything else about the parse.
        second = ProcessedDocument.model_validate(
            client.post(
                "/process",
                files={"file": ("r.pdf", data, "application/pdf")},
                data={"document_id": "doc_other"},
            ).json()
        )

        left = first.to_stable_dict()
        right = second.to_stable_dict()
        left.pop("document_id")
        right.pop("document_id")
        assert left == right

    def test_stable_dict_excludes_run_specific_provenance(
        self, client: TestClient
    ) -> None:
        document = ProcessedDocument.model_validate(
            client.post(
                "/process",
                files={"file": ("r.pdf", single_page_pdf(), "application/pdf")},
            ).json()
        )
        assert "processing" not in document.to_stable_dict()

    def test_json_serialisation_is_byte_identical(self, client: TestClient) -> None:
        data = simple_report_pdf()
        bodies = []
        for _ in range(2):
            body = client.post(
                "/process", files={"file": ("r.pdf", data, "application/pdf")}
            ).json()
            body.pop("processing")
            bodies.append(json.dumps(body, sort_keys=True))
        assert bodies[0] == bodies[1]


class TestConvenienceAccessors:
    """The helpers exist so the retrieval team does not each write them."""

    @pytest.fixture
    def document(self, client: TestClient) -> ProcessedDocument:
        return ProcessedDocument.model_validate(
            client.post(
                "/process",
                files={"file": ("r.pdf", simple_report_pdf(), "application/pdf")},
            ).json()
        )

    def test_page_lookup(self, document: ProcessedDocument) -> None:
        assert document.page(1) is not None
        assert document.page(99) is None

    def test_block_lookup(self, document: ProcessedDocument) -> None:
        first = document.pages[0].blocks[0]
        assert document.block(first.block_id) is first
        assert document.block("nope") is None

    def test_section_path_renders_a_citation_string(
        self, document: ProcessedDocument
    ) -> None:
        section = document.sections[-1]
        rendered = document.section_path(section.section_id)
        assert rendered == " > ".join(section.path)
        # Safe to call unguarded for a block with no section.
        assert document.section_path(None) == ""

    def test_iter_blocks_follows_reading_order_and_skips_furniture(
        self, document: ProcessedDocument
    ) -> None:
        seen = [b.block_id for b in document.iter_blocks()]
        expected = [bid for page in document.pages for bid in page.reading_order]
        assert seen == expected
        assert not any(b.type.is_page_furniture for b in document.iter_blocks())
