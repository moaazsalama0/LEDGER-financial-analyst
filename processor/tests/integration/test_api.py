"""End-to-end tests of the HTTP surface, including the whole error matrix.

The error matrix matters as much as the happy path: the orchestrator branches
on ``error_code``, so a wrong status or a missing code is a contract break
even when the document parses fine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ledger_doc_contract.v1.enums import BlockType, ErrorCode, ExtractionSource
from tests.fixtures.make_pdf import (
    PageSpec,
    TextItem,
    build_pdf,
    empty_text_pdf,
    large_pdf,
    simple_report_pdf,
    single_page_pdf,
)


def post_pdf(
    client: TestClient,
    data: bytes,
    *,
    filename: str = "report.pdf",
    content_type: str = "application/pdf",
    **form: str,
):
    return client.post(
        "/process",
        files={"file": (filename, data, content_type)},
        data=form,
    )


class TestHappyPath:
    def test_returns_the_contract_shape(self, client: TestClient) -> None:
        response = post_pdf(client, simple_report_pdf(), document_id="doc_0417")
        assert response.status_code == 200

        body = response.json()
        assert body["contract_version"] == "1.0"
        assert body["document_id"] == "doc_0417"
        assert body["filename"] == "report.pdf"
        assert body["page_count"] == 2
        assert len(body["pages"]) == 2
        assert len(body["content_sha256"]) == 64

    def test_every_block_carries_the_metadata_retrieval_needs(
        self, client: TestClient
    ) -> None:
        """The brief requires document_id / page / section / content_type on
        every chunk. A chunk is built from a block, so a block must carry them.
        """
        body = post_pdf(client, simple_report_pdf()).json()

        for page in body["pages"]:
            for block in page["blocks"]:
                assert block["page_number"] == page["page_number"]
                assert block["type"] in {t.value for t in BlockType}
                assert "section_id" in block
                assert block["block_id"].startswith(f"p{page['page_number']:04d}_b")

    def test_pages_report_how_their_text_was_obtained(self, client: TestClient) -> None:
        body = post_pdf(client, simple_report_pdf()).json()
        page = body["pages"][0]
        assert page["extraction_source"] == ExtractionSource.TEXT_LAYER.value
        # Not measured, rather than perfect. A fake 1.0 here would tell a
        # downstream quality gate the opposite of the truth.
        assert page["ocr_confidence"] is None
        assert page["low_confidence"] is False

    def test_bounding_boxes_lie_inside_the_page(self, client: TestClient) -> None:
        body = post_pdf(client, simple_report_pdf()).json()
        for page in body["pages"]:
            for block in page["blocks"]:
                box = block["bbox"]
                assert box is not None
                assert 0 <= box["x0"] <= box["x1"] <= page["width"] + 1
                assert 0 <= box["y0"] <= box["y1"] <= page["height"] + 1

    def test_reading_order_excludes_page_furniture(self, client: TestClient) -> None:
        body = post_pdf(client, simple_report_pdf()).json()
        page = body["pages"][0]
        furniture = {
            b["block_id"]
            for b in page["blocks"]
            if b["type"] in {BlockType.PAGE_HEADER.value, BlockType.PAGE_FOOTER.value}
        }
        assert furniture, "the fixture has a page footer to exclude"
        assert furniture.isdisjoint(page["reading_order"])
        # Excluded from the order, but still emitted so a caller can rebuild
        # the whole page.
        assert furniture <= {b["block_id"] for b in page["blocks"]}

    def test_section_paths_join_into_a_citation_string(self, client: TestClient) -> None:
        body = post_pdf(client, simple_report_pdf()).json()
        sections = body["sections"]
        assert sections, "the fixture has headings"
        deepest = max(sections, key=lambda s: s["level"])
        assert " > ".join(deepest["path"]).count(" > ") >= 1

    def test_section_tree_is_internally_consistent(self, client: TestClient) -> None:
        body = post_pdf(client, simple_report_pdf()).json()
        by_id = {s["section_id"]: s for s in body["sections"]}
        for section in body["sections"]:
            parent_id = section["parent_section_id"]
            if parent_id is None:
                assert len(section["path"]) == 1
                continue
            parent = by_id[parent_id]
            assert parent["level"] < section["level"]
            assert section["path"][:-1] == parent["path"]
            assert section["page_end"] <= parent["page_end"]


class TestDocumentId:
    def test_defaults_to_a_content_derived_id(self, client: TestClient) -> None:
        body = post_pdf(client, single_page_pdf()).json()
        assert body["document_id"].startswith("doc_")
        assert body["document_id"][4:] == body["content_sha256"][:16]

    def test_same_bytes_yield_the_same_default_id(self, client: TestClient) -> None:
        data = single_page_pdf()
        first = post_pdf(client, data).json()
        second = post_pdf(client, data).json()
        assert first["document_id"] == second["document_id"]

    @pytest.mark.parametrize(
        "bad", ["has space", "sla/sh", "a" * 129, "semi;colon", "quote'"]
    )
    def test_rejects_ids_outside_the_pattern(self, client: TestClient, bad: str) -> None:
        response = post_pdf(client, single_page_pdf(), document_id=bad)
        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_DOCUMENT_ID.value


class TestErrorMatrix:
    def test_not_a_pdf(self, client: TestClient) -> None:
        response = post_pdf(client, b"this is plainly not a pdf")
        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_PDF.value

    def test_empty_upload(self, client: TestClient) -> None:
        response = post_pdf(client, b"")
        assert response.status_code == 400
        assert response.json()["error_code"] == ErrorCode.INVALID_PDF.value

    def test_wrong_content_type(self, client: TestClient) -> None:
        response = post_pdf(client, single_page_pdf(), content_type="image/png")
        assert response.status_code == 415
        assert response.json()["error_code"] == ErrorCode.UNSUPPORTED_MEDIA_TYPE.value

    def test_octet_stream_is_tolerated(self, client: TestClient) -> None:
        """Many clients send octet-stream for any file; magic bytes decide."""
        response = post_pdf(
            client, single_page_pdf(), content_type="application/octet-stream"
        )
        assert response.status_code == 200

    def test_too_many_pages(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("DOC_MAX_PAGES", "1")
        from app.core.settings import get_settings

        get_settings.cache_clear()

        response = post_pdf(client, simple_report_pdf())
        assert response.status_code == 422
        assert response.json()["error_code"] == ErrorCode.TOO_MANY_PAGES.value

    def test_file_too_large(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("DOC_MAX_UPLOAD_MB", "1")
        from app.core.settings import get_settings

        get_settings.cache_clear()

        oversized = large_pdf(2 * 1024 * 1024)
        response = post_pdf(client, oversized)
        assert response.status_code == 413

        body = response.json()
        assert body["error_code"] == ErrorCode.FILE_TOO_LARGE.value
        assert body["detail"]["limit_bytes"] == 1024 * 1024

    def test_size_limit_must_be_positive(self) -> None:
        """A zero or negative limit would reject every upload, so the settings
        model refuses it rather than starting a service that accepts nothing."""
        import pytest as _pytest
        from pydantic import ValidationError

        from app.core.settings import Settings

        with _pytest.raises(ValidationError):
            Settings(max_upload_mb=0)

    def test_unknown_backend(self, client: TestClient) -> None:
        response = post_pdf(client, single_page_pdf(), options='{"backend": "nope"}')
        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == ErrorCode.BACKEND_UNAVAILABLE.value
        # The response names what is available, so the mistake is obvious
        # without reading the server logs.
        assert "docling" in body["detail"]["available"]

    def test_malformed_options_json(self, client: TestClient) -> None:
        response = post_pdf(client, single_page_pdf(), options="{not json")
        assert response.status_code == 422
        assert response.json()["error_code"] == ErrorCode.INVALID_OPTIONS.value

    def test_unknown_option_is_rejected_not_ignored(self, client: TestClient) -> None:
        """A misspelled flag must fail loudly, not be silently dropped."""
        response = post_pdf(client, single_page_pdf(), options='{"force_ocrr": true}')
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == ErrorCode.INVALID_OPTIONS.value
        assert "force_ocrr" in body["detail"]["unknown"]

    def test_every_error_body_has_the_documented_shape(self, client: TestClient) -> None:
        response = post_pdf(client, b"not a pdf")
        body = response.json()
        assert set(body) == {"error_code", "message", "document_id", "detail"}


class TestDegradedPages:
    def test_a_page_with_no_text_layer_is_reported_not_hidden(
        self, client: TestClient
    ) -> None:
        """A backend that cannot OCR must say so rather than returning a
        silently empty page that reads as a blank document."""
        response = post_pdf(client, empty_text_pdf())
        assert response.status_code == 200

        body = response.json()
        page = body["pages"][0]
        assert page["extraction_source"] == ExtractionSource.FAILED.value
        assert page["blocks"] == []

        codes = {w["code"] for w in body["processing"]["warnings"]}
        assert "SCANNED_PAGE_OCR_FALLBACK" in codes

    def test_one_bad_page_does_not_cost_the_document(self, client: TestClient) -> None:
        good = PageSpec(items=[TextItem("Readable page", x=72.0, y=700.0)])
        blank = PageSpec(items=[])
        response = post_pdf(client, build_pdf([good, blank, good]))

        assert response.status_code == 200
        body = response.json()
        assert body["page_count"] == 3
        sources = [p["extraction_source"] for p in body["pages"]]
        assert sources == ["text_layer", "failed", "text_layer"]


class TestCaching:
    def test_second_request_is_served_from_cache(self, client: TestClient) -> None:
        data = simple_report_pdf()
        first = post_pdf(client, data).json()
        second = post_pdf(client, data).json()

        assert first["processing"]["from_cache"] is False
        assert second["processing"]["from_cache"] is True

    def test_cached_response_matches_the_freshly_parsed_one(
        self, client: TestClient
    ) -> None:
        data = simple_report_pdf()
        first = post_pdf(client, data).json()
        second = post_pdf(client, data).json()

        first.pop("processing")
        second.pop("processing")
        assert first == second

    def test_cache_hit_still_honours_the_requested_document_id(
        self, client: TestClient
    ) -> None:
        """Two teams may ingest the same PDF under different ids; the cached
        parse is reusable but the id must not leak between them."""
        data = simple_report_pdf()
        post_pdf(client, data, document_id="doc_first")
        second = post_pdf(client, data, document_id="doc_second").json()

        assert second["processing"]["from_cache"] is True
        assert second["document_id"] == "doc_second"


class TestServiceEndpoints:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_ready_reports_the_backend_it_would_serve_with(
        self, client: TestClient
    ) -> None:
        body = client.get("/ready").json()
        assert body["ready"] is True
        assert body["backend"] == "textonly"

    def test_version_exposes_what_a_consumer_needs(self, client: TestClient) -> None:
        body = client.get("/version").json()
        assert body["contract_version"] == "1.0"
        assert body["service"] == "doc-processor-api"
        assert body["default_backend"] == "textonly"

    def test_schema_endpoint_serves_the_output_contract(
        self, client: TestClient
    ) -> None:
        schema = client.get("/contract/schema").json()
        assert schema["title"] == "ProcessedDocument"
        for required in ("document_id", "pages", "sections", "tables"):
            assert required in schema["properties"]


class TestBackendSelection:
    """The registry is a published seam, so its failure modes are contract."""

    def test_the_docling_backend_is_offered(self, client: TestClient) -> None:
        """The deep-learning backend must be selectable without a code change,
        because the brief requires comparing pipeline variants."""
        assert "docling" in client.get("/version").json()["available_backends"]

    def test_an_unknown_backend_is_refused_with_the_alternatives(
        self, client: TestClient
    ) -> None:
        """A misspelled backend name fails loudly rather than falling back.
        A service quietly running a different backend than asked for produces
        results nobody can reproduce."""
        response = post_pdf(
            client, simple_report_pdf(), options='{"backend": "doclign"}'
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "BACKEND_UNAVAILABLE"
        assert "docling" in body["detail"]["available"]


class TestProductionBackend:
    """Docling is the production path, and it is the only one.

    These run against the registry exactly as the service ships it. The fast
    suite's no-model baseline is deliberately not registered here, because a
    claim about what the service offers cannot be checked through a client that
    added something to it.
    """

    def test_docling_is_the_default(self, production_client: TestClient) -> None:
        body = production_client.get("/version").json()
        assert body["default_backend"] == "docling"

    def test_docling_is_the_only_backend_offered(
        self, production_client: TestClient
    ) -> None:
        body = production_client.get("/version").json()
        assert body["available_backends"] == ["docling"]

    def test_version_still_carries_the_service_and_contract_identity(
        self, production_client: TestClient
    ) -> None:
        body = production_client.get("/version").json()
        assert body["service"] == "doc-processor-api"
        assert body["contract_version"] == "1.0"

    def test_ready_names_docling(self, production_client: TestClient) -> None:
        """Reported honestly: not ready until the weights are loaded, which
        this client deliberately never does."""
        body = production_client.get("/ready").json()
        assert body["backend"] == "docling"
        assert body["ready"] is False

    def test_an_explicit_textonly_request_is_refused(
        self, production_client: TestClient
    ) -> None:
        """The Phase 2 control is not a production backend. Asking for it by
        name must fail the way any unserved name does, in the existing error
        contract, rather than quietly parsing with Docling instead."""
        response = post_pdf(
            production_client, simple_report_pdf(), options='{"backend": "textonly"}'
        )
        assert response.status_code == 503

        body = response.json()
        assert body["error_code"] == ErrorCode.BACKEND_UNAVAILABLE.value
        assert body["detail"]["requested"] == "textonly"
        assert body["detail"]["available"] == ["docling"]

    def test_the_registry_ships_docling_alone(self) -> None:
        """Asserted at the seam as well as over HTTP: the response body lists
        names only, so the registry itself is the thing worth pinning."""
        from app.backends import registry

        assert registry.available() == ("docling",)

    def test_a_registered_backend_cannot_leak_into_production(self) -> None:
        """``reset()`` restores the shipped registrations, so a backend one
        test or tool adds cannot make a later one believe it is served."""
        from app.backends import registry
        from evaluation import textonly_baseline

        textonly_baseline.register()
        assert "textonly" in registry.available()

        registry.reset()
        assert registry.available() == ("docling",)


class TestStartup:
    """Two failures that look alike at startup and must not be treated alike."""

    def test_a_misconfigured_backend_name_stops_the_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Misconfiguration is not something to serve traffic through."""
        from app.api.errors import BackendUnavailableError
        from app.core.settings import get_settings
        from app.main import create_app

        monkeypatch.setenv("DOC_BACKEND", "not_a_backend")
        get_settings.cache_clear()

        with pytest.raises(BackendUnavailableError):
            with TestClient(create_app()):
                pass

    def test_models_that_will_not_load_leave_the_service_up_but_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A different problem: no weights on disk, or no route to the model
        host. The service is more useful up and saying so than dead in a
        restart loop, and the orchestrator gets a reason instead of a refused
        connection."""
        from app.backends import registry
        from app.core.settings import get_settings
        from app.main import create_app

        class UnloadableBackend:
            name = "unloadable"
            version = "0"
            model_versions: dict[str, str] = {}

            def is_ready(self) -> bool:
                return False

            def warm(self) -> None:
                raise RuntimeError("could not reach the model host")

            def parse(self, data: bytes, *, force_ocr: bool = False):  # noqa: ANN201
                raise RuntimeError("no models")

        registry.register("unloadable", UnloadableBackend)
        monkeypatch.setenv("DOC_BACKEND", "unloadable")
        get_settings.cache_clear()

        with TestClient(create_app()) as client:
            body = client.get("/health").json()
            assert body["status"] == "ok", "the process is up"

            body = client.get("/ready").json()
            assert body["ready"] is False, "but it must not be sent traffic"
