"""Tests for the cache namespace.

A cache that serves one configuration's parse to a request that asked for
another produces a result nobody can reproduce — the exact failure the rest of
this service refuses to allow elsewhere, and the one place it would happen
silently rather than loudly.
"""

from __future__ import annotations

import app.cache.store as store
from app.cache.store import PROCESSING_VERSION, cache_variant


class TestVariant:
    def test_a_backend_with_no_models_is_named_plainly(self) -> None:
        """A backend that loads nothing has no model configuration to encode,
        so the directory stays readable. The Phase 2 text-layer control is the
        one that behaves this way, and its cache entries are still on disk."""
        assert cache_variant("textonly", {}) == f"textonly-p{PROCESSING_VERSION}"

    def test_the_backend_name_stays_legible(self) -> None:
        """Someone clearing a corpus cache by hand should be able to see which
        directory belongs to which backend."""
        variant = cache_variant("docling", {"table": "tableformer-accurate"})
        assert variant.startswith("docling-")

    def test_the_same_configuration_gives_the_same_variant(self) -> None:
        """Otherwise every restart would miss the whole corpus cache."""
        models = {"layout": "heron", "table": "tableformer-accurate"}
        assert cache_variant("docling", models) == cache_variant("docling", dict(models))

    def test_key_order_does_not_change_the_variant(self) -> None:
        first = cache_variant("docling", {"layout": "heron", "table": "fast"})
        second = cache_variant("docling", {"table": "fast", "layout": "heron"})
        assert first == second

    def test_a_different_table_model_is_a_different_variant(self) -> None:
        """TableFormer's accurate and fast modes recover different grids and
        score differently. Sharing a cache entry between them would make a
        measured backend comparison meaningless."""
        accurate = cache_variant("docling", {"table": "tableformer-accurate"})
        fast = cache_variant("docling", {"table": "tableformer-fast"})
        assert accurate != fast

    def test_a_different_layout_model_is_a_different_variant(self) -> None:
        older = cache_variant("docling", {"layout": "docling-layout-heron@main"})
        newer = cache_variant("docling", {"layout": "docling-layout-egret@main"})
        assert older != newer

    def test_disabling_ocr_is_a_different_variant(self) -> None:
        """A parse that could not read a scanned page is not interchangeable
        with one that did."""
        with_ocr = cache_variant("docling", {"table": "t", "ocr": "auto"})
        without = cache_variant("docling", {"table": "t", "ocr": "disabled"})
        assert with_ocr != without

    def test_two_backends_never_share_a_namespace(self) -> None:
        models = {"table": "tableformer-accurate"}
        assert cache_variant("docling", models) != cache_variant("surya", models)


class TestProcessingVersion:
    """What is cached is the assembled document, not the backend's raw parse.

    A change to reading order, heading levels, numeric parsing, or scale
    detection produces a different document from identical bytes and identical
    models. If the key did not move with it, improving the pipeline and then
    measuring the result would silently score the old build.
    """

    def test_the_variant_carries_the_processing_version(self) -> None:
        assert cache_variant("docling", {"table": "t"}).endswith(
            f"-p{PROCESSING_VERSION}"
        )

    def test_a_new_processing_build_cannot_read_the_old_entries(
        self, monkeypatch
    ) -> None:
        models = {"table": "tableformer-accurate"}
        before = cache_variant("docling", models)
        monkeypatch.setattr(store, "PROCESSING_VERSION", "999")
        assert cache_variant("docling", models) != before

    def test_it_applies_to_a_backend_with_no_models_too(self, monkeypatch) -> None:
        before = cache_variant("textonly", {})
        monkeypatch.setattr(store, "PROCESSING_VERSION", "999")
        assert cache_variant("textonly", {}) != before


class TestHistoricalVariantsStaySeparate:
    """Removing a backend must not let its cached parses be read as another's.

    The Phase 2 control's entries are still on disk under `textonly-p1` and
    `textonly-p2`. They are the frozen baseline's raw material, so they stay;
    what matters is that nothing can now mistake one for a Docling parse.
    """

    def test_the_retired_baseline_never_shares_docling_namespace(self) -> None:
        models = {"layout": "docling-layout-heron@main", "table": "tableformer-accurate"}
        assert cache_variant("textonly", {}) != cache_variant("docling", models)

    def test_the_retired_baseline_keeps_its_own_directory(self) -> None:
        """Unchanged by the removal, so the Phase 2 entries stay addressable
        by the evaluation harness that produced them."""
        assert cache_variant("textonly", {}).startswith("textonly-")
