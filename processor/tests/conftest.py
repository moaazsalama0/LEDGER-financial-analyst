"""Shared fixtures.

Two clients, because the suite has two different jobs.

``client`` drives the *service* — validation, assembly, contract, HTTP,
caching — and needs a backend that parses a real PDF without a model, a GPU, or
a network, so it can run on every commit. That is the frozen Phase 2 text-layer
baseline in ``evaluation.textonly_baseline``, registered here by the test
harness. It is deliberately not something the service registers for itself.

``production_client`` asserts the *configuration*: Docling is the only backend
this service offers, and a request for anything else is refused. It runs
against the registry exactly as it ships, with nothing added, which is the only
way that claim can be checked honestly. It is built without entering the
TestClient context so the lifespan does not warm Docling's weights — the
endpoints it exercises resolve the backend but never parse with it.

Configuration is driven through the environment rather than by patching
``get_settings``. Callers import that function by name at module import time,
so rebinding the attribute afterwards would leave those references pointing at
the original and the override would silently do nothing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.backends import registry
from app.core.settings import Settings, get_settings
from evaluation import textonly_baseline

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(autouse=True)
def isolated_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """Give each test its own cache directory and the shipped backend registry.

    Without the isolation, one test's cached document would be served to the
    next and the assertions would be describing the cache rather than the
    parser. ``registry.reset()`` restores the built-in registrations too, so a
    backend a previous test added cannot make the service look like it offers
    something it does not.
    """
    monkeypatch.setenv("DOC_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DOC_BACKEND", raising=False)

    get_settings.cache_clear()
    registry.reset()

    yield get_settings()

    get_settings.cache_clear()
    registry.reset()


@pytest.fixture
def baseline_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Register the no-model text-layer baseline and select it.

    Undone by ``isolated_settings``' teardown, which resets the registry to
    what the service ships with.
    """
    textonly_baseline.register()
    monkeypatch.setenv("DOC_BACKEND", textonly_baseline.BASELINE_BACKEND)
    get_settings.cache_clear()
    yield textonly_baseline.BASELINE_BACKEND


@pytest.fixture
def client(baseline_backend: str) -> Iterator[TestClient]:
    """A test client over a freshly built app, parsing with no model."""
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def production_client(isolated_settings: Settings) -> TestClient:
    """A client over the service exactly as configured for production.

    No lifespan: entering the context would warm Docling, which downloads
    weights and takes minutes. Everything asserted through this client —
    ``/version``, ``/ready``, and the refusal of an unsupported backend name —
    resolves the backend but never asks it to parse.
    """
    from app.main import create_app

    return TestClient(create_app())
