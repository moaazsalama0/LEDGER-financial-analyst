"""
Real, end-to-end integration tests — no mocking. These require:
  1. agent-service running on ORCHESTRATOR_AGENT_URL (default http://localhost:8004)
  2. answer-validator-api running on ORCHESTRATOR_VALIDATOR_URL (default http://localhost:8006)
  3. orchestrator-api itself started separately on http://localhost:8000
     (uvicorn app.main:app --host 0.0.0.0 --port 8000)

Run all three services first, THEN run:
    pytest -m integration -v

These are skipped automatically if orchestrator-api isn't reachable, so a
plain `pytest` run (no services running) still passes cleanly using only
the mocked unit tests in test_orchestrator_unit.py / test_api.py.

Adjust ORCHESTRATOR_BASE_URL / SAMPLE_QUESTION below to match a question
your indexed corpus can actually answer, once documents are ingested.
"""
import os

import httpx
import pytest

ORCHESTRATOR_BASE_URL = os.environ.get("ORCHESTRATOR_BASE_URL", "http://localhost:8000")
SAMPLE_QUESTION = os.environ.get(
    "LEDGER_SAMPLE_QUESTION",
    "What was the operating income in 2020?",
)


def _orchestrator_is_up() -> bool:
    try:
        r = httpx.get(f"{ORCHESTRATOR_BASE_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration
requires_live_stack = pytest.mark.skipif(
    not _orchestrator_is_up(),
    reason=f"orchestrator-api not reachable at {ORCHESTRATOR_BASE_URL} — start all three services first",
)


@requires_live_stack
def test_health_shows_both_dependencies_reachable():
    r = httpx.get(f"{ORCHESTRATOR_BASE_URL}/health", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    print("\n[integration] /health ->", body)

    for dep in ("agent-service", "answer-validator-api"):
        assert body["dependencies"].get(dep) == "reachable", (
            f"{dep} is not reachable — make sure it's actually running "
            f"and orchestrator-api's .env points at the right host/port."
        )


@requires_live_stack
def test_ask_end_to_end_returns_well_formed_response():
    r = httpx.post(
        f"{ORCHESTRATOR_BASE_URL}/ask",
        json={"question": SAMPLE_QUESTION},
        timeout=60.0,  # real LLM calls can be slow
    )
    assert r.status_code == 200
    body = r.json()
    print("\n[integration] /ask ->", body)

    assert body["status"] in ("validated", "rejected", "error")

    if body["status"] == "validated":
        # answer may legitimately be None only for a validated
        # insufficient_evidence result.
        assert "evidence" in body
    elif body["status"] == "rejected":
        assert body.get("reason"), "rejected responses should always carry a reason"
    elif body["status"] == "error":
        assert body.get("reason"), "error responses should always explain what failed"


@requires_live_stack
def test_ask_rejects_malformed_request():
    r = httpx.post(f"{ORCHESTRATOR_BASE_URL}/ask", json={"question": ""}, timeout=5.0)
    assert r.status_code == 422


@requires_live_stack
def test_ask_handles_a_question_the_corpus_cannot_answer():
    """
    Sanity check for the insufficient_evidence path end to end — swap this
    question for one you know isn't covered by your indexed documents.
    """
    nonsense_question = "What was the restructuring charge reported by a company that does not exist in this corpus?"
    r = httpx.post(f"{ORCHESTRATOR_BASE_URL}/ask", json={"question": nonsense_question}, timeout=60.0)
    assert r.status_code == 200
    body = r.json()
    print("\n[integration] insufficient_evidence check ->", body)
    # Not a hard assertion on status, since it depends on your actual corpus —
    # printed above so you can eyeball whether it correctly came back
    # insufficient_evidence rather than a hallucinated guess.
