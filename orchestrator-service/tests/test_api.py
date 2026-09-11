"""
Tests the /ask endpoint through FastAPI's TestClient — i.e. exercises
request validation, routing, and response shaping, not just the bare
process_question() function. Downstream service calls are still mocked
(see test_orchestrator_unit.py for why: no real services needed to verify
orchestrator logic).
"""
from app import orchestrator as orchestrator_module
from app.schemas import AgentServiceResponse, ValidatorSuccessResponse
from tests.conftest import make_answer


def test_ask_returns_validated_answer(client, monkeypatch):
    answer = make_answer(answer_type="direct", value="$142.5M")

    async def fake_call_agent_service(http_client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(http_client, answer_arg, trace_id):
        return ValidatorSuccessResponse(status="validated", answer=answer_arg)

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    response = client.post("/ask", json={"question": "What was the operating income in 2020?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "validated"
    assert body["answer"] == "$142.5M"
    assert body["evidence"][0]["document_id"] == "doc_041"


def test_ask_rejects_empty_question():
    # No monkeypatching needed — this should fail FastAPI's request
    # validation before ever reaching the orchestration logic.
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        response = c.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_health_endpoint_reports_dependency_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "orchestrator-api"
    assert "agent-service" in body["dependencies"]
    assert "answer-validator-api" in body["dependencies"]
    # With nothing actually running on 8004/8006 in a unit-test environment,
    # both should honestly report unreachable rather than error out.
    assert body["dependencies"]["agent-service"] in ("reachable", "degraded", "unreachable")
