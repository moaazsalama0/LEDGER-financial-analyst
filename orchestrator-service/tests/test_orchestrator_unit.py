"""
Unit tests for the core orchestration logic in app/orchestrator.py.

These mock out call_agent_service / call_answer_validator entirely, so they
run with no network access and no real downstream services — good for
verifying the orchestrator's own branching logic (validated / rejected /
error / insufficient_evidence) in isolation.

For tests against the real, running agent-service and answer-validator-api,
see tests/test_integration_live.py.
"""
import pytest

from app import orchestrator as orchestrator_module
from app.clients.base import DownstreamServiceError
from app.schemas import AgentServiceResponse, ValidatorRejectedResponse, ValidatorSuccessResponse
from tests.conftest import make_answer


class DummyClient:
    """Stand-in for httpx.AsyncClient — process_question just forwards it
    to the (mocked) client functions, so it never actually needs to make
    a request in these unit tests."""
    pass


@pytest.mark.asyncio
async def test_direct_answer_validated(monkeypatch):
    answer = make_answer(answer_type="direct", value="$142.5M")

    async def fake_call_agent_service(client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        return ValidatorSuccessResponse(status="validated", answer=answer_arg)

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result = await orchestrator_module.process_question("What was operating income?", DummyClient())

    assert result.status == "validated"
    assert result.answer == "$142.5M"
    assert result.evidence[0]["document_id"] == "doc_041"
    assert result.reason is None


@pytest.mark.asyncio
async def test_multi_span_answer_returns_list(monkeypatch):
    answer = make_answer(answer_type="multi_span", values=["Marketing", "R&D", "Logistics"])

    async def fake_call_agent_service(client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        return ValidatorSuccessResponse(status="validated", answer=answer_arg)

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result = await orchestrator_module.process_question("Which expenses increased?", DummyClient())

    assert result.status == "validated"
    assert result.answer == ["Marketing", "R&D", "Logistics"]


@pytest.mark.asyncio
async def test_insufficient_evidence_is_still_validated_not_an_error(monkeypatch):
    answer = make_answer(answer_type="insufficient_evidence", reason="No matching document in corpus.")

    async def fake_call_agent_service(client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        return ValidatorSuccessResponse(status="validated", answer=answer_arg)

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result = await orchestrator_module.process_question("What were restructuring costs?", DummyClient())

    # Per the Definition of Done: insufficient_evidence is a correct, validated
    # outcome — not a pipeline failure — so status should be "validated", not "error".
    assert result.status == "validated"
    assert result.answer is None
    assert result.evidence == []


@pytest.mark.asyncio
async def test_validator_rejection_is_surfaced_with_reason(monkeypatch):
    answer = make_answer(answer_type="direct", value="314,258 thousand")

    async def fake_call_agent_service(client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        return ValidatorRejectedResponse(status="rejected", reason="Missing required evidence")

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result = await orchestrator_module.process_question("What was the balance?", DummyClient())

    assert result.status == "rejected"
    assert result.reason == "Missing required evidence"
    assert result.answer is None


@pytest.mark.asyncio
async def test_agent_service_unreachable_returns_error_not_exception(monkeypatch):
    async def fake_call_agent_service(client, question, trace_id):
        raise DownstreamServiceError("agent-service", "connection failed: [Errno 111] Connection refused")

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)

    result = await orchestrator_module.process_question("Any question", DummyClient())

    assert result.status == "error"
    assert "agent-service" in result.reason


@pytest.mark.asyncio
async def test_validator_unreachable_returns_error_not_exception(monkeypatch):
    answer = make_answer(answer_type="direct", value="100")

    async def fake_call_agent_service(client, question, trace_id):
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        raise DownstreamServiceError("answer-validator-api", "request timed out after 30.0s")

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result = await orchestrator_module.process_question("Any question", DummyClient())

    assert result.status == "error"
    assert "answer-validator-api" in result.reason


@pytest.mark.asyncio
async def test_every_call_gets_a_unique_trace_id(monkeypatch):
    seen_trace_ids = []
    answer = make_answer(answer_type="direct", value="1")

    async def fake_call_agent_service(client, question, trace_id):
        seen_trace_ids.append(trace_id)
        return AgentServiceResponse(answer=answer)

    async def fake_call_answer_validator(client, answer_arg, trace_id):
        seen_trace_ids.append(trace_id)
        return ValidatorSuccessResponse(status="validated", answer=answer_arg)

    monkeypatch.setattr(orchestrator_module, "call_agent_service", fake_call_agent_service)
    monkeypatch.setattr(orchestrator_module, "call_answer_validator", fake_call_answer_validator)

    result_a = await orchestrator_module.process_question("Question A", DummyClient())
    result_b = await orchestrator_module.process_question("Question B", DummyClient())

    # Same trace_id used for both downstream calls within one request...
    assert seen_trace_ids[0] == seen_trace_ids[1]
    # ...but a different trace_id for a separate request.
    assert result_a.trace_id != result_b.trace_id
