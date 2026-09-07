import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AgentAnswer, Evidence


@pytest.fixture
def client():
    """
    FastAPI's TestClient runs the ASGI app in-process (no real network
    socket), including the startup/shutdown events that create/close the
    shared httpx.AsyncClient in app.main. Safe to use even though the app's
    endpoints are async.
    """
    with TestClient(app) as c:
        yield c


def make_answer(
    answer_type: str = "direct",
    value=None,
    values=None,
    reason: str = None,
    document_id: str = "doc_041",
    page: int = 2,
    section: str = "Operating Expenses",
) -> AgentAnswer:
    """Builds a valid AgentAnswer for the given answer_type, for use across tests."""
    params = {}
    if value is not None:
        params["value"] = value
    if values is not None:
        params["values"] = values
    if reason is not None:
        params["reason"] = reason

    evidence = [] if answer_type == "insufficient_evidence" else [
        Evidence(document_id=document_id, page=page, section=section)
    ]

    return AgentAnswer(answer_type=answer_type, evidence=evidence, params=params)
