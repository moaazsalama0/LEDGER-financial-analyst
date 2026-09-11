import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def post(payload):
    return client.post("/validate_answer", json=payload)


# ---------- direct ----------

def test_direct_valid():
    r = post({
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page_number": 1, "section": "Income Statement"}],
        "params": {"value": "$142.5M"},
    })
    assert r.status_code == 200
    assert r.json()["status"] == "validated"


def test_direct_missing_value():
    r = post({
        "answer_type": "direct",
        "evidence": [{"document_id": "doc_017", "page_number": 1}],
        "params": {},
    })
    assert r.status_code == 422
    assert "value" in r.json()["reason"]


def test_direct_missing_evidence():
    r = post({"answer_type": "direct", "evidence": [], "params": {"value": "1.0"}})
    assert r.status_code == 422
    assert r.json()["reason"] == "Missing required evidence citation."


# ---------- calculated ----------

def test_calculated_valid():
    r = post({
        "answer_type": "calculated",
        "evidence": [
            {"document_id": "doc_041", "page_number": 2, "section": "Operating Expenses"},
            {"document_id": "doc_041", "page_number": 2, "section": "Operating Expenses"},
        ],
        "params": {"value": 13.4, "formula": "(3875-3410)/3410*100"},
    })
    assert r.status_code == 200


def test_calculated_missing_formula():
    r = post({
        "answer_type": "calculated",
        "evidence": [{"document_id": "doc_041", "page_number": 2}],
        "params": {"value": 13.4},
    })
    assert r.status_code == 422
    assert "formula" in r.json()["reason"]


# ---------- multi_span ----------

def test_multi_span_valid():
    r = post({
        "answer_type": "multi_span",
        "evidence": [{"document_id": "doc_022", "page_number": 3, "section": "Operating Expenses"}],
        "params": {"values": ["Marketing", "R&D", "Logistics"]},
    })
    assert r.status_code == 200


def test_multi_span_single_value_rejected():
    r = post({
        "answer_type": "multi_span",
        "evidence": [{"document_id": "doc_022", "page_number": 3}],
        "params": {"values": ["Marketing"]},
    })
    assert r.status_code == 422


# ---------- insufficient_evidence ----------

def test_insufficient_evidence_valid():
    r = post({
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {"reason": "No document in the indexed corpus reports restructuring expenses."},
    })
    assert r.status_code == 200


def test_insufficient_evidence_missing_reason():
    r = post({"answer_type": "insufficient_evidence", "evidence": [], "params": {}})
    assert r.status_code == 422
    assert "reason" in r.json()["reason"]


# ---------- envelope-level ----------

def test_unknown_answer_type_rejected():
    r = post({"answer_type": "ranking", "evidence": [], "params": {}})
    assert r.status_code == 422