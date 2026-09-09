from fastapi.testclient import TestClient

from service.main import app

client = TestClient(app)


def test_end_to_end_eval_api_without_langfuse():
    trace_id = "api-test-trace"

    retrieval = {
        "trace_id": trace_id,
        "question_id": "P001",
        "latency": 0.1,
        "evidence": [
            {
                "document_id": "doc-1",
                "chunk_id": "c1",
                "rank": 1,
                "score": 0.9,
                "page_number": 1,
                "content_type": "text",
            }
        ],
    }
    assert client.post("/events/retrieval", json=retrieval).status_code == 200

    agent = {
        "trace_id": trace_id,
        "question_id": "P001",
        "latency": 0.5,
        "predicted_answer": "38.4 million",
        "unit": "million",
        "answer_type": "direct",
        "selected_evidence_ids": ["c1"],
    }
    assert client.post("/events/agent", json=agent).status_code == 200

    validator = {
        "trace_id": trace_id,
        "question_id": "P001",
        "latency": 0.01,
        "status": "validated",
        "confidence": 1.0,
    }
    assert client.post("/events/validator", json=validator).status_code == 200

    response = client.post(
        "/evaluate",
        json={
            "trace_id": trace_id,
            "ground_truth": {
                "question_id": "P001",
                "answer": "38.4 million",
                "scale": "million",
                "gold_doc_ids": ["doc-1"],
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_metrics"]["recall_at_1"] == 1.0
    assert body["answer_metrics"]["exact_match"] == 1.0
    assert body["trace_failure_analysis"]["retrieval_failure"] is False

    client.delete(f"/traces/{trace_id}")
