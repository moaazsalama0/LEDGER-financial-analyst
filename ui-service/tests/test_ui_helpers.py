from app.formatting import ResponseFormatter
from app.orchestrator_client import AskResult, UploadResult


def test_format_validated_list_answer():
    result = AskResult(
        ok=True,
        status="validated",
        answer=["A", "B"],
        evidence=[
            {
                "document_id": "doc_001",
                "page_number": 5,
                "section": "Income Statement",
            }
        ],
    )

    rendered = ResponseFormatter().format_ask_result(result)

    assert "- A\n- B" in rendered
    assert "doc_001" in rendered
    assert "page 5" in rendered


def test_format_rejected_answer_contains_reason():
    result = AskResult(
        ok=False,
        status="rejected",
        reason="Missing required evidence",
    )

    rendered = ResponseFormatter().format_ask_result(result)

    assert "Rejected" in rendered
    assert "Missing required evidence" in rendered


def test_format_insufficient_evidence_shows_reason():
    result = AskResult(
        ok=True,
        status="validated",
        answer_type="insufficient_evidence",
        reason="No relevant evidence was retrieved.",
    )

    rendered = ResponseFormatter().format_ask_result(result)

    assert "Insufficient evidence" in rendered
    assert "No relevant evidence was retrieved." in rendered
    assert "no value" not in rendered


def test_format_successful_upload():
    result = UploadResult(
        ok=True,
        status="success",
        document_id="doc_001",
        message="Indexed",
        filename="report.pdf",
    )

    rendered = ResponseFormatter().format_upload_result(result)

    assert "report.pdf" in rendered
    assert "doc_001" in rendered
