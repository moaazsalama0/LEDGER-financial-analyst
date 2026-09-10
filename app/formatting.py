"""
formatting.py
=============
Turns the raw AskResult / UploadResult objects into the Markdown strings
shown in the chat bubble. Kept separate from ui.py so the "how do we
phrase a rejected answer" logic isn't tangled up with Gradio wiring.
"""
from __future__ import annotations

from app.orchestrator_client import AskResult, UploadResult

STATUS_BADGE = {
    "validated": "✅ **Validated**",
    "rejected": "🟠 **Rejected by validator**",
    "error": "🔴 **Error**",
    "network_error": "🔴 **Connection error**",
}


class ResponseFormatter:
    """Stateless formatter — every method is a pure function of its arguments."""

    @staticmethod
    def _format_answer_value(answer: object) -> str:
        if answer is None:
            return "_(no value)_"
        if isinstance(answer, (list, tuple)):
            if not answer:
                return "_(empty list)_"
            return "\n".join(f"- {item}" for item in answer)
        return str(answer)

    @staticmethod
    def _format_evidence_line(evidence: dict) -> str:
        doc_id = evidence.get("document_id", "unknown-doc")
        page = evidence.get("page", evidence.get("page_number", "?"))
        section = evidence.get("section")
        if section:
            return f"`{doc_id}` · page {page} · _{section}_"
        return f"`{doc_id}` · page {page}"

    def format_ask_result(self, result: AskResult) -> str:
        badge = STATUS_BADGE.get(result.status, f"**{result.status}**")
        lines = [badge]

        if result.status == "validated":
            lines.append("")
            lines.append(self._format_answer_value(result.answer))
            if result.evidence:
                lines.append("")
                lines.append("**Evidence:**")
                for ev in result.evidence:
                    lines.append(f"- {self._format_evidence_line(ev)}")
        elif result.status == "rejected":
            lines.append("")
            lines.append(
                f"The candidate answer failed schema/grounding validation, so it was blocked "
                f"before reaching you (this is the pipeline working as intended, not a bug)."
            )
            if result.reason:
                lines.append(f"\n**Reason:** {result.reason}")
        else:
            lines.append("")
            lines.append(result.reason or "Something went wrong while processing this question.")

        if result.trace_id:
            lines.append(f"\n<sub>trace_id: `{result.trace_id}` · {result.latency_ms:.0f} ms</sub>")
        return "\n".join(lines)

    def format_upload_result(self, result: UploadResult) -> str:
        if result.status == "success":
            return (
                f"✅ **{result.filename}** processed and indexed.\n\n"
                f"`document_id`: **{result.document_id}**\n\n"
                f"{result.message}"
            )
        stage_note = f" (stage: `{result.stage}`)" if result.stage else ""
        return f"🔴 **{result.filename}** failed to index{stage_note}.\n\n{result.message}"
