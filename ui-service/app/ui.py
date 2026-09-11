"""
ui.py
=====
The Gradio front-end for Project LEDGER.

Everything here is organized as one class, LedgerUIApp, so the whole
service is a single object with a clear lifecycle:

    app = LedgerUIApp(settings)
    blocks = app.build()
    blocks.launch(...)

LedgerUIApp owns:
  - an OrchestratorClient  (the only way this service talks to the backend)
  - a DocumentStore        (everything uploaded this session, incl. raw PDF bytes)
  - a QueryLogStore        (every question asked this session, with latency)
  - a PDFEvidenceRenderer   (bonus: bounding-box highlight preview)
  - a ResponseFormatter    (turns raw responses into chat Markdown)

Every Gradio callback below is a *bound method* of LedgerUIApp, so they
all share this same state without any globals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

from app.config import Settings
from app.evidence_renderer import PDFEvidenceRenderer
from app.formatting import ResponseFormatter
from app.orchestrator_client import OrchestratorClient
from app.stores import DocumentRecord, DocumentStore, QueryLogStore, QueryRecord


CUSTOM_CSS = """
:root {
    --ledger-red: #C21E2A;
    --ledger-dark: #10162A;
    --ledger-accent: #D4A72C;
    --ledger-accent-soft: #2A2417;
}
.ledger-header {
    background: linear-gradient(135deg, var(--ledger-dark) 0%, #2A3648 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 6px;
    border-left: 6px solid var(--ledger-accent);
}
.ledger-header h1 {
    color: #ffffff !important;
    margin: 0 0 4px 0 !important;
    font-size: 1.6rem !important;
}
.ledger-header p {
    color: #C7CEDA !important;
    margin: 0 !important;
}
.ledger-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: var(--ledger-accent);
    color: #1B2431;
    margin-left: 8px;
    vertical-align: middle;
}
.stat-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}
.stat-card {
    flex: 1;
    min-width: 150px;
    border: 1px solid #3A4658;
    border-left: 3px solid var(--ledger-accent);
    border-radius: 12px;
    padding: 14px 18px;
    background: #212B3B;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9AA6B8;
}
.stat-card .stat-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 2px;
}
.evidence-note {
    font-size: 0.85rem;
    color: #9AA6B8;
    margin-top: -4px;
}
.health-ok {
    color: #35C97B;
    font-weight: 600;
}
.health-bad {
    color: var(--ledger-red);
    font-weight: 600;
}
.health-unknown {
    color: #C9A227;
    font-weight: 600;
}

.about-card {
    background: #182339;
    border: 1px solid #3A4658;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.about-card h2 {
    color: #ffffff;
    margin: 0 0 6px 0;
    font-size: 1.25rem;
}
.about-card p {
    color: #C7CEDA;
    line-height: 1.6;
}
.about-feature {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-top: 1px solid #2A3648;
}
.about-feature:first-of-type {
    border-top: none;
}
.about-feature .icon {
    font-size: 1.4rem;
}
.about-feature .label {
    color: #ffffff;
    font-weight: 700;
}
.about-feature .desc {
    color: #9AA6B8;
    font-size: 0.88rem;
    margin-top: 2px;
}
"""


class LedgerUIApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.client = OrchestratorClient(
            base_url=settings.ORCHESTRATOR_URL,
            timeout_seconds=settings.REQUEST_TIMEOUT_SECONDS,
            health_timeout_seconds=settings.HEALTH_TIMEOUT_SECONDS,
        )

        self.documents = DocumentStore()
        self.queries = QueryLogStore()
        self.renderer = PDFEvidenceRenderer(dpi=settings.EVIDENCE_RENDER_DPI)
        self.formatter = ResponseFormatter()

    def _merged_document_rows(self) -> list[dict]:
        """
        Merges the local session registry (documents uploaded THROUGH THIS
        UI session — has status/latency/notes) with orchestrator-api's
        GET /documents (the full corpus as orchestrator sees it, including
        anything indexed outside this session). Deduped by document_id;
        the local entry wins when both sources have the same document,
        since it carries richer per-upload detail.

        If orchestrator's /documents call fails for any reason
        (endpoint not built by retrieval-api yet, service down), this
        silently falls back to the local session list alone.
        """

        rows: dict[str, dict] = {}

        for r in self.documents.all():
            rows[r.document_id] = {
                "Uploaded": r.uploaded_at,
                "Filename": r.filename,
                "Document ID": r.document_id or "—",
                "Status": "✅ success" if r.status == "success" else f"🔴 {r.status}",
                "Latency (ms)": round(r.latency_ms, 1),
                "Notes": r.message,
            }

        remote = self.client.list_documents()

        if remote.ok:
            for d in remote.documents:
                if d.document_id in rows:
                    continue

                rows[d.document_id] = {
                    "Uploaded": d.indexed_at or "—",
                    "Filename": d.filename or d.document_title or d.document_id,
                    "Document ID": d.document_id,
                    "Status": "✅ indexed (external)",
                    "Latency (ms)": "—",
                    "Notes": "Indexed outside this UI session",
                }

        return list(rows.values())

    # ------------------------------------------------------------------
    # Shared data -> Gradio component value helpers
    # ------------------------------------------------------------------

    def _documents_dataframe(self) -> pd.DataFrame:
        rows = self._merged_document_rows()

        if not rows:
            return pd.DataFrame(
                columns=[
                    "Uploaded",
                    "Filename",
                    "Document ID",
                    "Status",
                    "Latency (ms)",
                    "Notes",
                ]
            )

        return pd.DataFrame(rows)

    def _detected_tables_dataframe(self) -> pd.DataFrame:
        """
        The Final Project brief's dashboard requirement:
        "detected tables" and "any extracted structured values".

        Table detail comes from orchestrator-api's GET /documents.
        """

        remote = self.client.list_documents()

        columns = [
            "Document ID",
            "Table ID",
            "Page",
            "Rows x Cols",
            "Sample values",
        ]

        if not remote.ok:
            return pd.DataFrame(columns=columns)

        rows = []

        for doc in remote.documents:
            for t in doc.tables:
                sample_str = "; ".join(
                    f"{(v.row or '—')} / {(v.column or '—')} = {v.value}"
                    for v in t.sample_values
                ) or "—"

                rows.append(
                    {
                        "Document ID": doc.document_id,
                        "Table ID": t.table_id or "—",
                        "Page": (
                            t.page_number
                            if t.page_number is not None
                            else "—"
                        ),
                        "Rows x Cols": (
                            f"{t.n_rows or '—'} x {t.n_cols or '—'}"
                        ),
                        "Sample values": sample_str,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows)

    def _queries_dataframe(self) -> pd.DataFrame:
        records = self.queries.recent(50)

        if not records:
            return pd.DataFrame(
                columns=[
                    "Asked",
                    "Question",
                    "Status",
                    "Latency (ms)",
                    "Evidence #",
                    "Answer preview",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "Asked": r.asked_at,
                    "Question": r.question,
                    "Status": r.status,
                    "Latency (ms)": round(r.latency_ms, 1),
                    "Evidence #": r.evidence_count,
                    "Answer preview": r.answer_preview,
                }
                for r in records
            ]
        )

    def _latency_plot_dataframe(self) -> pd.DataFrame:
        records = self.queries.all()

        if not records:
            return pd.DataFrame(
                {
                    "query #": [],
                    "latency_ms": [],
                }
            )

        return pd.DataFrame(
            {
                "query #": list(range(1, len(records) + 1)),
                "latency_ms": [r.latency_ms for r in records],
            }
        )

    def _stats_html(self) -> str:
        breakdown = self.queries.status_breakdown()

        validated = breakdown.get("validated", 0)
        total_q = self.queries.count()

        rate = (
            f"{(validated / total_q * 100):.0f}%"
            if total_q
            else "—"
        )

        avg_latency = (
            f"{self.queries.average_latency_ms():.0f} ms"
            if total_q
            else "—"
        )

        cards = [
            (
                "Documents indexed",
                str(len(self._merged_document_rows())),
            ),
            (
                "Questions asked",
                str(total_q),
            ),
            (
                "Validated answer rate",
                rate,
            ),
            (
                "Avg. response latency",
                avg_latency,
            ),
        ]

        cards_html = "".join(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>'
            for label, value in cards
        )

        return f'<div class="stat-row">{cards_html}</div>'

    def _dropdown_choices(self) -> list[str]:
        ids = set(self.documents.document_ids())

        remote = self.client.list_documents()

        if remote.ok:
            ids.update(d.document_id for d in remote.documents)

        return sorted(ids)

    # ------------------------------------------------------------------
    # Chat tab callbacks
    # ------------------------------------------------------------------

    def _build_effective_question(
        self,
        question: str,
        scope_doc_id: Optional[str],
    ) -> str:
        if scope_doc_id and scope_doc_id != "Entire corpus":
            return (
                f"Regarding only document {scope_doc_id}: "
                f"{question}"
            )

        return question

    def _render_evidence_gallery(
        self,
        evidence: list[dict],
        answer_value: object,
    ) -> tuple[list, str]:

        if not evidence:
            return [], "This answer has no cited evidence to preview."

        gallery_items = []
        notes = []

        for ev in evidence[:6]:
            doc_id = ev.get("document_id")
            page = ev.get("page") or ev.get("page_number") or 1
            section = ev.get("section")

            pdf_bytes = (
                self.documents.get_pdf_bytes(doc_id)
                if doc_id
                else None
            )

            if pdf_bytes is None:
                notes.append(
                    f"`{doc_id}` p.{page}: original PDF not cached in this "
                    "session — upload it in the Documents tab to preview."
                )
                continue

            rendered = self.renderer.render(
                pdf_bytes,
                page_number=page,
                answer_value=answer_value,
                section=section,
            )

            if rendered.ok and rendered.image is not None:
                caption = (
                    f"{doc_id} · page {page}"
                    + (f" · {section}" if section else "")
                )

                gallery_items.append(
                    (rendered.image, caption)
                )

                notes.append(
                    f"`{doc_id}` p.{page}: {rendered.note}"
                )
            else:
                notes.append(
                    f"`{doc_id}` p.{page}: {rendered.note}"
                )

        return gallery_items, "  \n".join(notes)

    def on_ask(
        self,
        question: str,
        scope_doc_id: str,
        history: list[dict],
    ):
        history = history or []

        if not question or not question.strip():
            return (
                history,
                "",
                [],
                "",
                self._stats_html(),
            )

        effective_question = self._build_effective_question(
            question,
            scope_doc_id,
        )

        result = self.client.ask(effective_question)

        reply_markdown = self.formatter.format_ask_result(result)

        history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": reply_markdown,
            },
        ]

        answer_preview = (
            str(result.answer)[:120]
            if result.answer is not None
            else (result.reason or "")[:120]
        )

        self.queries.add(
            QueryRecord(
                question=question,
                status=result.status,
                answer_preview=answer_preview,
                reason=result.reason,
                evidence_count=len(result.evidence),
                latency_ms=result.latency_ms,
                trace_id=result.trace_id,
            )
        )

        gallery_items, evidence_notes = (
            self._render_evidence_gallery(
                result.evidence,
                result.answer,
            )
        )

        return (
            history,
            "",
            gallery_items,
            evidence_notes,
            self._stats_html(),
        )

    def on_clear_chat(self):
        return [], [], ""

    # ------------------------------------------------------------------
    # Documents tab callbacks
    # ------------------------------------------------------------------

    def on_upload(self, files: list):
        if not files:
            return (
                "Please choose at least one PDF to upload.",
                self._documents_dataframe(),
                [],
                gr.Dropdown(
                    choices=[
                        "Entire corpus"
                    ] + self._dropdown_choices()
                ),
                self._stats_html(),
            )

        log_lines = []
        thumbnails = []

        for file_path in files:
            filename = Path(file_path).name

            try:
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()

            except Exception as exc:
                log_lines.append(
                    f"🔴 **{filename}** — could not read the "
                    f"uploaded file: {exc}"
                )
                continue

            result = self.client.upload_document(
                filename=filename,
                file_bytes=pdf_bytes,
            )

            log_lines.append(
                self.formatter.format_upload_result(result)
            )

            self.documents.add(
                DocumentRecord(
                    document_id=(
                        result.document_id
                        or f"(failed) {filename}"
                    ),
                    filename=filename,
                    status=result.status,
                    message=result.message,
                    stage=result.stage,
                    latency_ms=result.latency_ms,
                ),
                pdf_bytes=pdf_bytes,
            )

            if result.ok:
                thumb = self.renderer.render(
                    pdf_bytes,
                    page_number=1,
                )

                if thumb.ok and thumb.image is not None:
                    thumbnails.append(
                        (
                            thumb.image,
                            f"{filename} · page 1",
                        )
                    )

        return (
            "\n\n---\n\n".join(log_lines),
            self._documents_dataframe(),
            thumbnails,
            gr.Dropdown(
                choices=[
                    "Entire corpus"
                ] + self._dropdown_choices()
            ),
            self._stats_html(),
        )

    # ------------------------------------------------------------------
    # Dashboard tab callbacks
    # ------------------------------------------------------------------

    def on_refresh_dashboard(self):
        return (
            self._stats_html(),
            self._documents_dataframe(),
            self._queries_dataframe(),
            self._latency_plot_dataframe(),
            self._detected_tables_dataframe(),
        )

    def on_reset_session(self):
        self.documents = DocumentStore()
        self.queries = QueryLogStore()

        return (
            self._stats_html(),
            self._documents_dataframe(),
            self._queries_dataframe(),
            self._latency_plot_dataframe(),
            gr.Dropdown(
                choices=["Entire corpus"]
            ),
            self._detected_tables_dataframe(),
        )

    # ------------------------------------------------------------------
    # Health tab callback
    # ------------------------------------------------------------------

    def on_check_health(self):
        health = self.client.check_health()

        if not health.ok:
            return (
                f'<span class="health-bad">'
                f'🔴 orchestrator-api unreachable at '
                f'{self.settings.ORCHESTRATOR_URL}'
                f'</span><br/>'
                f'{health.error or ""}'
            )

        lines = [
            f'<b>orchestrator-api</b>: '
            f'<span class="health-ok">'
            f'✅ {health.service_status}'
            f'</span>'
        ]

        for name, status in health.dependencies.items():
            css_class = (
                "health-ok"
                if status == "reachable"
                else (
                    "health-unknown"
                    if status == "degraded"
                    else "health-bad"
                )
            )

            icon = (
                "✅"
                if status == "reachable"
                else (
                    "🟡"
                    if status == "degraded"
                    else "🔴"
                )
            )

            lines.append(
                f'<b>{name}</b>: '
                f'<span class="{css_class}">'
                f'{icon} {status}'
                f'</span>'
            )

        return "<br/>".join(lines)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        with gr.Blocks(
            title=self.settings.APP_TITLE,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="amber",
                secondary_hue="slate",
            ),
        ) as blocks:

            gr.HTML(
                f"""
                <div class="ledger-header">
                    <h1>
                        📒 LEDGER
                        <span class="ledger-badge">
                            MIA TRAINING '27
                        </span>
                    </h1>
                    <p>
                        Financial Document Intelligence Agent —
                        ask questions across your entire indexed
                        document corpus, with every answer grounded
                        and cited.
                    </p>
                </div>
                """
            )

            with gr.Tabs():

                # ---------------- About tab ----------------

                with gr.Tab("ℹ️ About LEDGER"):
                    gr.HTML(
                        """
                        <div class="about-card">
                            <h2>📒 What is Project LEDGER?</h2>
                            <p>
                                LEDGER is a financial-document
                                intelligence agent: upload real
                                financial-report PDFs, then ask
                                natural-language questions across
                                your entire indexed corpus. Every
                                answer comes back grounded and
                                cited to a real page — never fabricated.
                            </p>
                        </div>

                        <div class="about-card">
                            <h2>🧭 What each tab does</h2>

                            <div class="about-feature">
                                <div class="icon">💬</div>
                                <div>
                                    <div class="label">Ask LEDGER</div>
                                    <div class="desc">
                                        Ask a question across all indexed
                                        documents (or scope it to one).
                                        Every answer shows its citation,
                                        with the source page in the
                                        Evidence Inspector.
                                    </div>
                                </div>
                            </div>

                            <div class="about-feature">
                                <div class="icon">📁</div>
                                <div>
                                    <div class="label">Documents</div>
                                    <div class="desc">
                                        Upload a raw PDF — it's routed
                                        through orchestrator-api →
                                        doc-processor-api →
                                        retrieval-api for real OCR/layout
                                        extraction and indexing.
                                    </div>
                                </div>
                            </div>

                            <div class="about-feature">
                                <div class="icon">📊</div>
                                <div>
                                    <div class="label">Dashboard</div>
                                    <div class="desc">
                                        See how many documents/questions
                                        this session has processed,
                                        recent query latency, detected
                                        tables, extracted values, and
                                        the full session log.
                                    </div>
                                </div>
                            </div>

                            <div class="about-feature">
                                <div class="icon">🩺</div>
                                <div>
                                    <div class="label">System Health</div>
                                    <div class="desc">
                                        Check whether orchestrator-api
                                        and its downstream services are
                                        reachable right now.
                                    </div>
                                </div>
                            </div>
                        </div>
                        """
                    )

                # ---------------- Chat tab ----------------

                with gr.Tab("💬 Ask LEDGER"):

                    with gr.Row():

                        with gr.Column(scale=3):

                            chatbot = gr.Chatbot(
                                label="Corpus-wide Q&A",
                                type="messages",
                                height=460,
                                avatar_images=(None, "🤖"),
                            )

                            with gr.Row():

                                question_box = gr.Textbox(
                                    placeholder=(
                                        "e.g. What was the operating "
                                        "income reported in 2020?"
                                    ),
                                    label="Your question",
                                    scale=4,
                                    autofocus=True,
                                )

                                scope_dropdown = gr.Dropdown(
                                    choices=["Entire corpus"],
                                    value="Entire corpus",
                                    label="Scope (optional)",
                                    scale=2,
                                )

                            with gr.Row():

                                ask_btn = gr.Button(
                                    "Ask",
                                    variant="primary",
                                    scale=1,
                                )

                                clear_btn = gr.Button(
                                    "Clear conversation",
                                    scale=1,
                                )

                            live_stats = gr.HTML(
                                self._stats_html()
                            )

                        with gr.Column(scale=2):

                            gr.Markdown(
                                "### 🔍 Evidence Inspector"
                            )

                            gr.Markdown(
                                "Shows the source PDF page for the "
                                "last answer's citations, with the "
                                "matching text highlighted."
                            )

                            evidence_gallery = gr.Gallery(
                                label="Cited pages",
                                columns=1,
                                height=380,
                                object_fit="contain",
                            )

                            evidence_notes = gr.Markdown(
                                "",
                                elem_classes=["evidence-note"],
                            )

                    ask_btn.click(
                        self.on_ask,
                        inputs=[
                            question_box,
                            scope_dropdown,
                            chatbot,
                        ],
                        outputs=[
                            chatbot,
                            question_box,
                            evidence_gallery,
                            evidence_notes,
                            live_stats,
                        ],
                    )

                    question_box.submit(
                        self.on_ask,
                        inputs=[
                            question_box,
                            scope_dropdown,
                            chatbot,
                        ],
                        outputs=[
                            chatbot,
                            question_box,
                            evidence_gallery,
                            evidence_notes,
                            live_stats,
                        ],
                    )

                    clear_btn.click(
                        self.on_clear_chat,
                        outputs=[
                            chatbot,
                            evidence_gallery,
                            evidence_notes,
                        ],
                    )

                # ---------------- Documents tab ----------------

                with gr.Tab("📁 Documents"):

                    gr.Markdown(
                        "Upload raw financial-report PDFs. Each file "
                        "is sent to **orchestrator-api → "
                        "doc-processor-api → retrieval-api** for "
                        "OCR/layout extraction and indexing — the same "
                        "production ingestion path used for the whole "
                        "corpus, never a shortcut through the dataset's "
                        "pre-parsed JSON."
                    )

                    with gr.Row():

                        with gr.Column(scale=1):

                            file_uploader = gr.File(
                                label="PDF file(s)",
                                file_types=[".pdf"],
                                file_count="multiple",
                            )

                            upload_btn = gr.Button(
                                "Upload & Index",
                                variant="primary",
                            )

                            upload_log = gr.Markdown("")

                        with gr.Column(scale=1):

                            gr.Markdown(
                                "**First-page previews**"
                            )

                            thumb_gallery = gr.Gallery(
                                label="Indexed this session",
                                columns=2,
                                height=380,
                                object_fit="contain",
                            )

                    gr.Markdown(
                        "### Indexed documents (this session)"
                    )

                    documents_table = gr.Dataframe(
                        value=self._documents_dataframe(),
                        interactive=False,
                        wrap=True,
                    )

                    upload_btn.click(
                        self.on_upload,
                        inputs=[file_uploader],
                        outputs=[
                            upload_log,
                            documents_table,
                            thumb_gallery,
                            scope_dropdown,
                            live_stats,
                        ],
                    )

                # ---------------- Dashboard tab ----------------

                with gr.Tab("📊 Dashboard"):

                    dash_stats = gr.HTML(
                        self._stats_html()
                    )

                    with gr.Row():

                        refresh_btn = gr.Button(
                            "🔄 Refresh"
                        )

                        reset_btn = gr.Button(
                            "🧹 Reset session data",
                            variant="stop",
                        )

                    gr.Markdown(
                        "### Recent queries & latency"
                    )

                    latency_plot = gr.LinePlot(
                        value=self._latency_plot_dataframe(),
                        x="query #",
                        y="latency_ms",
                        y_title="latency (ms)",
                        height=260,
                    )

                    queries_table = gr.Dataframe(
                        value=self._queries_dataframe(),
                        interactive=False,
                        wrap=True,
                    )

                    gr.Markdown(
                        "### Indexed documents"
                    )

                    dash_documents_table = gr.Dataframe(
                        value=self._documents_dataframe(),
                        interactive=False,
                        wrap=True,
                    )

                    gr.Markdown(
                        "### Detected tables & extracted values"
                    )

                    dash_tables_table = gr.Dataframe(
                        value=self._detected_tables_dataframe(),
                        interactive=False,
                        wrap=True,
                    )

                    refresh_btn.click(
                        self.on_refresh_dashboard,
                        outputs=[
                            dash_stats,
                            dash_documents_table,
                            queries_table,
                            latency_plot,
                            dash_tables_table,
                        ],
                    )

                    reset_btn.click(
                        self.on_reset_session,
                        outputs=[
                            dash_stats,
                            dash_documents_table,
                            queries_table,
                            latency_plot,
                            scope_dropdown,
                            dash_tables_table,
                        ],
                    )

                # ---------------- Health tab ----------------

                with gr.Tab("🩺 System Health"):

                    gr.Markdown(
                        f"Pings **orchestrator-api** at "
                        f"`{self.settings.ORCHESTRATOR_URL}`, which "
                        "in turn reports the reachability of "
                        "agent-service and answer-validator-api."
                    )

                    health_btn = gr.Button(
                        "Check health",
                        variant="primary",
                    )

                    health_output = gr.HTML(
                        'Click "Check health" to ping the pipeline.'
                    )

                    health_btn.click(
                        self.on_check_health,
                        outputs=[health_output],
                    )

            blocks.load(
                self.on_refresh_dashboard,
                outputs=[
                    dash_stats,
                    dash_documents_table,
                    queries_table,
                    latency_plot,
                    dash_tables_table,
                ],
            )

        return blocks
