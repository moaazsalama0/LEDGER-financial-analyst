"""LangGraph workflow for the LEDGER agent-service (async‑aware)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Union

from app.config import get_settings
from app.state import AgentState
from app.tools.registry import TOOL_REGISTRY
from app.tools.retrieval import search_documents, search_tables, filter_documents
from app.tools.calculator import calculate
from app.llm.client import LLMClient
from app.schemas.answer_schema import (
    StrictAnswer,
    DirectAnswer,
    CalculatedAnswer,
    MultiSpanAnswer,
    InsufficientEvidenceAnswer,
)
from pydantic import BaseModel, Field, ValidationError
from pydantic import TypeAdapter, ValidationError

# ---------------------------------------------------------------------------
# Prompt templates (kept in sync with config.py / .env.example)
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "You are the classifier of a financial‑document RAG agent.  "
    "Given a user question, output JSON with exactly these keys:\n"
    '  "query_type": one of "direct", "calculated", "multi_span", "insufficient_evidence"\n'
    '  "entities": {"company": str|null, "period": str|null, "metric": str|null, "table_hint": bool}\n'
    "Return only valid JSON."
)

REASON_SYSTEM = (
    "You are the reasoning node of a financial‑document RAG agent.  "
    "Given a question and retrieved evidence, output JSON.\n"
    'If the question is a calculation, set "formula" to a Python expression using '
    "only the numeric values found in the evidence (operators + - * / // % "
    "and functions abs, round, min, max, sum, sqrt, floor, ceil, log, exp, pow).  "
    'Do NOT compute the result yourself; leave "value": null.\n'
    'For a direct fact, set "value" to the exact value from the evidence.\n'
    'For multi‑span, set "values" to the ordered list of items.\n'
    'If evidence is insufficient, set "reason" to a short explanation.\n'
    "Never invent numbers that are not present in the evidence."
)


# ---------------------------------------------------------------------------
# Evidence Helpers (Type-safe for dict or str)
# ---------------------------------------------------------------------------
def _get_deduped_evidence(evidence_list: list) -> list:
    """Deduplicate evidence list supporting both dict and string elements."""
    seen = set()
    deduped = []
    for e in evidence_list:
        if isinstance(e, dict):
            doc_id = e.get("document_id")
            page = e.get("page", e.get("page_number"))
            content = e.get("content") or e.get("text") or str(e)
            key = (doc_id, page, content)
        else:
            key = str(e)

        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


def _format_evidence_item(i: int, e: Any) -> str:
    """Format individual evidence item into readable string for LLM reasoning."""
    if isinstance(e, dict):
        doc_id = e.get("document_id", "N/A")
        page = e.get("page", e.get("page_number", "N/A"))
        content = e.get("content") or e.get("text") or str(e)
        return f"[{i}] doc={doc_id} page={page} | {content}"
    return f"[{i}] {e}"


# ---------------------------------------------------------------------------
# Helper: run an LLM call and guard against non‑JSON output
# ---------------------------------------------------------------------------
async def _llm_chat(
    prompt: str,
    system_prompt: str | None = None,
    response_schema: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    try:
        llm = LLMClient()
        return await llm.chat_json(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover
        # Return a minimal safe dict so the graph can still make a decision
        return {"query_type": "direct", "entities": {}, "draft": {}, "reason": f"LLM error: {exc}"}


# ---------------------------------------------------------------------------
# Node: classify_question  (LLM‑based type + entity extraction)
# ---------------------------------------------------------------------------
async def _classify_question(state: AgentState) -> AgentState:
    out = await _llm_chat(
        prompt=state["question"],
        system_prompt=CLASSIFY_SYSTEM,
    )
    return {
        "query_type": out.get("query_type", "direct"),
        "entities": out.get("entities", {}),
        "search_query": state.get("search_query") or state["question"],
        "tool_calls": [*state.get("tool_calls", []), {"step": "classify_question"}],
    }


# ---------------------------------------------------------------------------
# Node: retrieve_documents  (sync – no LLM involvement)
# ---------------------------------------------------------------------------
def _retrieve_documents(state: AgentState) -> AgentState:
    query = state.get("search_query") or state["question"]
    try:
        results = search_documents(
            query,
            get_settings().TOP_K,
            state.get("trace_id", "trace_agent"),
        )
        error = None
    except Exception as exc:  # pragma: no cover
        results, error = [], str(exc)
    new_evidence = state.get("evidence", []) + results
    deduped = _get_deduped_evidence(new_evidence)
    return {
        "evidence": deduped,
        "tool_calls": [*state.get("tool_calls", []), {"step": "search_documents", "result": results, "error": error}],
    }


# ---------------------------------------------------------------------------
# Node: retrieve_tables  (sync – no LLM involvement)
# ---------------------------------------------------------------------------
def _retrieve_tables(state: AgentState) -> AgentState:
    query = state.get("search_query") or state["question"]
    try:
        results = search_tables(
            query,
            get_settings().TOP_K,
            state.get("trace_id", "trace_agent"),
        )
        error = None
    except Exception as exc:
        results, error = [], str(exc)
    new_evidence = state.get("evidence", []) + results
    deduped = _get_deduped_evidence(new_evidence)
    return {
        "evidence": deduped,
        "tool_calls": [*state.get("tool_calls", []), {"step": "search_tables", "result": results, "error": error}],
    }


# ---------------------------------------------------------------------------
# Node: reason  (LLM‑based draft generation + formula / values / reason)
# ---------------------------------------------------------------------------
class ReasonResponse(BaseModel):
    value: Optional[Union[str, float, int]] = None
    formula: Optional[str] = None
    values: Optional[List[Union[str, float, int]]] = None
    reason: Optional[str] = None
async def _reason(state: AgentState) -> AgentState:
    evidence_items = state.get("evidence", []) or []
    evidence_str = (
        "\n".join(_format_evidence_item(i, e) for i, e in enumerate(evidence_items))
        or "(no evidence)"
    )

    user_prompt = f"Question: {state['question']}\n\nEvidence:\n{evidence_str}"

    out = await _llm_chat(
        prompt=user_prompt,
        system_prompt=REASON_SYSTEM,
        response_schema=ReasonResponse,
    )

    return {
        "draft": out if isinstance(out, dict) else out.model_dump(),
        "tool_calls": [*state.get("tool_calls", []), {"step": "reason"}],
    }


# ---------------------------------------------------------------------------
# Node: check_evidence  (deterministic – no LLM)
# ---------------------------------------------------------------------------
def _check_evidence(state: AgentState) -> AgentState:
    qt = state.get("query_type", "direct")
    evidence = state.get("evidence") or []
    draft = state.get("draft") or {}
    retries = state.get("retries", 0)
    
    settings = get_settings()
    max_retries = getattr(settings, "max_retries", getattr(settings, "MAX_RETRIES", 3))
    
    verdict = "sufficient"
    calculated_value = None
    error = None

    if qt == "calculated":
        formula = draft.get("formula")
        if not formula:
            verdict = "retry" if retries < max_retries else "give_up"
            error = "Reasoning produced no formula."
        else:
            try:
                calculated_value = calculate(formula)
            except Exception as exc:
                verdict = "retry" if retries < max_retries else "give_up"
                error = f"Calculator rejected formula: {exc}"
    elif qt == "multi_span":
        if not draft.get("values") or not evidence:
            verdict = "retry" if retries < max_retries else "give_up"
            error = "No values or no evidence for multi‑span answer."
    elif qt == "insufficient_evidence":
        verdict = "sufficient"
    else:  # direct
        if not evidence:
            verdict = "retry" if retries < max_retries else "give_up"
            error = "No evidence retrieved for direct answer."

    return {
        "verdict": verdict,
        "calculated_value": calculated_value,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Node: retry  (broaden search query, increment counter)
# ---------------------------------------------------------------------------
def _retry(state: AgentState) -> AgentState:
    retries = (state.get("retries") or 0) + 1
    query = state.get("search_query") or state["question"]
    broadened = f"{query} (expanded retrieval attempt {retries})"
    return {
        "retries": retries,
        "search_query": broadened,
        "evidence": [],
        "draft": {},
        "verdict": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node: format_answer  (strict schema validation)
# ---------------------------------------------------------------------------
def _format_answer(state: AgentState) -> AgentState:
    draft = state.get("draft") or {}
    query_type = state.get("query_type", "direct")

    citations = []
    seen = set()
    for item in state.get("evidence") or []:
        if not isinstance(item, dict) or not item.get("document_id"):
            continue
        page = item.get("page", item.get("page_number"))
        if page is None:
            continue
        citation = {
            "document_id": str(item["document_id"]),
            "page": int(page),
            "section": item.get("section"),
        }
        key = (citation["document_id"], citation["page"], citation["section"])
        if key not in seen:
            seen.add(key)
            citations.append(citation)

    if state.get("verdict") == "give_up" or (query_type != "insufficient_evidence" and not citations):
        payload = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": state.get("error") or draft.get("reason") or "No grounded evidence was available."},
        }
    elif query_type == "calculated":
        payload = {
            "answer_type": "calculated",
            "evidence": citations,
            "params": {"value": state.get("calculated_value"), "formula": draft.get("formula")},
        }
    elif query_type == "multi_span":
        payload = {
            "answer_type": "multi_span",
            "evidence": citations,
            "params": {"values": draft.get("values")},
        }
    elif query_type == "insufficient_evidence" or draft.get("reason"):
        payload = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": draft.get("reason") or "The available evidence is insufficient."},
        }
    else:
        payload = {
            "answer_type": "direct",
            "evidence": citations,
            "params": {"value": draft.get("value")},
        }

    try:
        if isinstance(StrictAnswer, type) and hasattr(StrictAnswer, "model_validate"):
            validated = StrictAnswer.model_validate(payload)
        else:
            validated = TypeAdapter(StrictAnswer).validate_python(payload)
            
        state["final_payload"] = validated.model_dump() if hasattr(validated, "model_dump") else validated
    except Exception as e:
        state["final_payload"] = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": f"Agent output failed schema validation: {e}"},
        }
        
    return state


# ---------------------------------------------------------------------------
# Node: verify_unanswerable  (passthrough)
# ---------------------------------------------------------------------------
def _verify_unanswerable(state: AgentState) -> AgentState:
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    """Compile and return the LangGraph state graph (mixed sync/async nodes)."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)

    # --- nodes ---
    graph.add_node("classify_question", _classify_question)
    graph.add_node("retrieve_documents", _retrieve_documents)
    graph.add_node("retrieve_tables", _retrieve_tables)
    graph.add_node("reason", _reason)
    graph.add_node("check_evidence", _check_evidence)
    graph.add_node("retry", _retry)
    graph.add_node("format_answer", _format_answer)
    graph.add_node("verify_unanswerable", _verify_unanswerable)

    # --- entry point ---
    graph.set_entry_point("classify_question")

    # --- conditional flow from classification ---
    def route_by_type(state: AgentState) -> str:
        qt = state.get("query_type", "direct")
        if qt == "insufficient_evidence":
            return "verify_unanswerable"
        if qt in ("calculated", "multi_span"):
            return "hybrid"
        if state.get("entities", {}).get("table_hint"):
            return "table"
        return "text"

    graph.add_conditional_edges(
        "classify_question",
        route_by_type,
        {
            "text": "retrieve_documents",
            "table": "retrieve_tables",
            "hybrid": "retrieve_tables",
            "verify_unanswerable": "verify_unanswerable",
        },
    )

    # after retrieval nodes → reason
    for node in ("retrieve_documents", "retrieve_tables", "verify_unanswerable"):
        graph.add_edge(node, "reason")

    # reason → evidence check
    graph.add_edge("reason", "check_evidence")

    # conditional after check
    def route_after_check(state: AgentState) -> str:
        if state.get("verdict") == "retry":
            return "retry"
        return "format_answer"

    graph.add_conditional_edges(
        "check_evidence",
        route_after_check,
        {"retry": "retry", "format_answer": "format_answer"},
    )

    # retry re‑routes based on type again
    graph.add_conditional_edges(
        "retry",
        route_by_type,
        {
            "text": "retrieve_documents",
            "table": "retrieve_tables",
            "hybrid": "retrieve_tables",
            "verify_unanswerable": "verify_unanswerable",
        },
    )

    # finally → format then end
    graph.add_edge("format_answer", END)

    return graph.compile()
