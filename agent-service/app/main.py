import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, AliasChoices

from app.llm.client import LLMClient
from app.graph import build_graph

app = FastAPI(title="agent-service", version="0.1.0")
_graph = build_graph()


class QueryRequest(BaseModel):
    trace_id: str = "trace_default"
    query: str = Field(..., validation_alias=AliasChoices('query', 'question'))
    top_k: int = 5
    context: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None


class AnswerResponse(BaseModel):
    answer: dict


def safe_eval_formula(formula_str: str) -> Optional[float]:
    """Safely evaluates basic math formulas if 'value' is missing."""
    try:
        # Strip out any non-arithmetic characters for safety
        cleaned = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', formula_str)
        if cleaned.strip():
            return float(eval(cleaned, {"__builtins__": None}, {}))
    except Exception:
        pass
    return None


def format_ledger_payload(raw_answer: Any) -> dict:
    """Ensures raw graph output strictly conforms to LEDGER Answer Schema."""
    if not isinstance(raw_answer, dict):
        return {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": str(raw_answer)}
        }

    # Already formatted properly according to contract
    if "answer_type" in raw_answer and "params" in raw_answer:
        params = raw_answer["params"]
        if raw_answer["answer_type"] == "calculated" and (params.get("value") is None or params.get("value") == ""):
            if params.get("formula"):
                params["value"] = safe_eval_formula(str(params["formula"]))
        return raw_answer

    evidence = raw_answer.get("evidence", [])

    # Map flat schema with 'formula' to 'calculated'
    if raw_answer.get("formula"):
        formula_str = str(raw_answer.get("formula"))
        val = raw_answer.get("value")
        if val is None or val == "":
            val = safe_eval_formula(formula_str)

        return {
            "answer_type": "calculated",
            "evidence": evidence,
            "params": {
                "value": val,
                "formula": formula_str
            }
        }

    # Map flat schema with 'reason' to 'insufficient_evidence'
    if raw_answer.get("reason") or raw_answer.get("answer_type") == "insufficient_evidence":
        return {
            "answer_type": "insufficient_evidence",
            "evidence": evidence,
            "params": {"reason": raw_answer.get("reason", "Insufficient evidence provided.")}
        }

    # Map flat schema with 'values' array to 'multi_span'
    if raw_answer.get("values"):
        return {
            "answer_type": "multi_span",
            "evidence": evidence,
            "params": {"values": raw_answer.get("values")}
        }

    # Map flat schema with 'value' to 'direct'
    if raw_answer.get("value"):
        return {
            "answer_type": "direct",
            "evidence": evidence,
            "params": {"value": raw_answer.get("value")}
        }

    # Default fallback
    return {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {"reason": "Could not extract grounded answer from context."}
    }


def normalize_evidence(items: Any) -> list[dict]:
    """Convert retrieval evidence into strict-schema citations."""
    citations = []
    seen = set()
    for item in items or []:
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
    return citations


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "agent-service"}


@app.post("/query", response_model=AnswerResponse)
async def ask(request: QueryRequest):
    ctx = request.context or ""
    docs = request.documents or ([{"content": ctx, "text": ctx}] if ctx else [])
    evidence_list = [ctx] if ctx else []

    initial_state = {
        "trace_id": request.trace_id,
        "query": request.query,
        "question": request.query,
        "top_k": request.top_k,
        "context": ctx,
        "evidence": evidence_list,
        "documents": docs,
    }
    
    result = await _graph.ainvoke(initial_state)
    
    raw_answer = (
        result.get("final_payload")
        or result.get("final_answer")
        or result.get("draft")
        or result
    )
    
    formatted_answer = format_ledger_payload(raw_answer)
    if (
        formatted_answer.get("answer_type") != "insufficient_evidence"
        and not formatted_answer.get("evidence")
    ):
        formatted_answer["evidence"] = normalize_evidence(result.get("evidence"))

    if (
        formatted_answer.get("answer_type") != "insufficient_evidence"
        and not formatted_answer.get("evidence")
    ):
        formatted_answer = {
            "answer_type": "insufficient_evidence",
            "evidence": [],
            "params": {"reason": "The agent produced an answer without grounded evidence."},
        }
    return AnswerResponse(answer=formatted_answer)
