"""AgentState - the single mutable state object threaded through the LangGraph.

`total=False` keeps every field optional so any node may emit only the keys it
changes; nodes must therefore read optional fields with `.get()`.
"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str                     # user question (never mutated)
    search_query: str                 # possibly broadened query used for retrieval
    query_type: str                   # direct | calculated | multi_span | insufficient_evidence
    entities: dict[str, Any]          # company / period / metric / table_hint extracted by classifier

    # --- retrieval ---
    evidence: list[dict[str, Any]]    # merged, deduplicated evidence list
    tool_calls: list[dict[str, Any]]  # full trace of tool invocations (name, args, result, error)

    # --- reasoning ---
    draft: dict[str, Any]             # raw LLM draft (value / formula / values / reason)
    calculated_value: float | None    # value produced by the deterministic calculator
    verdict: str | None               # sufficient | retry | give_up
    retries: int

    # --- output ---
    final_payload: dict[str, Any]     # strict answer schema payload for answer-validator-api
    error: str | None