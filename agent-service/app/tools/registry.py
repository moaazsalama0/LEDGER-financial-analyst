from __future__ import annotations

import ast

if not hasattr(ast, "Mul"):
    setattr(ast, "Mul", getattr(ast, "Mult", None))

from typing import Any, Dict

from app.schemas.tools import SearchRequest, FilterRequest, CalculateRequest
from app.tools.calculator import calculate
from app.tools import retrieval

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "search_documents": {
        "func": retrieval.search_documents,
        "input_schema": SearchRequest,
        "description": "Semantic (vector) search over the document corpus.",
    },
    "search_tables": {
        "func": retrieval.search_tables,
        "input_schema": SearchRequest,
        "description": "Table-specific search over the document corpus.",
    },
    "calculate": {
        "func": calculate,
        "input_schema": CalculateRequest,
        "description": "Deterministic arithmetic evaluator (safe Python math only).",
    },
    "filter_documents": {
        "func": retrieval.filter_documents,
        "input_schema": FilterRequest,
        "description": "Filter documents by metadata (company, period, metric, etc.).",
    },
}
