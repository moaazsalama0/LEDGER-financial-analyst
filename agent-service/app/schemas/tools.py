"""Pydantic I/O schemas for the four deterministic tools.

These double as the JSON-schema contracts for the tool registry and for the
retrieval-api HTTP client.
"""
from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results")


class SearchResult(BaseModel):
    document_id: str
    page: int
    section: str | None = None
    content_type: str = "text"  # text | table | mixed
    snippet: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult]


class FilterRequest(BaseModel):
    metadata: dict[str, Any] = Field(description="Structured metadata filter (company, period, metric)")
    top_k: int = Field(default=5, ge=1, le=50)


class CalculateRequest(BaseModel):
    expression: str = Field(description="Safe Python arithmetic expression, e.g. (9447-314258) or abs(9447-314258)")


class CalculateResponse(BaseModel):
    expression: str
    value: float
