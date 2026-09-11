from typing import Any, Dict, List

from app.config import get_settings

def _api_url() -> str:
    # Accept the old example value ending in /search, but keep the canonical
    # setting as the service base URL.
    return get_settings().RETRIEVAL_API_URL.rstrip("/").removesuffix("/search")

def _mock() -> bool:
    return get_settings().RETRIEVAL_MOCK

def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _mock():
        return _mock_post(payload)
    import httpx
    url = f"{_api_url()}/search"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

def _mock_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "").lower()
    if any(term in query for term in ("cts", "jabil", "operating income", "finished goods")):
        return {"evidence": [{"document_id": "doc_041", "chunk_id": "mock_001", "rank": 1, "page_number": 2, "section": "Balance Sheet", "content_type": "table", "content": "Finished goods: 314,258 (thousand)", "score": 0.92}]}
    return {"evidence": []}

def _search(query: str, top_k: int, trace_id: str) -> List[Dict[str, Any]]:
    data = _post({"trace_id": trace_id, "query": query, "top_k": top_k})
    return data.get("evidence", [])

def search_documents(query: str, top_k: int = 5, trace_id: str = "trace_agent") -> List[Dict[str, Any]]:
    return _search(query, top_k, trace_id)

def search_tables(query: str, top_k: int = 5, trace_id: str = "trace_agent") -> List[Dict[str, Any]]:
    # Retrieval has one hybrid /search route. Ask for extra candidates before
    # filtering so table-only searches can still return top_k table chunks.
    candidates = _search(query, min(max(top_k * 3, top_k), 30), trace_id)
    return [item for item in candidates if item.get("content_type") == "table"][:top_k]

def filter_documents(metadata: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Compatibility helper backed by the retrieval service's hybrid search."""
    query = " ".join(str(value) for value in metadata.values() if value is not None)
    return _search(query, top_k, "trace_filter") if query else []
