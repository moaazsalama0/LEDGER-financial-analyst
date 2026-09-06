from typing import Any, Dict, List

from app.config import get_settings

def _api_url() -> str:
    return get_settings().retrieval_api_url

def _mock() -> bool:
    return get_settings().retrieval_mock

def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if _mock():
        return _mock_post(path, payload)
    import httpx
    url = f"{_api_url()}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

def _mock_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "").lower()
    if any(term in query for term in ("cts", "jabil", "operating income", "finished goods")):
        return {"results": [{"document_id": "doc_041", "page": 2, "section": "Balance Sheet", "content_type": "table", "snippet": "Finished goods: 314,258 (thousand)", "score": 0.92}]}
    return {"results": []}

def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    data = _post("/search/documents", {"query": query, "top_k": top_k})
    return data.get("results", [])

def search_tables(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    data = _post("/search/tables", {"query": query, "top_k": top_k})
    return data.get("results", [])

def filter_documents(metadata: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    data = _post("/filter", {"metadata": metadata, "top_k": top_k})
    return data.get("results", [])
