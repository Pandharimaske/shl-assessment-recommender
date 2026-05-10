"""
Local BM25 retriever — provides fast keyword search over the SHL catalog.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.catalog.loader import get_catalog
from typing import Optional

_bm25: Optional[BM25Retriever] = None

def _create_bm25_docs(catalog: list[dict]) -> list[Document]:
    docs = []
    for item in catalog:
        levels = " ".join(item.get("job_levels", []))
        keys = " ".join(item.get("keys", []))
        content = f"{item['name']} {item['description']} {levels} {keys}"
        
        docs.append(Document(
            page_content=content,
            metadata={
                "id": item.get("entity_id", ""),
                "name": item.get("name", ""),
                "url": item.get("url", ""),
                "test_type": item.get("test_type", ""),
                "description": item.get("description", ""),
                "job_levels": item.get("job_levels", []),
                "duration": item.get("duration", ""),
                "keys": item.get("keys", []),
            }
        ))
    return docs

async def init_bm25_retriever():
    """Explicitly initialize the BM25 retriever at startup."""
    global _bm25
    if _bm25 is None:
        catalog = get_catalog()
        docs = _create_bm25_docs(catalog)
        _bm25 = BM25Retriever.from_documents(docs)
        _bm25.k = 10
    return _bm25

def get_bm25_retriever() -> BM25Retriever:
    """Sync fallback getter for the BM25 retriever."""
    global _bm25
    if _bm25 is None:
        catalog = get_catalog()
        docs = _create_bm25_docs(catalog)
        _bm25 = BM25Retriever.from_documents(docs)
        _bm25.k = 10
        
    return _bm25

def query_bm25(query: str, k: int = 10) -> list[dict]:
    """Returns list of matches: [{id, score, metadata}]"""
    retriever = get_bm25_retriever()
    retriever.k = k
    docs = retriever.invoke(query)
    
    return [
        {
            "id": d.metadata["id"],
            "score": 1.0, # BM25Retriever doesn't return scores easily in invoke
            "name": d.metadata["name"],
            "url": d.metadata["url"],
            "test_type": d.metadata["test_type"],
            "description": d.metadata["description"],
            "job_levels": d.metadata["job_levels"],
            "duration": d.metadata["duration"],
            "keys": d.metadata["keys"],
        }
        for d in docs
    ]
