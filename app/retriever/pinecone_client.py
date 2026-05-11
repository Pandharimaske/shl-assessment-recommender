"""
Pinecone client — upsert catalog embeddings and query at runtime.
"""
import threading
from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings


_pc: Pinecone | None = None
_index = None
_init_lock = threading.Lock()


def _get_index():
    global _pc, _index
    if _index is None:
        with _init_lock:
            if _index is None:
                settings = get_settings()
                _pc = Pinecone(api_key=settings.pinecone_api_key)
                _index = _pc.Index(settings.pinecone_index)
    return _index


def upsert_catalog(vectors: list[dict]):
    """
    vectors: list of {id, values, metadata}
    metadata must include: name, url, test_type, description, job_levels, duration
    """
    index = _get_index()
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
    print(f"Upserted {len(vectors)} vectors to Pinecone.")


def query_catalog(
    query_vector: list[float],
    top_k: int = 20,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Returns list of matches: [{id, score, metadata}]
    """
    index = _get_index()
    kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
    if filter_dict:
        kwargs["filter"] = filter_dict
    
    result = index.query(**kwargs)
    return [
        {
            "id":       m.id,
            "score":    m.score,
            "name":     m.metadata.get("name", ""),
            "url":      m.metadata.get("url", ""),
            "test_type": m.metadata.get("test_type", ""),
            "description": m.metadata.get("description", ""),
            "job_levels": m.metadata.get("job_levels", []),
            "duration": m.metadata.get("duration", ""),
            "keys":     m.metadata.get("keys", []),
        }
        for m in result.matches
    ]
