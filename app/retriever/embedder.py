"""
Embedder — local sentence-transformers model (all-MiniLM-L6-v2).
No network call per request. Model loaded once at startup, offloaded to
thread executor so the async event loop is never blocked.
"""
import asyncio
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def preload_model() -> None:
    """Call at startup to avoid cold-start latency on the first request."""
    _get_model()
    print(f"Embedding model '{_MODEL_NAME}' loaded.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (CPU-bound, synchronous)."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


async def aembed_query(text: str) -> list[float]:
    """Async embed — offloads CPU work to executor so event loop stays free."""
    if not text:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query, text)


def build_catalog_text(item: dict) -> str:
    """
    Create a rich text representation of a catalog item for embedding.
    Combines name + description + job_levels + keys.
    """
    parts = [item["name"]]
    if item.get("description"):
        parts.append(item["description"])
    if item.get("job_levels"):
        parts.append("Job levels: " + ", ".join(item["job_levels"]))
    if item.get("keys"):
        parts.append("Category: " + ", ".join(item["keys"]))
    if item.get("languages"):
        parts.append("Languages: " + ", ".join(item["languages"][:5]))
    return " | ".join(parts)
