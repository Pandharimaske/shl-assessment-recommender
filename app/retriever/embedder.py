"""
Embedder — fastembed (ONNX-based, no PyTorch).
Uses all-MiniLM-L6-v2 via fastembed: ~80 MB RAM vs ~400 MB for sentence-transformers+torch.
Model is baked into the Docker image at build time for zero cold-start download latency.
Async queries offload to thread executor so the event loop is never blocked.
"""
import asyncio
from fastembed import TextEmbedding

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def preload_model() -> None:
    """Call at startup (in executor) to warm the model before first request."""
    _get_model()
    print(f"fastembed model '{_MODEL_NAME}' loaded.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    if not texts:
        return []
    model = _get_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


async def aembed_query(text: str) -> list[float]:
    """Async embed — offloads to executor so the event loop stays unblocked."""
    if not text:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query, text)


def build_catalog_text(item: dict) -> str:
    """
    Rich text representation of a catalog item for embedding.
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
