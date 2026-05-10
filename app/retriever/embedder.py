"""
Embedder — wraps Hugging Face Inference API for catalog and query embedding.
Uses hosted API to save memory (no local PyTorch required).
"""
from huggingface_hub import InferenceClient, AsyncInferenceClient
from app.core.config import get_settings

_client: InferenceClient | None = None
_aclient: AsyncInferenceClient | None = None

def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = InferenceClient(token=settings.hf_token)
    return _client

def _get_aclient() -> AsyncInferenceClient:
    global _aclient
    if _aclient is None:
        settings = get_settings()
        _aclient = AsyncInferenceClient(token=settings.hf_token)
    return _aclient


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed catalog items (passages)."""
    if not texts:
        return []
    
    settings = get_settings()
    client = _get_client()
    
    # feature_extraction natively handles list of texts and returns list of lists
    response = client.feature_extraction(
        texts,
        model=settings.embed_model
    )
    # the InferenceClient returns a numpy array-like object (actually a list if it's parsed from JSON, or numpy if installed)
    # usually it returns a numpy array, so we convert it to list of floats
    if hasattr(response, "tolist"):
        return response.tolist()
    return response


def embed_query(text: str) -> list[float]:
    """Embed user search queries."""
    return embed_texts([text])[0]


async def aembed_query(text: str) -> list[float]:
    """Async embed user search queries."""
    if not text:
        return []
    
    settings = get_settings()
    client = _get_aclient()
    
    response = await client.feature_extraction(
        text,
        model=settings.embed_model
    )
    if hasattr(response, "tolist"):
        return response.tolist()
    return response


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
