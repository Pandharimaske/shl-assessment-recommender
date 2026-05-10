"""
One-time script: embed the SHL catalog and upsert all vectors into Pinecone.

Run ONCE before starting the server:
    uv run python scripts/ingest_catalog.py

Creates Pinecone index if it doesn't exist.
"""
import sys
import os

# Ensure app/ is importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pinecone import Pinecone, ServerlessSpec
from app.catalog.loader import get_catalog
from app.retriever.embedder import embed_texts, build_catalog_text
from app.core.config import get_settings


def create_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int):
    existing = pc.list_indexes().names()  # v6+: returns IndexList, not a plain list
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dim={dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{index_name}' already exists.")


def ingest():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    EMBED_DIM = 384

    create_index_if_not_exists(pc, settings.pinecone_index, EMBED_DIM)
    index = pc.Index(settings.pinecone_index)

    catalog = get_catalog()
    print(f"Embedding {len(catalog)} catalog items...")

    # Build text representations
    texts = [build_catalog_text(item) for item in catalog]

    # Embed in batches of 64 (sentence-transformers default)
    batch_size = 64
    all_vectors = []

    for i in range(0, len(catalog), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_items = catalog[i : i + batch_size]
        embeddings = embed_texts(batch_texts)

        for item, emb in zip(batch_items, embeddings):
            all_vectors.append({
                "id": str(item["entity_id"]),
                "values": emb,
                "metadata": {
                    "name":        item["name"],
                    "url":         item["url"],
                    "test_type":   item["test_type"],
                    "description": item["description"][:500],  # Pinecone metadata limit
                    "job_levels":  item["job_levels"],
                    "duration":    item["duration"],
                    "keys":        item["keys"],
                    "remote":      item["remote"],
                    "adaptive":    item["adaptive"],
                },
            })

        print(f"  Embedded {min(i + batch_size, len(catalog))}/{len(catalog)}")

    # Upsert to Pinecone in batches of 100
    print("Upserting to Pinecone...")
    upsert_batch = 100
    for i in range(0, len(all_vectors), upsert_batch):
        index.upsert(vectors=all_vectors[i : i + upsert_batch])
        print(f"  Upserted {min(i + upsert_batch, len(all_vectors))}/{len(all_vectors)}")

    stats = index.describe_index_stats()
    print(f"\nDone! Pinecone index now has {stats.total_vector_count} vectors.")


if __name__ == "__main__":
    ingest()
