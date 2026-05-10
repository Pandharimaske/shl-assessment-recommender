"""
FastAPI application entry point.

Startup sequence (all heavy work in executor to avoid blocking the event loop):
  1. Load SHL catalog from JSON into memory
  2. Pre-load embedding model (local sentence-transformers, CPU)
  3. Pre-initialize Pinecone index connection
  4. Initialize BM25 retriever (CPU-bound, runs in executor)
  5. Compile LangGraph agent graph
  6. Serve requests
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

from app.api.routes import router
from app.catalog.loader import get_catalog
from app.agent.graph import get_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    print("Loading SHL catalog...")
    catalog = get_catalog()
    print(f"Catalog loaded: {len(catalog)} assessments")

    print("Pre-loading embedding model (local)...")
    from app.retriever.embedder import preload_model
    await loop.run_in_executor(None, preload_model)
    print("Embedding model ready.")

    print("Pre-initializing Pinecone connection...")
    from app.retriever.pinecone_client import _get_index
    await loop.run_in_executor(None, _get_index)
    print("Pinecone ready.")

    print("Pre-initializing BM25 retriever...")
    from app.retriever.bm25_retriever import get_bm25_retriever
    await loop.run_in_executor(None, get_bm25_retriever)
    print("BM25 ready.")

    print("Compiling LangGraph agent...")
    get_graph()
    print("Agent ready. Serving requests.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments based on role requirements.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (for any frontend or evaluator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)
