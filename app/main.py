"""
FastAPI application entry point.

Startup sequence:
  1. Load SHL catalog from JSON into memory
  2. Compile LangGraph agent graph
  3. Serve requests
"""
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
    print("Loading SHL catalog...")
    catalog = get_catalog()
    print(f"Catalog loaded: {len(catalog)} assessments")

    print("Compiling agent graph...")
    get_graph()
    
    print("Pre-initializing BM25...")
    from app.retriever.bm25_retriever import init_bm25_retriever
    await init_bm25_retriever()
    
    print("Agent graph and retrievers ready.")

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
