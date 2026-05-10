"""
Pydantic schemas — API request/response models.
Schema is NON-NEGOTIABLE as per assignment spec.
"""
from pydantic import BaseModel, field_validator
from typing import Optional


# ── Request ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    # Optional: client passes back the last shortlist so the server can return
    # it verbatim on user confirmation without re-running retrieval.
    previous_recommendations: list[dict] = []

    @field_validator("messages")
    @classmethod
    def must_have_messages(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


# ── Response ─────────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # e.g. "K" or "K,S" or "P"


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]   # [] when clarifying/refusing
    end_of_conversation: bool


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
