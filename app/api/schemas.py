"""
Pydantic schemas — API request/response models.
Schema is NON-NEGOTIABLE as per assignment spec.
"""
from pydantic import BaseModel, field_validator
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    previous_recommendations: list[dict] = []

    @field_validator("messages")
    @classmethod
    def must_have_messages(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool



class HealthResponse(BaseModel):
    status: str
