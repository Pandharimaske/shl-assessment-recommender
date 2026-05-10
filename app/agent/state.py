"""
LangGraph agent state definition.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Full conversation history passed in by the API
    messages: list[dict]

    # User turn counter
    turn_count: int

    # Intent: pending | clarify | recommend | force_recommend | compare | refuse
    intent: str

    # Structured context extracted from conversation (new)
    conversation_context: dict

    # Which fields are missing when clarifying (new)
    missing_fields: list[str]

    # Top-25 candidates from Pinecone
    retrieved_items: list[dict]

    # Last confirmed shortlist — passed in by the client so EOC node can
    # return it verbatim without triggering a new retrieval + rerank cycle.
    previous_recommendations: list[dict]

    # Final output fields
    reply: str
    recommendations: list[dict]
    end_of_conversation: bool
