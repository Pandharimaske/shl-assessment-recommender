"""
LangGraph agent state definition.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    messages: list[dict]

    turn_count: int

    intent: str

    conversation_context: dict

    missing_fields: list[str]

    retrieved_items: list[dict]

    previous_recommendations: list[dict]

    reply: str
    recommendations: list[dict]
    end_of_conversation: bool
