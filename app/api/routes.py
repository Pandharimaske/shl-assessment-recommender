"""
FastAPI routes — /health and /chat endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.agent.graph import run_agent

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Readiness check — called by evaluator before tests."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    Receives full conversation history, returns next agent reply + recommendations.
    """
    # Convert Pydantic Message objects to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Basic sanity check — last message must be from user
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=422,
            detail="Last message in history must be from the user."
        )

    try:
        result = await run_agent(messages, previous_recommendations=request.previous_recommendations)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Never crash — return a graceful fallback
        return ChatResponse(
            reply="I encountered an issue. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Build typed recommendation list
    recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r["test_type"],
        )
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result["end_of_conversation"],
    )
