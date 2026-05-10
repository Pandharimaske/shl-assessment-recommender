"""
LangGraph StateGraph — wires all agent nodes together.

Graph flow:
  START
    └─► guard_node
          ├─ "refuse"  ──► refuse_node  ──► END
          └─ "pending" ──► eoc_node
                            ├─ "eoc"    ──► END   (user confirmed, return existing shortlist)
                            └─ "pending" ──► intent_node
                                              ├─ "clarify"         ──► clarify_node   ──► END
                                              ├─ "compare"         ──► compare_node   ──► END
                                              ├─ "recommend"       ──► retrieve_node
                                              │                          └──► recommend_node ──► END
                                              └─ "force_recommend" ──► retrieve_node
                                                                         └──► recommend_node ──► END
"""
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    supervisor_node, clarify_node, retrieve_node, 
    recommend_node, compare_node, refuse_node,
)

def route_after_supervisor(state: AgentState) -> str:
    """Routes based on the supervisor's analysis (safety, eoc, or user intent)."""
    intent = state["intent"]
    if intent == "refuse":
        return "refuse"
    if intent == "eoc":
        return "eoc_end"
    if intent == "clarify":
        return "clarify"
    if intent == "compare":
        return "compare"
    if intent in ("recommend", "force_recommend"):
        return "retrieve"
    return "clarify"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("clarify",    clarify_node)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("recommend",  recommend_node)
    graph.add_node("compare",    compare_node)
    graph.add_node("refuse",     refuse_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {
            "refuse": "refuse", 
            "eoc_end": END, 
            "clarify": "clarify", 
            "compare": "compare", 
            "retrieve": "retrieve"
        },
    )

    graph.add_edge("retrieve",  "recommend")
    graph.add_edge("clarify",   END)
    graph.add_edge("recommend", END)
    graph.add_edge("compare",   END)
    graph.add_edge("refuse",    END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public entry point ────────────────────────────────────────────────────────

async def run_agent(messages: list[dict], previous_recommendations: list[dict] | None = None) -> dict:
    """
    Entry point called by POST /chat.
    """
    from app.agent.nodes import _extract_previous_recommendations
    
    if not previous_recommendations:
        previous_recommendations = _extract_previous_recommendations(messages)

    initial_state: AgentState = {
        "messages":                  messages,
        "turn_count":                0,
        "intent":                    "pending",
        "conversation_context":      {},
        "missing_fields":            [],
        "retrieved_items":           [],
        "previous_recommendations":  previous_recommendations,
        "reply":                     "",
        "recommendations":           [],
        "end_of_conversation":       False,
    }
    final_state = await get_graph().ainvoke(initial_state)
    return {
        "reply":               final_state["reply"],
        "recommendations":     final_state["recommendations"],
        "end_of_conversation": final_state["end_of_conversation"],
    }
