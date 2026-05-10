"""
Agent nodes — each function is one node in the LangGraph StateGraph.

Key improvements over v1:
- Context extraction produces a structured ConversationContext dict
- Confidence scoring decides clarify vs recommend deterministically
- Pinecone query built from structured fields (not free text)
- Optional metadata filter by job_level passed to Pinecone
- LLM reranker uses context rubric (role/level/type match + gap fill)
"""
import re
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.schemas import AnalyzeOutput, RerankOutput, SimpleOutput
from app.agent.prompts import (
    SYSTEM_PROMPT,
    ANALYZE_PROMPT,
    CLARIFY_PROMPT,
    RERANK_PROMPT,
    COMPARE_PROMPT,
    REFUSE_PROMPT,
    HYDE_PROMPT,
    SENIORITY_TO_JOB_LEVEL,
)
from app.retriever.embedder import aembed_query
from app.retriever.pinecone_client import query_catalog
from app.catalog.loader import get_catalog
from app.core.config import get_settings


# ── LLM singletons ─────────────────────────────────────────────────────────────

_llm: BaseChatModel | None = None
_small_llm: BaseChatModel | None = None

def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        s = get_settings()
        primary_llm = ChatGroq(
            api_key=s.groq_api_key,
            model=s.groq_model,
            temperature=0.2,
            max_tokens=1024,
            max_retries=0,
        )
        
        fallbacks = []
        if s.google_api_key:
            fallbacks.append(ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=s.google_api_key, temperature=0.2, max_retries=0))
        if s.mistral_api_key:
            fallbacks.append(ChatMistralAI(api_key=s.mistral_api_key, model="mistral-small-latest", temperature=0.2, max_retries=0))
        if s.openrouter_api_key:
            fallbacks.append(ChatOpenAI(api_key=s.openrouter_api_key, base_url="https://openrouter.ai/api/v1", model=s.openrouter_model, temperature=0.2, max_retries=0))
            for backup in ["google/gemma-4-31b-it:free", "qwen/qwen3-next-80b-a3b-instruct:free"]:
                fallbacks.append(ChatOpenAI(api_key=s.openrouter_api_key, base_url="https://openrouter.ai/api/v1", model=backup, temperature=0.2, max_retries=0))

        _llm = primary_llm.with_fallbacks(fallbacks) if fallbacks else primary_llm
    return _llm

def _get_small_llm() -> BaseChatModel:
    global _small_llm
    if _small_llm is None:
        s = get_settings()
        primary_llm = ChatGroq(
            api_key=s.groq_api_key,
            model=s.groq_small_model,
            temperature=0.2,
            max_tokens=1024,
            max_retries=0,
        )
        
        fallbacks = []
        if s.google_api_key:
            fallbacks.append(ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=s.google_api_key, temperature=0.2, max_retries=0))
        if s.mistral_api_key:
            fallbacks.append(ChatMistralAI(api_key=s.mistral_api_key, model="mistral-small-latest", temperature=0.2, max_retries=0))
        if s.openrouter_api_key:
            fallbacks.append(ChatOpenAI(api_key=s.openrouter_api_key, base_url="https://openrouter.ai/api/v1", model="google/gemma-4-31b-it:free", temperature=0.2, max_retries=0))

        _small_llm = primary_llm.with_fallbacks(fallbacks) if fallbacks else primary_llm
    return _small_llm

async def _acall_llm(system: str, user_content: str, use_small: bool = False) -> str:
    llm = _get_small_llm() if use_small else _get_llm()
    resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user_content)])
    return resp.content.strip()

async def _acall_llm_structured(system: str, user_content: str, schema: type, use_small: bool = False) -> any:
    llm = _get_small_llm() if use_small else _get_llm()
    structured_llm = llm.with_structured_output(schema)
    return await structured_llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user_content)])

# ── Conversation utilities ────────────────────────────────────────────────────

def _conversation_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""

def _extract_previous_recommendations(messages: list[dict]) -> list[dict]:
    """Fallback: if the API payload lacks previous_recommendations, extract them from the assistant's last message URLs."""
    for m in reversed(messages):
        if m["role"] == "assistant":
            # Extract standard SHL catalog URLs from markdown/text (optional trailing slash)
            urls = re.findall(r"https://www.shl.com/products/product-catalog/view/[a-z0-9-]+/?", m["content"])
            if urls:
                catalog = get_catalog()
                recs = []
                for url in urls:
                    # Normalize URL to always have a trailing slash for matching
                    url = url if url.endswith('/') else url + '/'
                    for item in catalog:
                        if item["url"] == url:
                            # Avoid duplicates
                            if not any(r["url"] == url for r in recs):
                                recs.append({
                                    "name": item["name"],
                                    "url": item["url"],
                                    "test_type": item.get("test_type", "")
                                })
                            break
                if recs:
                    return recs
    return []


# ── Extraction + Logic ────────────────────────────────────────────────────────


def _confidence_score(ctx: dict) -> tuple[int, list[str]]:
    """
    Returns (score, missing_fields):
      score 0 → no role at all        → must clarify
      score 1 → role only             → clarify to get seniority
      score 2 → role + 1 signal       → recommend (use defaults for rest)
      score 3 → role + 2+ signals     → recommend confidently

    If the LLM explicitly flags ready_to_recommend (because user is evasive or we have enough), force score=3.
    """
    if ctx["jd_provided"] or ctx["ready_to_recommend"]:
        return 3, []

    missing = []
    score = 0

    if not ctx["job_role"]:
        missing.append("job_role")
        return 0, missing          # can't do anything without a role

    score = 1  # has role

    if ctx["seniority"]:
        score += 1
    else:
        missing.append("seniority")

    if ctx["skills"] or ctx["test_type_hints"] or ctx["industry"]:
        score += 1
    else:
        missing.append("skills or test type preference")

    return score, missing


def _build_search_query(ctx: dict) -> str:
    """
    Build a tight, structured search query from extracted context fields.
    Much more reliable than free-text summarisation.
    """
    parts = []

    if ctx["job_role"]:
        parts.append(ctx["job_role"])
    if ctx["seniority"]:
        parts.append(ctx["seniority"])
    if ctx["skills"]:
        parts.extend(ctx["skills"][:5])          # cap to avoid noise
    if ctx["test_type_hints"]:
        parts.extend(ctx["test_type_hints"])
    if ctx["industry"]:
        parts.append(ctx["industry"])
    if ctx["purpose"]:
        parts.append(ctx["purpose"])

    return " ".join(parts) if parts else "SHL assessment"


def _context_summary(ctx: dict) -> str:
    """Human-readable context string injected into the rerank prompt."""
    lines = []
    if ctx["job_role"]:      lines.append(f"Role: {ctx['job_role']}")
    if ctx["seniority"]:     lines.append(f"Seniority: {ctx['seniority']}")
    if ctx["skills"]:        lines.append(f"Skills: {', '.join(ctx['skills'])}")
    if ctx["test_type_hints"]: lines.append(f"Preferred test types: {', '.join(ctx['test_type_hints'])}")
    if ctx["industry"]:      lines.append(f"Industry: {ctx['industry']}")
    if ctx["purpose"]:       lines.append(f"Purpose: {ctx['purpose']}")
    if ctx.get("jd_provided"): lines.append("Note: User provided a full job description.")
    if ctx["explicit_removes"]: lines.append(f"Remove: {', '.join(ctx['explicit_removes'])}")
    if ctx["explicit_adds"]:    lines.append(f"Must include: {', '.join(ctx['explicit_adds'])}")
    return "\n".join(lines)


# ── Catalog context formatter ─────────────────────────────────────────────────

def _catalog_context(items: list[dict], max_items: int = 20) -> str:
    lines = []
    for item in items[:max_items]:
        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['url']}\n"
            f"  Type: {item['test_type']}\n"
            f"  Levels: {', '.join(item.get('job_levels', [])) or 'All'}\n"
            f"  Duration: {item.get('duration', 'N/A')}\n"
            f"  Description: {item.get('description', '')[:200]}"
        )
    return "\n\n".join(lines)


# ── Node 1: Supervisor (Unified Guard + Intent) ───────────────────────────────

async def supervisor_node(state: AgentState) -> AgentState:
    """
    One node to rule them all: safety check, context extraction, and intent routing.
    Saves one LLM round-trip per turn.
    """
    turn_count = sum(1 for m in state["messages"] if m["role"] == "user")
    state["turn_count"] = turn_count

    # 1. Fast-track heuristics (no LLM needed)
    if turn_count >= 6:
        state["intent"] = "force_recommend"
        # We still need context for retrieval, so we continue to LLM call below
    
    last_msg = _last_user_message(state["messages"]).lower()
    compare_signals = ["difference between", "compare", "vs ", "versus", "which is better", "what is the difference"]
    is_compare = any(s in last_msg for s in compare_signals)

    # 2. LLM Analysis (Safety + Context)
    conversation = _conversation_text(state["messages"])
    prompt = ANALYZE_PROMPT.format(conversation=conversation)
    
    analysis: AnalyzeOutput = await _acall_llm_structured(
        "You are a precise hiring context analyzer.", 
        prompt, 
        AnalyzeOutput
    )
    
    # Hydrate HyDE if ready
    ctx = analysis.model_dump()
    if (ctx["ready_to_recommend"] or ctx["jd_provided"]) and not ctx["hyde_description"]:
        context_sum = _context_summary(ctx)
        ctx["hyde_description"] = await _acall_llm(
            "You are an SHL product expert.", 
            HYDE_PROMPT.format(context_summary=context_sum)
        )
    
    state["conversation_context"] = ctx
    verdict = ctx["verdict"].upper()

    # 3. Routing Logic
    if "BLOCKED" in verdict:
        state["intent"] = "refuse"
    elif "EOC" in verdict:
        prev = state.get("previous_recommendations") or []
        if prev:
            state["reply"] = "Great — your assessment shortlist is confirmed. Good luck with the hiring process!"
            state["recommendations"] = prev
            state["end_of_conversation"] = True
            state["intent"] = "eoc"
        else:
            # If user says "thanks/done" but we have no recs, treat as ALLOWED/pending
            state["intent"] = "pending"
    else:
        # ALLOWED path
        if is_compare:
            state["intent"] = "compare"
        elif state["intent"] == "force_recommend":
            pass # already set
        else:
            score, missing = _confidence_score(ctx)
            if score < 2:
                state["intent"] = "clarify"
                state["missing_fields"] = missing
            else:
                state["intent"] = "recommend"
    
    return state


# ── Node 3: Clarify ───────────────────────────────────────────────────────────

async def clarify_node(state: AgentState) -> AgentState:
    """Ask exactly one targeted question based on what's missing."""
    missing = state.get("missing_fields", ["job_role"])
    prompt = CLARIFY_PROMPT.format(missing_fields=", ".join(missing))
    question = await _acall_llm(SYSTEM_PROMPT + "\n\n" + prompt,
                                _conversation_text(state["messages"]),
                                use_small=True)
    state["reply"] = question
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state


# ── Node 4: Retrieve ──────────────────────────────────────────────────────────

async def retrieve_node(state: AgentState) -> AgentState:
    """
    Build structured query and HyDE description → Hybrid search with RRF (Parallelized).
    """
    import asyncio
    from app.retriever.bm25_retriever import query_bm25
    from app.retriever.fusion import reciprocal_rank_fusion
    from app.agent.prompts import HYDE_PROMPT
    from app.retriever.embedder import aembed_query

    ctx = state.get("conversation_context")
    if not ctx:
        # Fallback to ensure we have context even if supervisor is skipped (unlikely)
        from app.agent.nodes import supervisor_node
        state = await supervisor_node(state)
        ctx = state["conversation_context"]

    # Build structured query
    query_str = _build_search_query(ctx)
    
    # 1. HyDE: Use pre-generated description from context if available
    hyde_desc = ctx.get("hyde_description")
    if not hyde_desc:
        context_sum = _context_summary(ctx)
        try:
            hyde_desc = await _acall_llm("You are an SHL product expert.", 
                                         HYDE_PROMPT.format(context_summary=context_sum))
        except Exception as e:
            print(f"DEBUG: HyDE generation failed, falling back to query string: {e}")
            hyde_desc = query_str

    # 2. Embedding for both query and HyDE
    q_vec_task = aembed_query(query_str)
    hyde_vec_task = aembed_query(hyde_desc)
    
    q_vector, hyde_vector = await asyncio.gather(q_vec_task, hyde_vec_task)
    
    # 3. Metadata filter
    pinecone_filter = None
    if ctx["seniority"] and ctx["seniority"] in SENIORITY_TO_JOB_LEVEL:
        job_levels = SENIORITY_TO_JOB_LEVEL[ctx["seniority"]]
        pinecone_filter = {"job_levels": {"$in": job_levels}}

    # 3. Multimodal retrieval (Parallel)
    loop = asyncio.get_running_loop()
    
    p_q_task = loop.run_in_executor(None, query_catalog, q_vector, 20, pinecone_filter)
    p_h_task = loop.run_in_executor(None, query_catalog, hyde_vector, 20, pinecone_filter)
    bm25_task = loop.run_in_executor(None, query_bm25, query_str, 20)
    
    pinecone_q, pinecone_hyde, bm25_results = await asyncio.gather(p_q_task, p_h_task, bm25_task)
    
    # 4. Fuse with RRF
    fused = reciprocal_rank_fusion(pinecone_q, pinecone_hyde, bm25_results, top_n=30)
    
    # 5. Fallback: If fused results are thin (<8) or missing seniority matches, 
    # retry without filter to avoid missing good general tests
    if len(fused) < 8 and pinecone_filter:
        p_no_filter = await loop.run_in_executor(None, query_catalog, hyde_vector, 20)
        fused = reciprocal_rank_fusion(fused, p_no_filter, top_n=30)

    # 6. Inject personality anchor (OPQ32r) if missing
    opq_keywords = {"opq32r", "opq", "occupational personality"}
    has_opq = any(
        any(kw in item["name"].lower() for kw in opq_keywords)
        for item in fused
    )
    if not has_opq:
        opq_vec = await aembed_query("OPQ32r occupational personality questionnaire behavior")
        opq_results = await loop.run_in_executor(None, query_catalog, opq_vec, 3)
        fused = opq_results + fused

    # 7. Inject any explicit_adds from context
    if ctx["explicit_adds"]:
        full_catalog = get_catalog()
        for add_name in ctx["explicit_adds"]:
            for item in full_catalog:
                if add_name.lower() in item["name"].lower():
                    if not any(item["url"] == f["url"] for f in fused):
                        fused.insert(0, item)
                    break

    # Final dedup by URL
    seen = set()
    final_unique = []
    for item in fused:
        url = item["url"]
        if url not in seen:
            seen.add(url)
            final_unique.append(item)

    state["retrieved_items"] = final_unique[:25]
    return state


# ── Node 5: Recommend (LLM reranker) ─────────────────────────────────────────

async def recommend_node(state: AgentState) -> AgentState:
    """
    LLM reranks top-25 candidates using ConversationContext as scoring rubric.
    """
    ctx = state.get("conversation_context") or {}
    context_summary = _context_summary(ctx)
    catalog_ctx = _catalog_context(state["retrieved_items"])

    prompt = RERANK_PROMPT.format(
        context_summary=context_summary,
        catalog_context=catalog_ctx,
    )

    parsed: RerankOutput = await _acall_llm_structured(SYSTEM_PROMPT, prompt, RerankOutput)
    
    try:
        urls = parsed.recommendation_urls
        
        catalog = get_catalog()
        recs = []
        for url in urls:
            url = url if url.endswith('/') else url + '/'
            for item in catalog:
                if item["url"] == url:
                    if not any(r["url"] == url for r in recs):
                        recs.append({
                            "name": item["name"],
                            "url": item["url"],
                            "test_type": item.get("test_type", "")
                        })
                    break
                    
        state["reply"] = parsed.reply
        state["recommendations"] = recs[:10]
        state["end_of_conversation"] = parsed.end_of_conversation

    except Exception as e:
        print(f"ERROR in recommend_node: {e}")
        state["reply"] = "I had trouble formatting the recommendations. Could you rephrase?"
        state["recommendations"] = []
        state["end_of_conversation"] = False

    return state


# ── Node 6: Compare ───────────────────────────────────────────────────────────

async def compare_node(state: AgentState) -> AgentState:
    last = _last_user_message(state["messages"])
    full_catalog = get_catalog()

    mentioned = [
        item for item in full_catalog
        if item["name"].lower() in last.lower()
    ]
    if not mentioned:
        mentioned = state.get("retrieved_items", [])[:5]

    catalog_ctx = _catalog_context(mentioned)
    conversation = _conversation_text(state["messages"])
    prompt = COMPARE_PROMPT.format(catalog_context=catalog_ctx, conversation=conversation)
    
    parsed: SimpleOutput = await _acall_llm_structured(SYSTEM_PROMPT, prompt, SimpleOutput)

    state["reply"] = parsed.reply
    state["recommendations"] = []
    state["end_of_conversation"] = False

    return state


# ── Node 7: Refuse ────────────────────────────────────────────────────────────

async def refuse_node(state: AgentState) -> AgentState:
    parsed: SimpleOutput = await _acall_llm_structured(SYSTEM_PROMPT, REFUSE_PROMPT, SimpleOutput, use_small=True)
    state["reply"] = parsed.reply
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state
