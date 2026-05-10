# SHL Assessment Recommender — Full Project Summary & Path Ahead

---

## WHAT WE ARE BUILDING

A conversational AI agent exposed as a stateless REST API.
A hiring manager describes a role → agent clarifies if vague → retrieves from
SHL catalog → returns 1–10 grounded recommendations.

Deadline: 11 May 2026, 6:00 PM
Submission: Public API URL + 2-page approach doc

---

## ASSIGNMENT HARD CONSTRAINTS (non-negotiable)

- GET /health → {"status": "ok"}
- POST /chat → exact schema: reply (str), recommendations (list), end_of_conversation (bool)
- Each recommendation: name, url (catalog only), test_type
- Max 8 turns per conversation
- 30-second timeout per API call
- Stateless: zero server-side session state
- recommendations=[] when clarifying or refusing
- end_of_conversation=true only on user confirmation

---

## TECH STACK (DECIDED)

| Tool               | Role                                      |
|--------------------|-------------------------------------------|
| Python 3.12        | Language                                  |
| uv                 | Package manager (fast, lockfile)          |
| FastAPI            | API framework                             |
| Pydantic v2        | Schema validation                         |
| LangGraph          | Agent state machine (node-by-node flow)   |
| LangChain          | LLM wrappers, prompt templates            |
| Groq API           | LLM inference (fast, free tier)           |
| sentence-transformers | Local embeddings (all-MiniLM-L6-v2)   |
| Pinecone           | Vector DB (managed, free tier)            |
| Docker             | Containerization                          |
| Render             | Deployment platform                       |
| pytest + httpx     | Testing                                   |

---

## CATALOG (SOLVED - NO SCRAPER NEEDED)

Official catalog JSON provided directly by SHL at:
https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json

~380 individual test solutions already saved at:
app/catalog/shl_product_catalog.json

Each item has: entity_id, name, link, description, keys, job_levels,
languages, duration, remote, adaptive

keys → test_type mapping:
  Ability & Aptitude           → A
  Biodata & Situational Judgment → B
  Competencies                 → C
  Development & 360            → D
  Assessment Exercises         → E
  Knowledge & Skills           → K
  Personality & Behavior       → P
  Simulations                  → S

---

## KEY DESIGN DECISIONS MADE

### 1. Stateless API — how it works
  Every POST /chat call receives the FULL conversation history from the client.
  Server reads it, processes it, returns response, forgets everything.
  No Redis, no DB sessions, no in-memory dicts.
  The conversation array IS the memory.

### 2. Structured Context Extraction (CORE INSIGHT)
  Before any decision, extract structured schema from full conversation:

  ConversationContext = {
    job_role:         "Java developer" | null,
    seniority:        "entry|graduate|mid|senior|manager|director|executive" | null,
    skills:           ["Spring", "REST APIs", "AWS"],
    test_type_hints:  ["technical", "personality", "cognitive"],
    purpose:          "hiring|development" | null,
    jd_provided:      bool,
    explicit_removes: ["OPQ32r"],
    explicit_adds:    ["AWS test"],
    languages:        ["Spanish"],
    industry:         "healthcare" | null
  }

  This drives EVERYTHING: confidence scoring, Pinecone query,
  metadata filter, rerank rubric. Nothing depends on free-text anymore.

### 3. Deterministic Confidence Scoring (no LLM coin flip)
  score = 0 → no job_role found         → ALWAYS clarify
  score = 1 → role only                 → clarify (ask seniority)
  score = 2 → role + 1 signal           → recommend (enough)
  score = 3 → role + 2+ signals         → recommend confidently
  jd_provided = True                    → shortcut to score 3

### 4. Two-Stage Retrieval + LLM Reranking
  Stage 1: Build structured query from context fields → embed →
           Pinecone ANN top-25 (cosine similarity)
           Optional: metadata filter by job_level for precision
           Fallback: retry without filter if <5 results

  Stage 2: LLM reranker scores top-25 on:
           - Role match
           - Level match (seniority alignment)
           - Type match (user's test_type_hints)
           - Coverage (cognitive + personality + skills)
           - Explicit removes honored
           - OPQ32r gap-fill (always inject if absent)

### 5. OPQ32r Default Personality Anchor
  OPQ32r appears in 8/10 sample conversations.
  Always included unless user explicitly removes it.
  Injected via secondary Pinecone query if not in top-25.

### 6. URL Guardrail (two layers of anti-hallucination)
  Layer 1: Prompt tells LLM "only use URLs from catalog excerpts I gave you"
  Layer 2: _validate_recommendations() strips any URL not in get_valid_url_set()
  Both layers run before response leaves the server.

### 7. end_of_conversation Logic
  Keyword list (EOC_KEYWORDS) checks last user message for confirmation phrases:
  "confirmed", "perfect", "that works", "locking it in", etc.
  Should be checked BEFORE intent classification (currently a gap — see below).

---

## WHAT EACH FILE DOES

app/
  main.py              FastAPI app + lifespan startup (loads catalog + compiles graph)
  core/config.py       Typed settings from .env via pydantic-settings
  api/
    schemas.py         Pydantic request/response models (exact spec schema)
    routes.py          GET /health + POST /chat
  catalog/
    loader.py          Load JSON, map keys→test_type, URL allowlist set
    shl_product_catalog.json  Official SHL catalog (380 items)
  retriever/
    embedder.py        sentence-transformers wrapper + build_catalog_text()
    pinecone_client.py Pinecone upsert + query with metadata
  agent/
    state.py           LangGraph TypedDict state
    prompts.py         All prompts: system, guard, context_extract,
                       clarify, rerank, compare, refuse + EOC_KEYWORDS
    nodes.py           7 node functions: guard, intent, clarify,
                       retrieve, recommend, compare, refuse
    graph.py           StateGraph wiring + run_agent() entry point

scripts/
  ingest_catalog.py    One-time: embed catalog → upsert to Pinecone

tests/
  test_api.py          Schema compliance tests (10 tests)
  test_agent.py        Behavior probe tests (10 tests)
  test_retriever.py    Recall@10 eval against public traces

docs/
  approach.md          2-page submission document

Dockerfile + docker-compose.yml + pyproject.toml + .env.example

---

## SAMPLE CONVERSATIONS ANALYSIS (10 traces studied)

Critical findings:
1. URL base is /products/product-catalog/view/ (NOT /solutions/products/)
2. recommendations=[] not null when clarifying (use spec, not sample format)
3. end_of_conversation=true only on explicit user confirmation
4. Recommend turn 1 if full JD or enough context given
5. OPQ32r is default personality anchor in 8/10 traces
6. test_type can be multi-value: "K,S"
7. Refinement is surgical (add/remove items, never restart)
8. Compare returns [] recommendations, answers in reply only
9. Refuse + continue (never close conversation on refusal)
10. Agent admits catalog gaps honestly and suggests nearest alternative

---

## COMPLETE AGENT DECISION TREE

Every POST /chat call:

GUARD CHECK
  → injection/legal/off-topic/competitor  → REFUSE (continue conv)
  → SHL-related                           → continue

[MISSING GAP] EOC CHECK — should run here before intent
  → user confirmed                        → return current shortlist + true
  → not confirmed                         → continue

EXTRACT ConversationContext (structured schema from full history)

INTENT CLASSIFICATION
  → compare signals in last message       → COMPARE
  → confidence=0 (no role)               → CLARIFY (ask role)
  → confidence=1 (role only)             → CLARIFY (ask seniority)
  → confidence≥2                         → RETRIEVE → RERANK → RECOMMEND
  → turn≥6 (hard deadline)               → force RETRIEVE → RERANK → RECOMMEND

CLARIFY path
  missing=job_role    → "What role are you hiring for?"
  missing=seniority   → "What seniority/experience level?"
  missing=skills      → "Any specific skills or domains to test?"

RETRIEVE path
  Build query from ConversationContext fields (structured, not free text)
  Pinecone top-25 with optional job_level metadata filter
  Fallback: remove filter if <5 results
  Inject OPQ32r if not present
  Inject explicit_adds from context
  Dedup by URL

RERANK path (LLM scores top-25)
  Criteria: role match, level match, type match, coverage, removes, gap-fill
  Trim to 1-10
  URL guardrail strips hallucinated URLs
  Catalog gap notice if skill has no test

COMPARE path
  Extract named assessments from user message
  Fetch their catalog data
  LLM compares grounded in descriptions only
  Returns [] recommendations

---

## KNOWN GAP TO FIX

EOC (end_of_conversation) check currently runs INSIDE recommend_node.
Should be a dedicated check BEFORE intent_node so we skip re-retrieval:

Current (wrong):
  guard → intent → retrieve → recommend → [EOC check here]

Correct:
  guard → EOC_check → [confirmed] → return existing shortlist + true
                    → [not confirmed] → intent → ...

Impact: Without this fix, a confirmation message triggers a full
re-retrieve + rerank unnecessarily and may change the shortlist.

---

## PATH AHEAD — IN ORDER

### STEP 1: Fix EOC gap (30 min)
  Add eoc_node to graph.py
  Add route_after_guard to check EOC first
  Add "previous_recommendations" to AgentState to return existing shortlist
  Update graph wiring

### STEP 2: Setup local environment (15 min)
  uv venv && uv sync
  cp .env.example .env
  Fill in: GROQ_API_KEY, PINECONE_API_KEY, GROQ_MODEL

### STEP 3: Ingest catalog into Pinecone (10 min, run once)
  uv run python scripts/ingest_catalog.py
  Verify: ~380 vectors in Pinecone index

### STEP 4: Run server locally and smoke test (15 min)
  uv run uvicorn app.main:app --reload --port 8000
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"I need an assessment"}]}'

### STEP 5: Run all tests (30 min)
  uv run pytest tests/test_api.py -v
  uv run pytest tests/test_agent.py -v
  uv run python tests/test_retriever.py
  Fix any failures before deploying

### STEP 6: Deploy to Render (20 min)
  Push to GitHub
  New Web Service on Render → connect repo → Docker runtime
  Set env vars: GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, GROQ_MODEL
  Deploy → wait for /health to return 200
  Note the public URL (e.g. https://shl-recommender.onrender.com)

### STEP 7: Final submission checks (15 min)
  Test live URL: GET /health → {"status":"ok"}
  Test live URL: POST /chat with sample conversations
  Verify all URLs in responses are from shl.com/products/product-catalog/view/
  Verify schema compliance on every response

### STEP 8: Submit (5 min)
  Fill submission form with:
    - Public API endpoint URL
    - docs/approach.md (already written)

---

## SCORING STRATEGY

Hard evals (must pass):
  Schema compliance          → Pydantic validates every response before it leaves
  Catalog-only URLs          → URL allowlist check after every LLM response
  Turn cap ≤8                → turn_count tracked in state, force_recommend at 6

Recall@10:
  Two-stage retrieval + LLM reranker maximises relevant items in top-10
  OPQ32r always injected (appears in 8/10 ground truth traces)
  Structured query from context (not noisy free text) improves precision

Behavior probes:
  No recommend on vague turn 1   → confidence scoring (score 0/1 → clarify)
  Refuse off-topic               → guard_node first, always
  Honor refinement               → explicit_adds/removes in context schema
  No hallucinated URLs           → two-layer guardrail
  EOC on confirmation            → EOC_KEYWORDS check (fix gap first)

---

## TOTAL TIME ESTIMATE TO SUBMISSION

  Fix EOC gap          30 min
  Environment setup    15 min
  Ingest catalog       10 min
  Smoke test           15 min
  Run all tests        30 min
  Fix test failures    30 min (buffer)
  Deploy to Render     20 min
  Final checks         15 min
  Submit               5 min
  ─────────────────────────
  Total:               ~3 hours
