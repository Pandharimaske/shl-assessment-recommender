# SHL Assessment Recommender — Project Plan
**Assignment:** Take-home, AI Intern, SHL Labs  
**Deadline:** 11 May 2026, 6:00 PM  
**Stack:** Python · uv · FastAPI · LangChain · LangGraph · Groq · Pinecone · Docker · Playwright

---

## 1. WHAT WE ARE BUILDING

A conversational AI agent exposed as a stateless REST API.  
A hiring manager describes a role → agent clarifies if vague → retrieves from SHL catalog → returns 1–10 grounded recommendations.

### Hard constraints from the assignment
- `GET /health` → `{"status": "ok"}`
- `POST /chat` → exact schema: `reply`, `recommendations[]`, `end_of_conversation`
- Each recommendation must have: `name`, `url` (from catalog only), `test_type`
- Max 8 turns per conversation
- 30-second timeout per API call
- Stateless: zero server-side session state

---

## 2. FOLDER STRUCTURE

```
shl-assessment-recommender/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /health + /chat endpoints
│   │   └── schemas.py         # Pydantic request/response models (NON-NEGOTIABLE schema)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py           # LangGraph StateGraph — the agent's brain
│   │   ├── nodes.py           # Individual node functions (clarify, retrieve, respond, guard)
│   │   ├── state.py           # TypedDict for graph state
│   │   └── prompts.py         # All system + node-level prompts
│   │
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── pinecone_client.py # Pinecone init, upsert, query
│   │   └── embedder.py        # Embedding logic (cohere/openai-compatible via Groq or HF)
│   │
│   ├── catalog/
│   │   ├── __init__.py
│   │   ├── catalog.json       # Scraped SHL catalog (source of truth)
│   │   └── loader.py          # Load + validate catalog.json
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Settings via pydantic-settings (env vars)
│   │
│   └── main.py                # FastAPI app entry point
│
├── scripts/
│   └── scrape_catalog.py      # Playwright scraper → writes catalog.json
│
├── tests/
│   ├── test_api.py            # Schema compliance + endpoint tests
│   ├── test_agent.py          # Behavior probes (refuse off-topic, clarify on vague, etc.)
│   └── test_retriever.py      # Recall@10 on public traces
│
├── docs/
│   └── approach.md            # 2-page approach document for submission
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             # uv project config + dependencies
├── uv.lock
├── .env.example
└── README.md
```

---

## 3. TECH STACK — EACH TOOL AND WHY

| Tool | Role | Why |
|---|---|---|
| **Python 3.12** | Language | Ecosystem |
| **uv** | Package manager | 10–100x faster than pip, lockfile, venvs |
| **FastAPI** | API framework | Auto OpenAPI docs, async, Pydantic validation |
| **Pydantic v2** | Schema validation | Strict schema enforcement (assignment is schema-strict) |
| **LangGraph** | Agent orchestration | Stateful graph; clear node-by-node reasoning = defensible in interviews |
| **LangChain** | LLM wrappers + tools | ChatGroq client, prompt templates, message formatting |
| **Groq API** | LLM inference | Free tier, ~300 tokens/sec, fits in 30s timeout easily |
| **Pinecone** | Vector DB | Managed, free tier (1 index), fast ANN search |
| **Playwright** | Catalog scraper | SHL catalog is JS-rendered; playwright handles it |
| **Docker** | Containerization | Reproducible deploy; Render/Railway support Docker |
| **pytest + httpx** | Testing | Async endpoint tests + behavior probe assertions |

### Suggested additions (missing from your list)
| Addition | Why it matters |
|---|---|
| **`pydantic-settings`** | Load `.env` cleanly into typed config — prevents config bugs at runtime |
| **`sentence-transformers`** or **Cohere embed** | Need an embedder to push catalog into Pinecone; Groq doesn't do embeddings |
| **`playwright`** | SHL catalog is JavaScript-rendered; requests+BS4 will get empty tables |
| **`httpx`** | Async HTTP client for tests (pairs with FastAPI TestClient) |
| **`pytest-asyncio`** | Async test support |
| **Render or Railway** | Free deployment platform; Render supports Docker, has a free tier with cold-start |

---

## 4. AGENT DESIGN — LANGGRAPH STATE MACHINE

```
START
  │
  ▼
[guard_node]  ──── off-topic / injection ──→ [refuse_node] ──→ END
  │
  ▼ (in-scope)
[intent_node]
  ├── "compare"  ──→ [compare_node]  ──→ [respond_node] ──→ END
  ├── "vague"    ──→ [clarify_node]  ──→ END (ask question)
  └── "ready"    ──→ [retrieve_node] ──→ [rank_node] ──→ [respond_node] ──→ END
```

### Node responsibilities
- **guard_node** — Detect prompt injection, off-topic (legal, general HR advice). Refuse immediately.
- **intent_node** — Classify: vague / ready-to-recommend / compare-request. Uses turn count to force recommend by turn 6.
- **clarify_node** — Ask ONE targeted question (role, seniority, skills needed, remote or supervised).
- **retrieve_node** — Build query from full conversation → Pinecone semantic search → top-20 candidates.
- **rank_node** — LLM re-ranks top-20 → selects 1–10 → justifies each pick.
- **compare_node** — Retrieve catalog data for named assessments → LLM compares grounded in catalog facts.
- **respond_node** — Format final structured JSON response.

---

## 5. RETRIEVAL DESIGN

### Catalog ingestion (one-time, via `scripts/scrape_catalog.py`)
1. Playwright opens `https://www.shl.com/solutions/products/product-catalog/?type=1` (Individual Test Solutions)
2. Paginate through all pages (start=0, 12, 24, … ~384 items)
3. For each product link → fetch product detail page → extract name, description, test_type, duration, remote_testing, job_levels
4. Save to `app/catalog/catalog.json`
5. Embed each item (name + description + job_levels) → upsert to Pinecone

### At query time
1. Extract intent + key facts from conversation history
2. Build a dense query string: `"{role} {seniority} {skills} {test preferences}"`
3. Pinecone ANN search → top 20
4. LLM re-ranks and filters to 1–10 final recommendations

---

## 6. PROMPT DESIGN (KEY PRINCIPLES)

- **System prompt** establishes SHL-only scope, schema format, and catalog-grounded reasoning
- **Context injection** feeds retrieved catalog snippets directly into the prompt
- **Structured output** — prompt asks LLM to reply ONLY with valid JSON matching the schema
- **Turn counter** injected into prompt — forces a recommendation by turn 6 even if info is partial
- **Anti-hallucination** — prompt explicitly says "only return URLs from the catalog I gave you"

---

## 7. BUILD ORDER (WHAT WE CODE FIRST)

```
Day 1 (Scraper + Catalog)
  └── scripts/scrape_catalog.py  → catalog.json  → Pinecone upsert

Day 1 (Core API skeleton)
  └── main.py → routes.py → schemas.py → /health working

Day 2 (Agent)
  └── state.py → nodes.py → graph.py → agent end-to-end working

Day 2 (Retriever)
  └── embedder.py → pinecone_client.py → integrated into retrieve_node

Day 3 (Tests + Eval)
  └── test_api.py (schema) → test_agent.py (behavior probes) → Recall@10 eval

Day 3 (Deploy)
  └── Dockerfile → docker-compose.yml → push to Render/Railway → submit URL

Day 3 (Docs)
  └── docs/approach.md (2 pages)
```

---

## 8. SCORING STRATEGY

| Scoring criterion | How we address it |
|---|---|
| Schema compliance | Pydantic model validates every response before it leaves the API |
| Catalog-only URLs | Catalog loaded at startup; agent retrieves from Pinecone only; URL allowlist check before response |
| Turn cap ≤ 8 | Turn counter in LangGraph state; intent_node forces recommend at turn 6 |
| Recall@10 | Pinecone semantic search + LLM reranking; broad retrieval (top-20) then narrow |
| Refuse off-topic | guard_node is the first node; tested explicitly in test_agent.py |
| No recommend on turn 1 for vague query | intent_node checks query richness before proceeding |
| Honors edits / refinement | Full conversation history passed every turn; state carries accumulated facts |
| Hallucination rate | Catalog context injected in prompt; LLM told to never invent URLs |

---

## 9. DEPLOYMENT PLAN

- **Platform:** Render (free tier, supports Docker)
- **Cold start:** Pinecone client lazy-init; catalog loaded at startup (~2s); `/health` allows 2-min cold start
- **Env vars on Render:** `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`
- **Docker:** Single-stage, Python 3.12-slim, uv for deps

---

## 10. SUBMISSION CHECKLIST

- [ ] `GET /health` → `{"status": "ok"}` — HTTP 200
- [ ] `POST /chat` — exact schema every time
- [ ] URLs in recommendations come exclusively from catalog.json
- [ ] Agent clarifies before recommending on vague queries
- [ ] Agent recommends 1–10 items (not 0, not 11+)
- [ ] Agent handles mid-conversation refinement
- [ ] Agent compares assessments grounded in catalog data
- [ ] Agent refuses off-topic, legal, prompt-injection
- [ ] Max 8 turns honored
- [ ] Deployed and URL accessible
- [ ] docs/approach.md written (≤2 pages)
