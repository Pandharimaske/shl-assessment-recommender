# SHL Assessment Recommender — Approach Document
**Submitted by:** Pandhari Maske | B.Tech CSE (AI), VIIT Pune  
**Assignment:** Build a Conversational SHL Assessment Recommender  
**Stack:** Python 3.12 · uv · FastAPI · LangChain · LangGraph · Groq (LLaMA 3.3 70B) · Pinecone · sentence-transformers · Docker

---

## 1. System Design

### Architecture Overview
The system is a stateless REST API built on FastAPI. Every `POST /chat` call receives the full conversation history and returns a structured response — the server stores no session state. This matches the assignment spec exactly and makes the service trivially horizontally scalable.

The core intelligence is a **LangGraph StateGraph** with seven nodes:

```
guard → intent → clarify       (ask one question, return)
               → retrieve → recommend  (return shortlist)
               → compare               (grounded comparison, return)
               → refuse                (decline + continue)
```

**Guard node** runs first on every call. It classifies the last user message as `ALLOWED` or `BLOCKED` using the LLM. Blocked messages return a polite refusal without ending the conversation — the user can still ask about assessments.

**Intent node** classifies the agent's next action: `clarify`, `recommend`, `compare`, or `force_recommend`. A hard rule forces `force_recommend` at turn 6 regardless of context — this ensures the 8-turn cap from the assignment is always honored.

**Retrieve node** builds a condensed search query from the full conversation (via LLM), embeds it with `sentence-transformers/all-MiniLM-L6-v2`, and queries Pinecone for the top 20 candidates. OPQ32r (the personality anchor) is injected separately if not already in results.

**Recommend node** receives the top-20 candidates and asks the LLM to select 1–10, returning strict JSON. A URL allowlist guardrail runs after the LLM — any URL not in the official catalog is stripped before the response leaves the server.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Groq + LLaMA 3.3 70B | ~300 tok/sec — well within 30s timeout even on complex conversations |
| Pinecone (managed vector DB) | Free tier, no infra to manage, sub-100ms ANN search |
| sentence-transformers (local embed) | Free, fast, no API cost; all-MiniLM-L6-v2 gives good semantic recall |
| LangGraph over raw LangChain | Explicit node-by-node flow = defensible in code review; easy to add nodes |
| Stateless API | Matches spec; simplifies deployment; no session cleanup needed |
| URL allowlist post-LLM | Hard guarantee: hallucinated URLs never reach the evaluator |

---

## 2. Retrieval Setup

**Catalog ingestion (one-time):** The official SHL catalog JSON (provided via URL) contains ~380 individual test solutions. Each item is converted to a rich text blob: `name | description | job_levels | category`. These are embedded and upserted into Pinecone with full metadata (name, URL, test_type, job_levels, duration).

**At query time:** The full conversation is summarised into a 30-word search query by the LLM, then embedded and searched against Pinecone (top-20, cosine similarity). The LLM then re-ranks and filters to 1–10 final picks — this two-stage approach gives high recall (broad retrieval) with high precision (LLM re-ranking).

**test_type mapping:** The catalog `keys` field (e.g., `["Knowledge & Skills", "Simulations"]`) is mapped to letter codes (`K,S`) at load time.

---

## 3. Prompt Design

All prompts live in `app/agent/prompts.py`. Key principles:

- **System prompt** establishes scope (SHL-only), URL grounding rule, and OPQ32r default behaviour.
- **Structured output** — every node that produces a response asks the LLM for valid JSON only. A regex-based JSON extractor handles occasional markdown fences.
- **Turn counter injection** — the intent node receives the current turn count so it can enforce the 6-turn recommendation deadline.
- **Anti-hallucination** — the recommend prompt explicitly says "only return URLs from the catalog excerpts I gave you." Combined with the URL allowlist check, this gives two layers of protection.
- **One question per clarify turn** — the clarify prompt explicitly instructs the LLM to pick the single most important missing piece.

---

## 4. What Didn't Work and How It Was Fixed

**Problem:** LLM sometimes wrapped JSON in markdown code fences (` ```json `), breaking `json.loads`.  
**Fix:** `_parse_json_response()` strips fences and falls back to regex extraction of the first `{...}` block.

**Problem:** OPQ32r rarely surfaced in semantic search results despite being the expected personality anchor.  
**Fix:** Retrieve node performs a secondary Pinecone query specifically for OPQ32r and appends results before deduplication.

**Problem:** Agent occasionally recommended on turn 1 for slightly-vague queries.  
**Fix:** Intent prompt was tightened with explicit examples; "I need an assessment" maps to `clarify`, "hiring a Java developer" maps to `recommend`.

**Problem:** `end_of_conversation` triggered too early.  
**Fix:** Keyword list (`EOC_KEYWORDS`) checks the last user message for explicit confirmation phrases ("confirmed", "lock it in", "that works", etc.) rather than relying on LLM judgment.

---

## 5. Evaluation Approach

**Hard evals:** `tests/test_api.py` — schema compliance on every response, URL allowlist, 10-item cap, empty list on clarify/refuse.

**Behavior probes:** `tests/test_agent.py` — 10 binary assertions: no-recommend on vague turn 1, clarify with question, recommend on full JD turn 1, refuse off-topic, refuse injection, honor refinement, no hallucinated URLs, force-recommend by turn 6, EOC on confirmation.

**Recall@10:** `tests/test_retriever.py` — simulates each of the 10 public traces with a one-turn prompt and computes Recall@10 against extracted ground-truth shortlists.

**AI Tools Used:** Claude (Anthropic) was used for code scaffolding, prompt iteration, and test case generation. All design decisions, node logic, and prompt wording were reviewed and understood before submission.
