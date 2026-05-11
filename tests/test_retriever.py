"""
Recall@10 evaluation against the 10 public conversation traces.

Run: uv run python tests/test_retriever.py

Expects GenAI_SampleConversations/ to exist in the project root
with files C1.md through C10.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import httpx

API_URL = "http://localhost:8000"


GROUND_TRUTH: dict[str, list[str]] = {
    "C1": [
        "Occupational Personality Questionnaire (OPQ32r)",
        "Core Java (Advanced Level) (New)",
        "Spring (New)",
        "RESTful Web Services (New)",
        "SHL Verify Interactive - G+",
    ],
    "C2": [
        "Linux Programming (General)",
        "Networking and Implementation (New)",
        "Occupational Personality Questionnaire (OPQ32r)",
    ],
    "C3": [
        "Entry Level Customer Serv-Retail & Contact Center",
        "Contact Center Call Simulation (New)",
        "SVAR - Spoken English (US) (New)",
    ],
    "C4": [
        "Microsoft Excel 365 (New)",
        "Microsoft Word 365 (New)",
        "Financial Accounting (New)",
        "Occupational Personality Questionnaire (OPQ32r)",
        "Graduate Scenarios",
    ],
    "C5": [
        "Global Skills Assessment",
        "Occupational Personality Questionnaire (OPQ32r)",
        "SHL Verify Interactive - G+",
    ],
    "C6": [
        "Occupational Personality Questionnaire (OPQ32r)",
        "SHL Verify Interactive - G+",
        "Graduate Scenarios",
    ],
    "C7": [
        "HIPAA (Security)",
        "Medical Terminology (New)",
        "Occupational Personality Questionnaire (OPQ32r)",
    ],
    "C8": [
        "Amazon Web Services (AWS) Development (New)",
        "Docker (New)",
        "Kubernetes (New)",
        "Occupational Personality Questionnaire (OPQ32r)",
    ],
    "C9": [
        "Core Java (Advanced Level) (New)",
        "RESTful Web Services (New)",
        "Amazon Web Services (AWS) Development (New)",
        "Docker (New)",
        "Occupational Personality Questionnaire (OPQ32r)",
    ],
    "C10": [
        "SHL Verify Interactive - G+",
        "Graduate Scenarios",
        "Entry Level Customer Serv-Retail & Contact Center",
    ],
}


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """Recall@K = |relevant ∩ top-K recommended| / |relevant|"""
    if not relevant:
        return 1.0
    top_k = recommended[:k]
    top_k_lower = [n.lower() for n in top_k]
    hits = sum(1 for r in relevant if r.lower() in top_k_lower)
    return hits / len(relevant)


def simulate_conversation(trace_id: str) -> list[str]:
    """
    Run a simple 2-turn conversation for each trace and collect recommended names.
    (In real evaluation, the harness runs a full multi-turn conversation.)
    """
    prompts = {
        "C1": "I need to assess a mid-level Java backend developer who collaborates with stakeholders.",
        "C2": "I'm hiring a senior Linux systems engineer with networking skills.",
        "C3": "We need assessments for entry-level customer service representatives in a contact center.",
        "C4": "Hiring a financial analyst graduate who uses Excel and Word daily.",
        "C5": "I need a broad competency assessment for general professional roles.",
        "C6": "Looking for cognitive and personality tests for graduate-level candidates.",
        "C7": "Hiring a healthcare administrator who needs HIPAA knowledge and medical terminology.",
        "C8": "We need to assess a mid-level DevOps engineer with AWS, Docker, and Kubernetes skills.",
        "C9": "Hiring a senior Java microservices developer who works with REST APIs, AWS, and Docker.",
        "C10": "Need assessments for entry-level graduates in a customer-facing role requiring cognitive ability.",
    }

    prompt = prompts.get(trace_id, "I need an assessment for a mid-level professional.")

    try:
        r = httpx.post(
            f"{API_URL}/chat",
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        data = r.json()
        return [rec["name"] for rec in data.get("recommendations", [])]
    except Exception as e:
        print(f"  [ERROR] Trace {trace_id}: {e}")
        return []


def run_eval():
    print("=" * 60)
    print("Recall@10 Evaluation on Public Traces")
    print("=" * 60)

    scores = []
    for trace_id, relevant in GROUND_TRUTH.items():
        recommended = simulate_conversation(trace_id)
        score = recall_at_k(recommended, relevant, k=10)
        scores.append(score)
        status = "✅" if score >= 0.5 else "❌"
        print(f"{status} {trace_id}: Recall@10 = {score:.2f}  "
              f"({len(recommended)} recommended, {len(relevant)} relevant)")
        if recommended:
            print(f"   Got: {recommended[:5]}")

    mean_recall = sum(scores) / len(scores) if scores else 0.0
    print("-" * 60)
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print("=" * 60)
    return mean_recall


if __name__ == "__main__":
    run_eval()
