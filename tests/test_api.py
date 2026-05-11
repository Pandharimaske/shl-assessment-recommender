"""
API schema compliance tests.
Tests EVERY hard requirement from the assignment spec.

Run: uv run pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)



def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_returns_ok():
    r = client.get("/health")
    assert r.json() == {"status": "ok"}



def _chat(messages: list[dict]) -> dict:
    r = client.post("/chat", json={"messages": messages})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    return r.json()


def test_response_has_required_fields():
    resp = _chat([{"role": "user", "content": "I need an assessment"}])
    assert "reply" in resp
    assert "recommendations" in resp
    assert "end_of_conversation" in resp


def test_reply_is_string():
    resp = _chat([{"role": "user", "content": "I need an assessment"}])
    assert isinstance(resp["reply"], str)
    assert len(resp["reply"]) > 0


def test_recommendations_is_list():
    resp = _chat([{"role": "user", "content": "I need an assessment"}])
    assert isinstance(resp["recommendations"], list)


def test_end_of_conversation_is_bool():
    resp = _chat([{"role": "user", "content": "I need an assessment"}])
    assert isinstance(resp["end_of_conversation"], bool)


def test_recommendation_schema():
    """Each recommendation must have name, url, test_type."""
    resp = _chat([
        {"role": "user", "content": "I need to assess a mid-level Java developer with stakeholder communication skills"}
    ])
    for rec in resp["recommendations"]:
        assert "name" in rec, "recommendation missing 'name'"
        assert "url" in rec, "recommendation missing 'url'"
        assert "test_type" in rec, "recommendation missing 'test_type'"


def test_recommendations_max_10():
    resp = _chat([
        {"role": "user", "content": "I need to assess a mid-level Java developer with stakeholder communication skills"}
    ])
    assert len(resp["recommendations"]) <= 10


def test_all_urls_are_shl_catalog():
    """All URLs must come from shl.com/products/product-catalog/."""
    resp = _chat([
        {"role": "user", "content": "Hiring a Python data scientist, senior level, needs ML and statistics knowledge"}
    ])
    for rec in resp["recommendations"]:
        assert "shl.com/products/product-catalog/view/" in rec["url"], (
            f"URL not from catalog: {rec['url']}"
        )


def test_empty_messages_rejected():
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 422


def test_non_user_last_message_rejected():
    r = client.post("/chat", json={
        "messages": [{"role": "assistant", "content": "Hello"}]
    })
    assert r.status_code == 422
