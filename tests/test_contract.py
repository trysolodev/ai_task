"""Contract tests (§7): roundtrip + shapes + status codes, concurrent-session
isolation, and malformed-input resilience. Run against a live service."""
from __future__ import annotations

import concurrent.futures

import pytest


def _turn(session, user, content, ts="2025-03-15T10:30:00Z"):
    return {
        "session_id": session,
        "user_id": user,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Noted."},
        ],
        "timestamp": ts,
        "metadata": {},
    }


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_turn_recall_roundtrip_and_shapes(client, run_id):
    user = f"{run_id}_u"
    sess = f"{run_id}_s"
    try:
        r = client.post("/turns", json=_turn(sess, user, "I just moved to Berlin from NYC last month."))
        assert r.status_code == 201
        body = r.json()
        assert isinstance(body.get("id"), str) and body["id"]

        # Recall is queryable immediately (no eventual consistency).
        r = client.post("/recall", json={
            "query": "Where does this user live?",
            "session_id": "other-session", "user_id": user, "max_tokens": 512,
        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["context"], str)
        assert isinstance(data["citations"], list)
        assert "berlin" in data["context"].lower()
        for c in data["citations"]:
            assert set(c.keys()) >= {"turn_id", "score", "snippet"}

        # Structured memories exist (not raw message chunks).
        r = client.get(f"/users/{user}/memories")
        assert r.status_code == 200
        mems = r.json()["memories"]
        assert any(m["type"] in ("fact", "preference", "opinion", "event") for m in mems)
        assert any(m["key"].startswith("location") for m in mems)
        for m in mems:
            assert set(m.keys()) >= {"id", "type", "key", "value", "confidence", "active"}
    finally:
        client.delete(f"/users/{user}")
        client.delete(f"/sessions/{sess}")


def test_search_shape(client, run_id):
    user = f"{run_id}_u"
    sess = f"{run_id}_s"
    try:
        client.post("/turns", json=_turn(sess, user, "I love hiking in the Alps."))
        r = client.post("/search", json={"query": "hiking", "user_id": user, "session_id": None, "limit": 5})
        assert r.status_code == 200
        for item in r.json()["results"]:
            assert set(item.keys()) >= {"content", "score", "session_id", "timestamp", "metadata"}
    finally:
        client.delete(f"/users/{user}")


def test_cold_session_returns_empty_not_error(client, run_id):
    r = client.post("/recall", json={
        "query": "anything?", "session_id": f"{run_id}_cold", "user_id": f"{run_id}_nobody", "max_tokens": 256,
    })
    assert r.status_code == 200
    assert r.json() == {"context": "", "citations": []}


def test_concurrent_sessions_do_not_bleed(client, run_id):
    u1, u2 = f"{run_id}_a", f"{run_id}_b"
    try:
        client.post("/turns", json=_turn(f"{run_id}_sa", u1, "I work at Acme Corp."))
        client.post("/turns", json=_turn(f"{run_id}_sb", u2, "I work at Globex."))

        r1 = client.post("/recall", json={"query": "where does the user work?", "user_id": u1,
                                          "session_id": f"{run_id}_sa", "max_tokens": 400}).json()
        r2 = client.post("/recall", json={"query": "where does the user work?", "user_id": u2,
                                          "session_id": f"{run_id}_sb", "max_tokens": 400}).json()
        assert "acme" in r1["context"].lower() and "globex" not in r1["context"].lower()
        assert "globex" in r2["context"].lower() and "acme" not in r2["context"].lower()
    finally:
        client.delete(f"/users/{u1}")
        client.delete(f"/users/{u2}")


def test_parallel_writes_are_safe(client, run_id):
    user = f"{run_id}_par"
    sess = f"{run_id}_par_s"
    facts = [
        "I live in Tokyo.", "I have a cat named Mochi.", "I am allergic to peanuts.",
        "I work at Initech.", "I prefer concise answers.",
    ]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(
                lambda c: client.post("/turns", json=_turn(sess, user, c)).status_code, facts
            ))
        assert all(code == 201 for code in results)
        mems = client.get(f"/users/{user}/memories").json()["memories"]
        keys = {m["key"] for m in mems if m["active"]}
        assert "location.city" in keys and "employment.company" in keys
    finally:
        client.delete(f"/users/{user}")


@pytest.mark.parametrize("payload,raw", [
    ({"_bad": True}, None),                                  # missing session_id
    (None, "{not valid json"),                               # malformed JSON
    ({"session_id": "", "messages": []}, None),              # empty session_id
])
def test_malformed_turn_returns_4xx_not_crash(client, payload, raw):
    if raw is not None:
        r = client.post("/turns", content=raw, headers={"Content-Type": "application/json"})
    else:
        r = client.post("/turns", json=payload)
    assert 400 <= r.status_code < 500
    # Service still alive afterward.
    assert client.get("/health").status_code == 200


def test_unicode_and_emoji_are_handled(client, run_id):
    user = f"{run_id}_uni"
    sess = f"{run_id}_uni_s"
    try:
        content = "I'm Zoë 🌍, I live in München and I ❤️ 日本語. My dog is named Köhler."
        r = client.post("/turns", json=_turn(sess, user, content))
        assert r.status_code == 201
        r = client.post("/recall", json={"query": "where do they live?", "user_id": user,
                                         "session_id": sess, "max_tokens": 400})
        assert r.status_code == 200  # no crash on unicode
    finally:
        client.delete(f"/users/{user}")


def test_empty_messages_turn_is_accepted(client, run_id):
    user = f"{run_id}_empty"
    sess = f"{run_id}_empty_s"
    try:
        r = client.post("/turns", json={"session_id": sess, "user_id": user,
                                        "messages": [], "timestamp": "2025-03-15T10:30:00Z", "metadata": {}})
        assert r.status_code == 201
    finally:
        client.delete(f"/users/{user}")


def test_delete_endpoints_return_204(client, run_id):
    user = f"{run_id}_del"
    sess = f"{run_id}_del_s"
    client.post("/turns", json=_turn(sess, user, "I live in Oslo."))
    assert client.delete(f"/sessions/{sess}").status_code == 204
    assert client.delete(f"/users/{user}").status_code == 204
    # After user delete, memories are gone.
    assert client.get(f"/users/{user}/memories").json()["memories"] == []
