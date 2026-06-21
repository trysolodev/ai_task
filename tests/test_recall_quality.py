"""Recall-quality self-eval (§7).

Ingests `fixtures/conversations.json`, runs `fixtures/probes.json` against
`/recall`, and reports "X of Y expected facts surfaced". This is the iteration
loop — run it after every change. Ids are namespaced per run and cleaned up.

The pass threshold is intentionally modest so the suite stays green on the
rule-based extractor (no API key). With LLM extraction it scores higher; the
CHANGELOG tracks the numbers across iterations.
"""
from __future__ import annotations

import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
RECALL_THRESHOLD = 0.80  # fraction of expected facts that must appear in context


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _ns(value, run_id):
    return f"{run_id}_{value}" if value else value


@pytest.fixture
def ingested(client, run_id):
    convo = _load("conversations.json")
    users, sessions = set(), set()
    for turn in convo["turns"]:
        user = _ns(turn.get("user_id"), run_id)
        sess = _ns(turn["session_id"], run_id)
        users.add(user)
        sessions.add(sess)
        payload = {
            "session_id": sess,
            "user_id": user,
            "messages": turn["messages"],
            "timestamp": turn.get("timestamp"),
            "metadata": turn.get("metadata", {}),
        }
        r = client.post("/turns", json=payload)
        assert r.status_code == 201, r.text
    yield run_id
    for u in users:
        client.delete(f"/users/{u}")
    for s in sessions:
        client.delete(f"/sessions/{s}")


def test_recall_quality_report(client, ingested, run_id, capsys):
    probes = _load("probes.json")["probes"]
    total_expected = 0
    found_expected = 0
    probe_pass = 0
    lines = []

    for p in probes:
        user = _ns(p.get("user_id"), run_id) if p.get("user_id") else None
        sess = _ns(p.get("session_id"), run_id)
        resp = client.post("/recall", json={
            "query": p["query"], "user_id": user, "session_id": sess, "max_tokens": 1024,
        })
        assert resp.status_code == 200
        ctx = resp.json()["context"].lower()

        exp = [e.lower() for e in p.get("expect", [])]
        unexp = [u.lower() for u in p.get("unexpected", [])]
        hits = [e for e in exp if e in ctx]
        total_expected += len(exp)
        found_expected += len(hits)

        ok = len(hits) == len(exp)
        bad = [u for u in unexp if u in ctx]
        if bad:
            ok = False
        if p.get("empty"):
            ok = ok and ctx.strip() == ""
        probe_pass += 1 if ok else 0

        status = "PASS" if ok else "FAIL"
        detail = f"{len(hits)}/{len(exp)} expected"
        if bad:
            detail += f", leaked {bad}"
        lines.append(f"  [{status}] {p['name']}: {detail}")

    recall = (found_expected / total_expected) if total_expected else 1.0
    report = (
        "\n=== Recall-quality self-eval ===\n"
        + "\n".join(lines)
        + f"\n  expected-fact recall: {found_expected}/{total_expected} = {recall:.2f}\n"
        + f"  probes fully passing: {probe_pass}/{len(probes)}\n"
    )
    with capsys.disabled():
        print(report)

    # Hard invariants: isolation + noise + cold must never fail.
    critical = {"cross_user_isolation", "noise_resistance_offtopic", "cold_unknown_user"}
    for p in probes:
        if p["name"] in critical:
            user = _ns(p.get("user_id"), run_id) if p.get("user_id") else None
            sess = _ns(p.get("session_id"), run_id)
            ctx = client.post("/recall", json={
                "query": p["query"], "user_id": user, "session_id": sess, "max_tokens": 1024,
            }).json()["context"].lower()
            for u in [x.lower() for x in p.get("unexpected", [])]:
                assert u not in ctx, f"{p['name']} leaked '{u}'"
            if p.get("empty"):
                assert ctx.strip() == "", f"{p['name']} should be empty"

    assert recall >= RECALL_THRESHOLD, f"recall {recall:.2f} < {RECALL_THRESHOLD}\n{report}"
