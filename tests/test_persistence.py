"""Restart-persistence test (§7): data written before a full
`docker compose down && up` is recallable afterward.

This drives docker from the *host*, so it's opt-in (it can't run inside the
app container). Enable with:

    RUN_DOCKER_TESTS=1 pytest tests/test_persistence.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

import httpx
import pytest

pytestmark = pytest.mark.docker

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("MEMORY_BASE_URL", "http://localhost:8080")


def _enabled() -> bool:
    return os.environ.get("RUN_DOCKER_TESTS") == "1"


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=REPO, check=True)


def _wait_health(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as c:
        while time.time() < deadline:
            try:
                if c.get("/health").status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
    raise TimeoutError("service did not become healthy after restart")


@pytest.mark.skipif(not _enabled(), reason="set RUN_DOCKER_TESTS=1 to run docker restart test")
def test_data_survives_down_up():
    user = "persist_user_xyz"
    sess = "persist_sess_xyz"
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        _wait_health()
        c.post("/turns", json={
            "session_id": sess, "user_id": user,
            "messages": [{"role": "user", "content": "I live in Reykjavik and work at Volcano Labs."}],
            "timestamp": "2025-03-15T10:30:00Z", "metadata": {},
        }).raise_for_status()
        before = c.post("/recall", json={"query": "where does the user live?", "user_id": user,
                                         "session_id": sess, "max_tokens": 300}).json()
        assert "reykjavik" in before["context"].lower()

    # Full recreate — named volume must preserve the data.
    _compose("down")
    _compose("up", "-d")
    _wait_health()

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        after = c.post("/recall", json={"query": "where does the user live?", "user_id": user,
                                        "session_id": sess, "max_tokens": 300}).json()
        assert "reykjavik" in after["context"].lower(), "data lost across restart"
        mems = c.get(f"/users/{user}/memories").json()["memories"]
        assert any(m["key"] == "location.city" for m in mems)
        c.delete(f"/users/{user}")
        c.delete(f"/sessions/{sess}")
