"""Shared pytest fixtures.

The HTTP-level tests run against a *live* service (the eval workflow does
`docker compose up` first). Inside the app container, `MEMORY_BASE_URL`
defaults to localhost:8080 — so `docker compose exec app pytest` works out of
the box. If the service isn't reachable, those tests skip with a clear reason.
"""
from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

BASE_URL = os.environ.get("MEMORY_BASE_URL", "http://localhost:8080")
AUTH_TOKEN = os.environ.get("MEMORY_AUTH_TOKEN", "")


def _wait_for_health(client: httpx.Client, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200:
                return
            last = f"status {r.status_code}"
        except Exception as exc:  # connection refused while booting
            last = repr(exc)
        time.sleep(1.0)
    pytest.skip(f"memory-service not healthy at {BASE_URL} ({last})")


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=60.0) as c:
        _wait_for_health(c)
        yield c


@pytest.fixture
def run_id() -> str:
    """A unique suffix so test data never collides across runs or with eval data."""
    return "t" + uuid.uuid4().hex[:10]
