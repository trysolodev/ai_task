"""Postgres connection pool, pgvector wiring, and schema migrations.

One backing store does three jobs: relational rows (turns, memories, the
supersession chain), vector similarity (pgvector ``<=>``), and full-text
search (``tsvector`` + GIN). Keeping them in one ACID store is what gives us
the contract's *synchronous correctness* guarantee — after ``/turns`` commits,
every read path sees the data.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger("memory.db")

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.row_factory = dict_row


def _connect_with_retry(conninfo: str, attempts: int = 30, delay: float = 1.0) -> psycopg.Connection:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return psycopg.connect(conninfo, autocommit=True)
        except Exception as exc:  # pragma: no cover - depends on DB startup timing
            last = exc
            logger.warning("DB not ready (attempt %d/%d): %s", i + 1, attempts, exc)
            time.sleep(delay)
    assert last is not None
    raise last


def _schema_sql(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS turns (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            user_id       TEXT,
            content       TEXT NOT NULL,
            messages      JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            embedding     VECTOR({dim}),
            tsv           TSVECTOR GENERATED ALWAYS AS
                          (to_tsvector('english', coalesce(content, ''))) STORED
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            id             TEXT PRIMARY KEY,
            user_id        TEXT,
            session_id     TEXT,
            type           TEXT NOT NULL,
            key            TEXT NOT NULL,
            value          TEXT NOT NULL,
            subject        TEXT,
            aliases        TEXT NOT NULL DEFAULT '',
            confidence     REAL NOT NULL DEFAULT 0.6,
            attributes     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            source_session TEXT,
            source_turn    TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            supersedes     TEXT,
            superseded_by  TEXT,
            active         BOOLEAN NOT NULL DEFAULT TRUE,
            embedding      VECTOR({dim}),
            tsv            TSVECTOR GENERATED ALWAYS AS
                           (to_tsvector('english',
                              coalesce(key, '') || ' ' ||
                              coalesce(value, '') || ' ' ||
                              coalesce(subject, '') || ' ' ||
                              coalesce(aliases, ''))) STORED
        )
        """,
        "CREATE INDEX IF NOT EXISTS turns_session_idx ON turns (session_id)",
        "CREATE INDEX IF NOT EXISTS turns_user_idx ON turns (user_id)",
        "CREATE INDEX IF NOT EXISTS turns_ts_idx ON turns (ts DESC)",
        "CREATE INDEX IF NOT EXISTS turns_tsv_idx ON turns USING GIN (tsv)",
        "CREATE INDEX IF NOT EXISTS mem_user_idx ON memories (user_id)",
        "CREATE INDEX IF NOT EXISTS mem_session_idx ON memories (session_id)",
        # The supersession lookup: active memory for a (scope, key).
        "CREATE INDEX IF NOT EXISTS mem_user_key_active_idx ON memories (user_id, key) WHERE active",
        "CREATE INDEX IF NOT EXISTS mem_session_key_active_idx ON memories (session_id, key) WHERE active",
        "CREATE INDEX IF NOT EXISTS mem_tsv_idx ON memories USING GIN (tsv)",
        "CREATE INDEX IF NOT EXISTS mem_subject_idx ON memories (lower(subject))",
    ]


def _vector_indexes(dim: int) -> list[str]:
    # HNSW gives sub-linear ANN search if the store grows large; harmless on
    # small data. Created best-effort so a missing operator class never blocks
    # startup.
    return [
        "CREATE INDEX IF NOT EXISTS turns_embedding_idx ON turns "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS mem_embedding_idx ON memories "
        "USING hnsw (embedding vector_cosine_ops)",
    ]


def init_db(database_url: str, dim: int) -> None:
    """Create extension, tables and indexes, then open the pooled connections."""
    global _pool
    conn = _connect_with_retry(database_url)
    try:
        with conn.cursor() as cur:
            for stmt in _schema_sql(dim):
                cur.execute(stmt)
            for stmt in _vector_indexes(dim):
                try:
                    cur.execute(stmt)
                except Exception as exc:  # pragma: no cover
                    logger.warning("skipping vector index (%s)", exc)
    finally:
        conn.close()

    _pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=True,
        timeout=10.0,
    )
    _pool.wait(timeout=10.0)
    logger.info("database ready (vector dim=%d)", dim)


def close_db() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    if _pool is None:
        raise RuntimeError("database pool not initialized")
    with _pool.connection() as conn:
        yield conn


def healthy() -> bool:
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:  # pragma: no cover
        return False
