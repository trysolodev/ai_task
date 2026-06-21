"""Persistence and fact-evolution logic.

Writes turns and memories, and — crucially — handles *supersession*: when a
new memory shares a canonical ``key`` with an existing active one in the same
scope but carries a different value, the old row is marked inactive and linked
into a chain (``supersedes`` / ``superseded_by``) rather than deleted. Recall
reads only ``active`` rows; ``/users/{id}/memories`` exposes the full chain.

Scope rules (documented in the README):
* If a turn has a ``user_id``, its memories are **user-scoped** and shared across
  that user's sessions (the smoke test depends on this).
* If ``user_id`` is null, memories are **session-scoped** and never bleed.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg.types.json import Jsonb

from . import db
from .embeddings import to_vector_literal
from .extraction import SUPERSEDING_TYPES, ExtractedMemory

logger = logging.getLogger("memory.store")

_MAX_CONTENT_CHARS = 200_000
_EMBED_CHARS = 8_000

# Extra search terms folded into each memory's FTS document + embedding so that
# natural-language queries ("where do they live?") match topic-keyed facts
# ("location.city: Berlin"). Matched by exact key first, then by key prefix.
_KEY_ALIASES: Dict[str, str] = {
    "identity.name": "name called named",
    "employment.company": "work job employer company works employed workplace where work",
    "employment.role": "job title role position occupation profession",
    "employment.status": "job employment status work",
    "location.city": "live lives living location city home based reside residence where live",
    "location.origin": "from origin hometown originally moved before",
    "location.country": "country live location",
    "education.school": "school university college study studied education degree",
    "family.spouse": "wife husband spouse married partner family",
    "family.partner": "partner family relationship significant other",
    "family.child": "son daughter child kid children family",
    "family.sibling": "brother sister sibling family",
    "pet.name": "pet animal name dog cat",
    "pet.type": "pet animal kind species type",
    "diet": "diet eat food eating dietary",
    "diet.avoid": "diet avoid eat food restriction",
    "allergy": "allergy allergic reaction intolerant",
    "health.condition": "health condition medical illness",
    "preference.communication_style": "prefer style communication answers responses tone",
}
_PREFIX_ALIASES: Dict[str, str] = {
    "employment": "work job career employer",
    "location": "live location where home",
    "family": "family relative",
    "pet": "pet animal",
    "diet": "diet food eat",
    "preference": "prefer preference like",
    "opinion": "opinion think feel view stance about",
    "skill": "skill know experienced proficient",
    "goal": "goal plan working toward want",
    "event": "event happened recently doing",
    "education": "education school study",
    "health": "health medical",
}


def _aliases_for(key: str, subject: Optional[str]) -> str:
    terms = [key.replace(".", " ").replace("_", " ")]
    if key in _KEY_ALIASES:
        terms.append(_KEY_ALIASES[key])
    prefix = key.split(".", 1)[0]
    if prefix in _PREFIX_ALIASES:
        terms.append(_PREFIX_ALIASES[prefix])
    if subject:
        terms.append(subject)
    return " ".join(terms)


def parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def build_turn_text(messages: Sequence[Tuple[str, str, Optional[str]]]) -> str:
    lines = []
    for role, content, name in messages:
        if not content:
            continue
        label = role
        if name:
            label = f"{role}:{name}"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)[:_MAX_CONTENT_CHARS]


class MemoryStore:
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    # --- writes ---------------------------------------------------------- #
    def store_turn(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        messages_json: List[Dict[str, Any]],
        content: str,
        metadata: Dict[str, Any],
        ts: datetime,
    ) -> str:
        turn_id = _new_id("turn")
        embedding = self._safe_embed(content[:_EMBED_CHARS])
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO turns (id, session_id, user_id, content, messages, metadata, ts, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        turn_id,
                        session_id,
                        user_id,
                        content,
                        Jsonb(messages_json),
                        Jsonb(metadata),
                        ts,
                        embedding,
                    ),
                )
        return turn_id

    def known_active_facts(
        self, user_id: Optional[str], session_id: Optional[str], limit: int = 40
    ) -> List[Dict[str, str]]:
        where, params = self._scope_where(user_id, session_id)
        sql = f"""
            SELECT key, value FROM memories
            WHERE active AND type IN ('fact','preference') AND {where}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT %s
        """
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*params, limit))
                return [{"key": r["key"], "value": r["value"]} for r in cur.fetchall()]

    def apply_extracted(
        self,
        extracted: List[ExtractedMemory],
        *,
        user_id: Optional[str],
        session_id: str,
        turn_id: str,
        ts: datetime,
    ) -> List[str]:
        """Persist extracted memories with supersession. Returns affected ids."""
        affected: List[str] = []
        with db.connection() as conn:
            with conn.transaction():
                for mem in extracted:
                    if not mem.value:
                        continue
                    affected.append(self._apply_one(conn, mem, user_id, session_id, turn_id, ts))
        return affected

    def _apply_one(self, conn, mem: ExtractedMemory, user_id, session_id, turn_id, ts) -> str:
        where, params = self._scope_where(user_id, session_id)
        with conn.cursor() as cur:
            if mem.type in SUPERSEDING_TYPES:
                cur.execute(
                    f"SELECT id, value FROM memories WHERE active AND key=%s AND {where} "
                    f"ORDER BY updated_at DESC LIMIT 1",
                    (mem.key, *params),
                )
                existing = cur.fetchone()
                if existing and existing["value"].strip().lower() == mem.value.strip().lower():
                    cur.execute(
                        "UPDATE memories SET updated_at=%s, confidence=GREATEST(confidence,%s), "
                        "source_turn=%s, source_session=%s WHERE id=%s",
                        (ts, mem.confidence, turn_id, session_id, existing["id"]),
                    )
                    return existing["id"]
                new_id = self._insert(conn, mem, user_id, session_id, turn_id, ts,
                                      supersedes=existing["id"] if existing else None)
                if existing:
                    cur.execute(
                        "UPDATE memories SET active=FALSE, superseded_by=%s, updated_at=%s WHERE id=%s",
                        (new_id, ts, existing["id"]),
                    )
                return new_id
            # event: accumulate, dedupe identical active value
            cur.execute(
                f"SELECT id FROM memories WHERE active AND key=%s AND lower(value)=lower(%s) AND {where} LIMIT 1",
                (mem.key, mem.value, *params),
            )
            dup = cur.fetchone()
            if dup:
                cur.execute("UPDATE memories SET updated_at=%s WHERE id=%s", (ts, dup["id"]))
                return dup["id"]
            return self._insert(conn, mem, user_id, session_id, turn_id, ts, supersedes=None)

    def _insert(self, conn, mem: ExtractedMemory, user_id, session_id, turn_id, ts, supersedes) -> str:
        mem_id = _new_id("mem")
        aliases = _aliases_for(mem.key, mem.subject)
        search_text = f"{mem.key.replace('.', ' ')} {mem.value} {mem.subject or ''} {aliases}"
        embedding = self._safe_embed(search_text)
        attrs = dict(mem.attributes or {})
        if mem.correction:
            attrs["correction"] = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories
                    (id, user_id, session_id, type, key, value, subject, aliases, confidence,
                     attributes, source_session, source_turn, created_at, updated_at,
                     supersedes, active, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s::vector)
                """,
                (
                    mem_id, user_id, session_id, mem.type, mem.key, mem.value, mem.subject,
                    aliases, mem.confidence, Jsonb(attrs),
                    session_id, turn_id, ts, ts, supersedes, embedding,
                ),
            )
        return mem_id

    # --- reads ----------------------------------------------------------- #
    def list_memories(self, user_id: str) -> List[Dict[str, Any]]:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, type, key, value, subject, confidence, attributes,
                           source_session, source_turn, created_at, updated_at,
                           supersedes, superseded_by, active
                    FROM memories WHERE user_id = %s
                    ORDER BY key, active DESC, updated_at DESC
                    """,
                    (user_id,),
                )
                return [self._memory_row(r) for r in cur.fetchall()]

    @staticmethod
    def _memory_row(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": r["id"],
            "type": r["type"],
            "key": r["key"],
            "value": r["value"],
            "confidence": round(float(r["confidence"]), 3),
            "source_session": r["source_session"],
            "source_turn": r["source_turn"],
            "created_at": _iso(r["created_at"]),
            "updated_at": _iso(r["updated_at"]),
            "supersedes": r["supersedes"],
            "superseded_by": r["superseded_by"],
            "active": r["active"],
            "subject": r.get("subject"),
            "attributes": r.get("attributes") or {},
        }

    # --- deletes --------------------------------------------------------- #
    def delete_session(self, session_id: str) -> None:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM turns WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM memories WHERE session_id = %s", (session_id,))

    def delete_user(self, user_id: str) -> None:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM turns WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))

    # --- helpers --------------------------------------------------------- #
    @staticmethod
    def _scope_where(user_id: Optional[str], session_id: Optional[str]) -> Tuple[str, tuple]:
        if user_id:
            return "user_id = %s", (user_id,)
        return "user_id IS NULL AND session_id = %s", (session_id,)

    def _safe_embed(self, text: str):
        try:
            return to_vector_literal(self.embedder.embed(text))
        except Exception as exc:  # pragma: no cover - never let embedding fail a write
            logger.warning("embedding failed; storing NULL embedding: %s", exc)
            return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
