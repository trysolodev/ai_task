"""Recall and search: turn a query into ranked context for the next agent turn.

``/recall`` is the primary signal. End to end:

1. **Retrieve** candidates from two channels — pgvector cosine and Postgres
   full-text (``ts_rank_cd``) — over active memories and over turns, then fuse
   the two rank lists with **Reciprocal Rank Fusion** (RRF). Hybrid beats
   vanilla cosine-top-k: keyword-heavy probes ("what's the dog's name?") lean
   on FTS, paraphrases lean on vectors.
2. **Understand** the query with a cheap intent classifier that maps it to
   canonical key families (e.g. "where do they live" -> ``location.*``) and
   reorders facts so the relevant ones survive a tight budget.
3. **Assemble** under ``max_tokens`` with explicit priority tiers:
   stable facts -> preferences -> query-relevant memories -> relevant
   conversation. Tiers 3-4 are relevance-gated so off-topic queries get empty
   context instead of hallucinated memories.

Supersession is respected everywhere: only ``active`` rows are retrieved, and a
fact rendered from the chain shows "(updated DATE; previously VALUE)".
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import db, tokens
from .embeddings import to_vector_literal

logger = logging.getLogger("memory.recall")

# Order facts are surfaced in when budget is tight (most stable/identifying first).
_FACT_PRIORITY = [
    "identity.name", "employment.company", "employment.role", "employment.status",
    "location.city", "location.origin", "location.country",
    "family.spouse", "family.partner", "family.child", "family.sibling",
    "pet.name", "pet.type", "diet", "diet.avoid", "allergy", "health.condition",
]

# Query intent -> key prefixes to prioritize. Heuristic, runs with no LLM.
_INTENT_PATTERNS: List[Tuple[re.Pattern, Tuple[str, ...]]] = [
    (re.compile(r"\b(live|living|located|location|city|where.*(live|based)|address|reside)\b", re.I), ("location",)),
    (re.compile(r"\b(work|works|job|employer|company|career|profession|occupation|role|title)\b", re.I), ("employment",)),
    (re.compile(r"\b(pet|dog|cat|animal)\b", re.I), ("pet",)),
    (re.compile(r"\b(eat|food|diet|vegetarian|vegan|allerg|dietary)\b", re.I), ("diet", "allergy")),
    (re.compile(r"\b(wife|husband|spouse|partner|married|kid|kids|son|daughter|child|children|family)\b", re.I), ("family",)),
    (re.compile(r"\b(name|called)\b", re.I), ("identity",)),
    (re.compile(r"\b(school|university|college|study|studied|degree|education)\b", re.I), ("education",)),
    (re.compile(r"\b(prefer|preference|style|like.*answer|concise)\b", re.I), ("preference",)),
]

# Capitalized interrogatives / filler that should never be treated as entities
# during multi-hop user resolution.
_QUERY_STOPWORDS = {
    "what", "where", "who", "when", "why", "how", "which", "tell", "the", "this",
    "that", "their", "them", "they", "does", "did", "do", "is", "are", "user",
    "person", "people", "city", "dog", "cat", "pet", "name", "named", "called",
    "a", "an", "i", "me", "my", "about", "and", "of", "for",
}

_FACT_PHRASING = {
    "identity.name": "Name: {v}",
    "employment.company": "Works at {v}",
    "employment.role": "Role: {v}",
    "employment.status": "Employment: {v}",
    "location.city": "Lives in {v}",
    "location.origin": "Originally from {v}",
    "location.country": "Country: {v}",
    "family.spouse": "Spouse: {v}",
    "family.partner": "Partner: {v}",
    "family.child": "Child: {v}",
    "family.sibling": "Sibling: {v}",
    "allergy": "Allergic to {v}",
    "diet": "Diet: {v}",
    "diet.avoid": "Avoids eating {v}",
    "health.condition": "Health: {v}",
}


def _intent_prefixes(query: str) -> List[str]:
    hits: List[str] = []
    for pat, prefixes in _INTENT_PATTERNS:
        if pat.search(query):
            hits.extend(prefixes)
    return hits


def _rrf(*ranked_lists: List[str], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ids in ranked_lists:
        for rank, _id in enumerate(ids):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _date(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


class _Budget:
    """Accumulate lines while staying within a token budget."""

    def __init__(self, max_tokens: int) -> None:
        self.max = max_tokens
        self.lines: List[str] = []
        self._used = 0

    def remaining(self) -> int:
        return self.max - self._used

    def add(self, line: str) -> bool:
        cost = tokens.estimate_tokens(line) + 1  # newline
        if self._used + cost > self.max:
            return False
        self.lines.append(line)
        self._used += cost
        return True

    def text(self) -> str:
        return "\n".join(self.lines).strip()


class RecallEngine:
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    # ------------------------------------------------------------------ #
    # /recall
    # ------------------------------------------------------------------ #
    def recall(
        self, query: str, user_id: Optional[str], session_id: Optional[str], max_tokens: int
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        # Resolve the owning user for multi-hop when only an entity is given.
        if not user_id and query:
            user_id = self._resolve_user(query, session_id) or user_id

        facts = self._active_by_type(user_id, session_id, ("fact",))
        prefs = self._active_by_type(user_id, session_id, ("preference",))
        if not facts and not prefs and not query:
            return {"context": "", "citations": []}

        intent = _intent_prefixes(query)
        facts = self._order_facts(facts, intent)

        rel_memories = self._relevant_memories(query, user_id, session_id) if query else []
        rel_turns = self._relevant_turns(query, user_id, session_id) if query else []

        budget = _Budget(max(1, int(max_tokens * 0.95)))
        citations: List[Dict[str, Any]] = []
        self._render(budget, citations, facts, prefs, rel_memories, rel_turns)

        return {"context": budget.text(), "citations": citations}

    def _render(self, budget, citations, facts, prefs, rel_memories, rel_turns) -> None:
        seen_pet = any(f["key"] == "pet.name" for f in facts)

        # Tier 1 — stable facts
        if facts:
            if budget.add("## Known facts about this user"):
                for f in facts:
                    if f["key"] == "pet.type" and seen_pet:
                        continue
                    line = "- " + self._fact_line(f)
                    if not budget.add(line):
                        break
                    self._cite(citations, f, 1.0)

        # Tier 2 — preferences
        if prefs and budget.remaining() > 8:
            if budget.add("\n## Preferences"):
                for p in prefs:
                    line = "- " + self._pref_line(p)
                    if not budget.add(line):
                        break
                    self._cite(citations, p, 0.9)

        # Tier 3/4 — query-relevant memories + conversation
        episodic = [m for m in rel_memories if m["type"] in ("opinion", "event", "skill", "goal")]
        if (episodic or rel_turns) and budget.remaining() > 8:
            if budget.add("\n## Relevant from recent conversations"):
                for m in episodic:
                    line = "- " + self._episodic_line(m)
                    if not budget.add(line):
                        return
                    self._cite(citations, m, float(m.get("score", 0.5)))
                for t in rel_turns:
                    snippet = self._turn_snippet(t["content"])
                    line = f"- [{_date(t['ts'])}] {snippet}"
                    if not budget.add(line):
                        return
                    citations.append(
                        {"turn_id": t["id"], "score": round(float(t.get("score", 0.5)), 4),
                         "snippet": snippet}
                    )

    # ------------------------------------------------------------------ #
    # /search  (structured results, not prose)
    # ------------------------------------------------------------------ #
    def search(
        self, query: str, session_id: Optional[str], user_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        mem = self._relevant_memories(query, user_id, session_id, limit=limit, scoped=bool(user_id or session_id))
        turns = self._relevant_turns(query, user_id, session_id, limit=limit)
        results: List[Dict[str, Any]] = []
        for m in mem:
            results.append({
                "content": f"{m['key']}: {m['value']}",
                "score": round(float(m.get("score", 0.0)), 4),
                "session_id": m.get("session_id"),
                "timestamp": _iso(m.get("updated_at")),
                "metadata": {"type": m["type"], "kind": "memory", "key": m["key"],
                             "active": m["active"], "confidence": round(float(m["confidence"]), 3)},
            })
        for t in turns:
            results.append({
                "content": t["content"][:1000],
                "score": round(float(t.get("score", 0.0)), 4),
                "session_id": t.get("session_id"),
                "timestamp": _iso(t.get("ts")),
                "metadata": {"kind": "turn"},
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------ #
    # Retrieval primitives
    # ------------------------------------------------------------------ #
    def _active_by_type(self, user_id, session_id, types: Tuple[str, ...]) -> List[Dict[str, Any]]:
        where, params = _mem_scope(user_id, session_id)
        type_ph = ",".join(["%s"] * len(types))
        # Scalar subquery (not a join) keeps the unqualified scope columns
        # unambiguous and the FROM single-table.
        sql = f"""
            SELECT m.*, (SELECT value FROM memories p WHERE p.id = m.supersedes) AS prev_value
            FROM memories m
            WHERE m.active AND m.type IN ({type_ph}) AND {where}
            ORDER BY m.confidence DESC, m.updated_at DESC
        """
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*types, *params))
                return cur.fetchall()

    def _relevant_memories(
        self, query, user_id, session_id, limit: int = 8, scoped: bool = True
    ) -> List[Dict[str, Any]]:
        q_emb = self._safe_embed(query)
        where, params = _mem_scope(user_id, session_id) if scoped else ("TRUE", ())
        with db.connection() as conn:
            vec_ids = self._vector_ids(conn, "memories", q_emb, where, params, active=True)
            fts_ids = self._fts_ids(conn, "memories", query, where, params, active=True)
            fused = _rrf(vec_ids, fts_ids)
            if not fused:
                return []
            ids = list(fused.keys())
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT m.*, prev.value AS prev_value FROM memories m "
                    "LEFT JOIN memories prev ON prev.id = m.supersedes WHERE m.id = ANY(%s)",
                    (ids,),
                )
                rows = {r["id"]: r for r in cur.fetchall()}
        ranked = []
        for _id, score in fused.items():
            row = rows.get(_id)
            if not row:
                continue
            row["score"] = score + 0.02 * float(row["confidence"])
            ranked.append(row)
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:limit]

    def _relevant_turns(self, query, user_id, session_id, limit: int = 4) -> List[Dict[str, Any]]:
        q_emb = self._safe_embed(query)
        where, params = _turn_scope(user_id, session_id)
        with db.connection() as conn:
            vec_ids = self._vector_ids(conn, "turns", q_emb, where, params, active=False)
            fts_ids = self._fts_ids(conn, "turns", query, where, params, active=False)
            fused = _rrf(vec_ids, fts_ids)
            if not fused:
                return []
            ids = list(fused.keys())
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM turns WHERE id = ANY(%s)", (ids,))
                rows = {r["id"]: r for r in cur.fetchall()}
        ranked = []
        for _id, score in fused.items():
            row = rows.get(_id)
            if not row:
                continue
            row["score"] = score
            ranked.append(row)
        ranked.sort(key=lambda r: (r["score"], r["ts"]), reverse=True)
        return ranked[:limit]

    def _vector_ids(self, conn, table, q_emb, where, params, active: bool) -> List[str]:
        if q_emb is None:
            return []
        active_clause = "active AND " if active else ""
        sql = (
            f"SELECT id FROM {table} WHERE {active_clause}embedding IS NOT NULL AND {where} "
            f"ORDER BY embedding <=> %s::vector LIMIT 20"
        )
        with conn.cursor() as cur:
            cur.execute(sql, (*params, q_emb))
            return [r["id"] for r in cur.fetchall()]

    def _fts_ids(self, conn, table, query, where, params, active: bool) -> List[str]:
        active_clause = "active AND " if active else ""
        sql = (
            f"SELECT id, ts_rank_cd(tsv, q) AS rank "
            f"FROM {table}, websearch_to_tsquery('english', %s) q "
            f"WHERE {active_clause}tsv @@ q AND {where} "
            f"ORDER BY rank DESC LIMIT 20"
        )
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (query, *params))
            except Exception:  # malformed tsquery on odd unicode etc.
                return []
            return [r["id"] for r in cur.fetchall()]

    def _resolve_user(self, query: str, session_id: Optional[str]) -> Optional[str]:
        """Multi-hop: find the user a query's entity belongs to (e.g. 'Biscuit').

        Only triggers when ``/recall`` is called with no ``user_id``. Matches
        proper-noun / "named X" entities in the query against memory *values*,
        so an off-topic query with no matching entity resolves to nothing
        (preserving noise resistance) rather than guessing a user.
        """
        candidates = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query))
        candidates.update(re.findall(r"\b(?:named|called)\s+([A-Za-z][\w]+)", query, re.I))
        candidates = [c for c in candidates if c.lower() not in _QUERY_STOPWORDS]
        if not candidates:
            return None
        with db.connection() as conn:
            with conn.cursor() as cur:
                for cand in candidates:  # exact value match first (precise)
                    cur.execute(
                        "SELECT user_id FROM memories WHERE active AND user_id IS NOT NULL "
                        "AND lower(value) = lower(%s) LIMIT 1", (cand,))
                    row = cur.fetchone()
                    if row:
                        return row["user_id"]
                for cand in candidates:  # then substring match
                    cur.execute(
                        "SELECT user_id FROM memories WHERE active AND user_id IS NOT NULL "
                        "AND value ILIKE %s LIMIT 1", (f"%{cand}%",))
                    row = cur.fetchone()
                    if row:
                        return row["user_id"]
        return None

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    def _order_facts(self, facts: List[Dict[str, Any]], intent: List[str]) -> List[Dict[str, Any]]:
        def sort_key(f):
            key = f["key"]
            prefix = key.split(".", 1)[0]
            intent_rank = 0 if (intent and prefix in intent) else 1
            try:
                prio = _FACT_PRIORITY.index(key)
            except ValueError:
                prio = len(_FACT_PRIORITY)
            return (intent_rank, prio, -float(f["confidence"]))

        return sorted(facts, key=sort_key)

    def _fact_line(self, f: Dict[str, Any]) -> str:
        key, value, subject = f["key"], f["value"], f.get("subject")
        if key == "pet.name":
            base = f"Has a {subject} named {value}" if subject and subject != "pet" else f"Has a pet named {value}"
        elif key == "pet.type":
            base = f"Has a {value}"
        elif key in _FACT_PHRASING:
            base = _FACT_PHRASING[key].format(v=value)
        else:
            base = f"{key.replace('.', ' ').replace('_', ' ').capitalize()}: {value}"
        prev = f.get("prev_value")
        if prev and prev.strip().lower() != value.strip().lower():
            base += f" (updated {_date(f.get('updated_at'))}; previously {prev})"
        return base

    def _pref_line(self, p: Dict[str, Any]) -> str:
        key, value = p["key"], p["value"]
        if key == "preference.communication_style":
            base = f"Prefers {value} answers"
        elif key == "diet":
            base = value.capitalize()
        else:
            base = value if key.startswith("preference") else f"{key.replace('.', ' ')}: {value}"
        prev = p.get("prev_value")
        if prev and prev.strip().lower() != value.strip().lower():
            base += f" (previously {prev})"
        return base

    def _episodic_line(self, m: Dict[str, Any]) -> str:
        value = m["value"]
        date = _date(m.get("updated_at"))
        prefix = f"[{date}] " if date else ""
        if m["type"] == "opinion":
            line = f"{prefix}Opinion on {m.get('subject') or m['key'].split('.', 1)[-1]}: {value}"
            prev = m.get("prev_value")
            if prev and prev.strip().lower() != value.strip().lower():
                line += f" (evolved from: {prev})"
            return line
        return f"{prefix}{value}"

    @staticmethod
    def _turn_snippet(content: str, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", content or "").strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    @staticmethod
    def _cite(citations: List[Dict[str, Any]], mem: Dict[str, Any], score: float) -> None:
        turn_id = mem.get("source_turn")
        if not turn_id:
            return
        citations.append({
            "turn_id": turn_id,
            "score": round(float(mem.get("score", score)), 4),
            "snippet": f"{mem['key']}: {mem['value']}",
        })

    def _safe_embed(self, text: str):
        try:
            return to_vector_literal(self.embedder.embed(text))
        except Exception:  # pragma: no cover
            return None


# scope helpers ----------------------------------------------------------- #
def _mem_scope(user_id: Optional[str], session_id: Optional[str]) -> Tuple[str, tuple]:
    if user_id:
        return "user_id = %s", (user_id,)
    if session_id:
        return "user_id IS NULL AND session_id = %s", (session_id,)
    return "TRUE", ()


def _turn_scope(user_id: Optional[str], session_id: Optional[str]) -> Tuple[str, tuple]:
    # Recent-conversation recall spans a user's sessions (documented sharing);
    # falls back to session scope when there's no user id.
    if user_id:
        return "user_id = %s", (user_id,)
    if session_id:
        return "session_id = %s", (session_id,)
    return "TRUE", ()


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
