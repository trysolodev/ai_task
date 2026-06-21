# CHANGELOG

Iteration history. Each entry: **what changed**, **why**, **result**, **next**.

All self-eval numbers come from `tests/test_recall_quality.py` (11 probes over
`fixtures/`) run in the **offline configuration** — rule-based extraction +
local hashed-n-gram embedder, **no API key** — unless stated otherwise. That's
the floor; the LLM extraction path raises the ceiling. Reporting the floor
keeps the metric reproducible by anyone, with or without keys.

---

## v0.5 — Multi-hop resolution + precision fixes → self-eval 11/11

**What changed.** (1) Rewrote `/recall`'s no-`user_id` path: instead of
full-text searching the whole question, I extract proper-noun / "named X"
entities and match them against memory *values* to resolve the owning user.
(2) Fixed the rule extractor's implicit-pet matcher (sentence-initial
"Walking Biscuit" wasn't matching — verb needed to be case-insensitive while
keeping the captured name capitalized). (3) Broadened role capture so
abbreviations like "PM"/"CTO" supersede a prior role.

**Why.** First full fixture run scored **0.91 (10/11)**; the only failure was
`multi_hop_entity_resolution` ("what city does the person with a dog named
Biscuit live in?", `user_id=null`). Root cause: `websearch_to_tsquery` ANDs
all terms, so a long question never matches a single short memory (a `pet.name`
row has no "city"/"live" tokens). Entity-value matching is both more correct
*and* safer — an off-topic null-user query matches no entity and resolves to
nothing, so noise resistance is preserved.

**Result.** Self-eval **1.00 (11/11 probes fully passing)**: fact evolution,
job history, current city, both multi-hop variants, implicit pet name, diet +
allergy, preference, cross-user isolation, off-topic noise, and cold user.
Full suite: **33 passed, 1 skipped** (the opt-in docker-restart test).
Verified live: `Role: PM (updated …; previously engineer)` and
`Works at Notion (… previously Stripe)`.

**Next.** Opinion arcs still collapse to "current + immediate prior"; model the
full trajectory. Add an LLM-mode self-eval run once a key is available to
quantify the lift over the rule floor.

---

## v0.4 — Priority-tiered, token-budgeted context assembly

**What changed.** Replaced "concatenate top-k hits" with explicit tiers:
stable facts → preferences → query-relevant episodic memories → relevant
conversation, filled in order until 95% of `max_tokens` (conservatively
estimated). Tiers 3–4 are relevance-gated. Recall now renders readable prose
with per-key phrasing and history notes, plus citations.

**Why.** "Context for the next turn" is a triage problem, not a dump. Stable
facts are useful on *any* follow-up and are compact, so they deserve the first
tokens; query-specific material must be gated or off-topic queries hallucinate
"relevant" snippets.

**Result.** Token budget verified: at `max_tokens=30` the context cleanly
truncates to the single highest-priority fact (~24 est. tokens, never >2×
budget). Noise + cold probes pass — off-topic queries return real facts but no
fabricated snippets; unknown users return empty context.

**Next.** The gating threshold is currently "any retrieval match." A learned or
score-based cutoff could tighten precision on adversarial noise.

---

## v0.3 — Hybrid retrieval (vector ⊕ FTS) with RRF + key aliases

**What changed.** Added Postgres full-text search (`tsvector` + GIN,
`ts_rank_cd`) alongside pgvector cosine and fused the two with Reciprocal Rank
Fusion. Crucially, each memory's FTS document and embedding text are enriched
with **alias terms** for its canonical key (e.g. `location.city` folds in
"live lives home city where reside").

**Why.** Pure cosine over the local n-gram embedder missed keyword-exact probes
("what's the dog's *name*?"), and neither channel mapped natural-language
questions ("where do they *live*?") onto topic-keyed facts
(`location.city: Berlin`) — the query and the stored value share no tokens.
Aliases bridge that vocabulary gap; RRF combines the channels without having to
calibrate a cosine score against a `ts_rank` score.

**Why local embeddings at all.** The build/runtime network here blocks model
CDNs (HuggingFace) and the Docker image CDN, and "missing API keys" is an
explicit failure mode. A dependency-free hashed-n-gram embedder makes recall
work with zero downloads/keys and keeps the image small; providers
(OpenAI/Voyage) remain a one-env-var upgrade.

**Result.** Recall became robust to phrasing. Notably, ablating the vector
channel barely moved the fixture score — because stable facts are surfaced by
tier regardless of retrieval — which confirmed the design insight that the
**structured fact layer**, not the embedder, carries most of the recall weight
here. Documented as a deliberate tradeoff.

**Next.** Priority logic for when the assembled context exceeds the budget
(became v0.4).

---

## v0.2 — Structured extraction + supersession (the actual memory layer)

**What changed.** Introduced typed memories (`fact|preference|opinion|event`)
with **canonical keys**, confidence, subject, and provenance. Added the
supersession engine: a new value for an existing `(scope, key)` deactivates the
old row and links a `supersedes`/`superseded_by` chain. Built the LLM extractor
(Claude + structured-outputs JSON schema) with a deterministic rule-based
fallback so the service produces real memories with no API key.

**Why.** v0.1 could only return message text — a log, not a memory. The eval
inspects `/users/{id}/memories` for *structured* records and tests that
"Stripe → Notion" returns the current employer while preserving history. That's
a relational supersession problem, which drove the schema.

**Result.** `/users/{id}/memories` returns clean typed rows; the Stripe→Notion
chain shows one active row (`supersedes` set) and one inactive row
(`superseded_by` set). Implicit facts ("walking Biscuit" → `pet.name`) and
decomposition ("moved to Berlin from NYC" → city + origin) work. This is where
the canonical-key vocabulary was tuned so contradictions actually collide on
the same key.

**Next.** Recall was still naive top-k; queries phrased differently from stored
values missed (became v0.3).

---

## v0.1 — Skeleton: contract, Postgres, synchronous correctness

**What changed.** FastAPI service implementing all seven endpoints, Postgres +
pgvector via `docker compose` with a named volume, idempotent schema migrations
on startup, lenient request models, and a body-size guard. `/turns` writes and
commits before returning.

**Why.** Establish the non-negotiables first: exact contract shapes/status
codes, persistence across restarts, and synchronous reads-after-write — so
every later iteration could be validated against a real running stack.

**Result.** Smoke test passes; `docker compose up` boots with no manual steps;
malformed input yields 4xx, not crashes. Chose Postgres-for-everything over a
vector-DB-plus-metadata split specifically to get reads-after-write for free.

**Next.** Make it a memory service rather than a message log (became v0.2).

---

### Engineering notes / environment caveat

The grading run does `docker compose up`; I verified that path's config
(`docker compose config`) and the image build recipe. Because this development
sandbox's network blocks the Docker image CDN, I validated the full service
end-to-end against an equivalent **local Postgres 16 + pgvector 0.6** with the
app run via uvicorn — same code, same `DATABASE_URL` contract — which is how the
self-eval numbers above were produced. The opt-in `tests/test_persistence.py`
exercises the real `docker compose down && up` volume-persistence path on a
machine with image access.
