"""Pure unit tests — no service, no DB, no network.

Cover the deterministic building blocks: the rule-based extractor, token
budgeting, the local embedder, and the alias map. These run anywhere
(`pytest tests/test_extraction_unit.py`) and gate the logic the LLM path
falls back to.
"""
from __future__ import annotations

import math

from memory_service.embeddings import LocalEmbedder, to_vector_literal
from memory_service.extraction import RuleExtractor
from memory_service import tokens


def _extract(text: str):
    msgs = [("user", text, None)]
    mems = RuleExtractor().extract(msgs, [])
    return {(m.key): m for m in mems}


def test_extracts_employment_company_and_role():
    by_key = _extract("I work at Stripe as a backend engineer.")
    assert by_key["employment.company"].value == "Stripe"
    assert "engineer" in by_key["employment.role"].value


def test_company_capture_does_not_swallow_trailing_words():
    by_key = _extract("I work at Notion as a product manager.")
    assert by_key["employment.company"].value == "Notion"


def test_started_at_company():
    by_key = _extract("I just started at Notion as a product manager.")
    assert by_key["employment.company"].value == "Notion"
    assert by_key["employment.company"].attributes.get("event") == "started"


def test_location_move_with_origin():
    by_key = _extract("I moved to Berlin from San Francisco last month.")
    assert by_key["location.city"].value == "Berlin"
    assert by_key["location.origin"].value == "San Francisco"


def test_multiword_location():
    by_key = _extract("I live in San Francisco.")
    assert by_key["location.city"].value == "San Francisco"


def test_pet_explicit():
    by_key = _extract("I have a dog named Biscuit.")
    assert by_key["pet.name"].value == "Biscuit"
    assert by_key["pet.name"].subject == "dog"


def test_pet_implicit_low_confidence():
    by_key = _extract("Walking Biscuit this morning was lovely.")
    assert "pet.name" in by_key
    assert by_key["pet.name"].confidence < 0.6  # implicit => lower confidence


def test_diet_and_allergy():
    by_key = _extract("I'm vegetarian and I'm allergic to shellfish.")
    assert by_key["diet"].value == "vegetarian"
    assert by_key["allergy"].value == "shellfish"


def test_name_extraction():
    by_key = _extract("Hi, I'm Alice.")
    assert by_key["identity.name"].value == "Alice"


def test_name_not_confused_with_diet():
    by_key = _extract("I am vegetarian.")
    assert "identity.name" not in by_key


def test_opinion_with_stance():
    by_key = _extract("I love TypeScript.")
    assert any(k.startswith("opinion") for k in by_key)


def test_correction_flag():
    msgs = [("user", "Actually, I work at Notion now.", None)]
    mems = RuleExtractor().extract(msgs, [])
    assert any(m.correction for m in mems)


def test_no_false_positives_on_smalltalk():
    mems = RuleExtractor().extract([("user", "thanks, that helps a lot!", None)], [])
    assert mems == []


# --- token budgeting -------------------------------------------------------
def test_token_estimate_monotonic():
    assert tokens.estimate_tokens("") == 0
    assert tokens.estimate_tokens("hello world") > 0
    assert tokens.estimate_tokens("a" * 400) > tokens.estimate_tokens("a" * 40)


def test_truncate_respects_budget():
    text = "word " * 500
    out = tokens.truncate_to_tokens(text, 20)
    assert tokens.estimate_tokens(out) <= 20


# --- local embedder --------------------------------------------------------
def test_embedder_dim_and_norm():
    emb = LocalEmbedder(dim=128)
    v = emb.embed("the quick brown fox")
    assert len(v) == 128
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-5


def test_embedder_similarity_orders_sensibly():
    emb = LocalEmbedder(dim=256)

    def cos(a, b):
        va, vb = emb.embed(a), emb.embed(b)
        return sum(x * y for x, y in zip(va, vb))

    related = cos("I have a dog named Biscuit", "the user's dog is called Biscuit")
    unrelated = cos("I have a dog named Biscuit", "quarterly financial projections")
    assert related > unrelated


def test_embedder_deterministic():
    a = LocalEmbedder(64).embed("hello there")
    b = LocalEmbedder(64).embed("hello there")
    assert a == b


def test_empty_embedding_is_zero_vector():
    v = LocalEmbedder(32).embed("")
    assert v == [0.0] * 32


def test_vector_literal_roundtrip_format():
    assert to_vector_literal(None) is None
    lit = to_vector_literal([0.1, 0.2, 0.3])
    assert lit.startswith("[") and lit.endswith("]")
