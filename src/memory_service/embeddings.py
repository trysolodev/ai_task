"""Pluggable text embedder.

Design goal: recall must work with **zero external dependencies** — no model
download, no API key, no network. The default ``LocalEmbedder`` is a
deterministic hashed n-gram vectorizer (the "hashing trick") that captures
lexical *and* sub-lexical (morphological / typo) overlap. It is not a
transformer, so it won't capture deep synonymy — but the recall pipeline
leans on structured, topic-keyed memories and Postgres full-text search for
that, and treats the vector channel as one signal among several (see
``recall.py``).

Operators who want true semantic embeddings can set ``EMBED_PROVIDER=openai``
or ``voyage``; both request ``EMBED_DIM``-dimensional vectors so the schema
stays fixed. The embedder is a *deploy-time* choice — mixing providers over
the same store yields incomparable vectors.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Protocol

import numpy as np

logger = logging.getLogger("memory.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> List[float]: ...


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_ngrams(token: str, n: int = 3) -> List[str]:
    padded = f"#{token}#"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _hash(feature: str, salt: str) -> int:
    h = hashlib.blake2b(f"{salt}:{feature}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class LocalEmbedder:
    """Deterministic hashed bag-of-features vectorizer.

    Each word token and each character trigram is hashed into the vector with
    a signed weight (the signed hashing trick reduces collision bias). Word
    tokens are weighted above sub-word trigrams. Term frequency is dampened
    sub-linearly. The final vector is L2-normalized so dot product == cosine.
    """

    WORD_WEIGHT = 1.0
    NGRAM_WEIGHT = 0.45

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text or "")
        if not tokens:
            return vec.tolist()

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, tf in counts.items():
            weight = (1.0 + math.log(tf)) * self.WORD_WEIGHT
            self._add(vec, tok, weight)
            for gram in _char_ngrams(tok):
                self._add(vec, gram, self.NGRAM_WEIGHT)

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _add(self, vec: "np.ndarray", feature: str, weight: float) -> None:
        idx = _hash(feature, "idx") % self.dim
        sign = 1.0 if (_hash(feature, "sgn") & 1) == 0 else -1.0
        vec[idx] += sign * weight


class _APIEmbedder:
    """Base for HTTP embedding providers with a local fallback per call.

    Falling back to the local embedder on a transient error keeps writes from
    failing, but note: a mix of provider and local vectors is only weakly
    comparable. Providers are best used on a store that was *always* built with
    that provider.
    """

    def __init__(self, dim: int, api_key: str) -> None:
        self.dim = dim
        self._api_key = api_key
        self._fallback = LocalEmbedder(dim)

    def _remote(self, text: str) -> List[float]:  # pragma: no cover - network
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        try:
            vec = self._remote(text or " ")
        except Exception as exc:  # pragma: no cover - network
            logger.warning("embedding provider failed, using local fallback: %s", exc)
            return self._fallback.embed(text)
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr /= norm
        return arr.tolist()


class OpenAIEmbedder(_APIEmbedder):  # pragma: no cover - network
    def _remote(self, text: str) -> List[float]:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": "text-embedding-3-small",
                "input": text,
                "dimensions": self.dim,  # OpenAI supports dimension reduction
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


class VoyageEmbedder(_APIEmbedder):  # pragma: no cover - network
    def _remote(self, text: str) -> List[float]:
        import httpx

        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": "voyage-3", "input": [text], "output_dimension": self.dim},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def to_vector_literal(embedding) -> str | None:
    """Serialize an embedding to pgvector's text form, e.g. ``[0.1,0.2]``.

    Using the text literal + an explicit ``::vector`` cast at the SQL site makes
    inserts and distance queries independent of any client-side type adapter.
    """
    if embedding is None:
        return None
    return "[" + ",".join(f"{float(x):.6g}" for x in embedding) + "]"


def build_embedder(provider: str, dim: int, *, openai_key: str = "", voyage_key: str = "") -> Embedder:
    provider = (provider or "local").lower()
    if provider == "openai" and openai_key:
        logger.info("using OpenAI embeddings (dim=%d)", dim)
        return OpenAIEmbedder(dim, openai_key)
    if provider == "voyage" and voyage_key:
        logger.info("using Voyage embeddings (dim=%d)", dim)
        return VoyageEmbedder(dim, voyage_key)
    if provider in ("openai", "voyage"):
        logger.warning("provider %s requested but no API key; using local embedder", provider)
    logger.info("using local hashed n-gram embedder (dim=%d)", dim)
    return LocalEmbedder(dim)
