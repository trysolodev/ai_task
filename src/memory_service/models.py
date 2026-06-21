"""Pydantic request/response models for the HTTP contract (§3).

Models are deliberately lenient: unknown fields are ignored and scalar
content is coerced to text, so malformed-but-parseable payloads are handled
gracefully rather than crashing. Structurally invalid payloads (bad JSON,
wrong types that can't be coerced) still surface as FastAPI 422s.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        # Some agents send structured content blocks; flatten to text.
        import json

        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""
    name: Optional[str] = None

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, v: Any) -> str:
        return (str(v).strip().lower() or "user") if v is not None else "user"

    @field_validator("content", mode="before")
    @classmethod
    def _content(cls, v: Any) -> str:
        return _coerce_text(v)


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", mode="before")
    @classmethod
    def _session(cls, v: Any) -> str:
        s = ("" if v is None else str(v)).strip()
        if not s:
            raise ValueError("session_id must be a non-empty string")
        return s

    @field_validator("metadata", mode="before")
    @classmethod
    def _metadata(cls, v: Any) -> Dict[str, Any]:
        return v if isinstance(v, dict) else {}


class TurnResponse(BaseModel):
    id: str


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    max_tokens: int = 1024

    @field_validator("query", mode="before")
    @classmethod
    def _query(cls, v: Any) -> str:
        return _coerce_text(v)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _budget(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 1024
        return max(0, min(n, 32000))


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: List[Citation]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10

    @field_validator("query", mode="before")
    @classmethod
    def _query(cls, v: Any) -> str:
        return _coerce_text(v)

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 10
        return max(1, min(n, 100))


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: Optional[str]
    timestamp: Optional[str]
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]


class MemoryOut(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    source_session: Optional[str]
    source_turn: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    supersedes: Optional[str]
    superseded_by: Optional[str]
    active: bool
    subject: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class MemoriesResponse(BaseModel):
    memories: List[MemoryOut]
