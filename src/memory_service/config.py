"""Runtime configuration, read once from the environment at import time."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _clean(value: str | None) -> str:
    return (value or "").strip()


@dataclass(frozen=True)
class Settings:
    database_url: str
    auth_token: str            # empty => auth disabled
    extraction_backend: str    # auto | llm | rules
    extraction_model: str
    anthropic_api_key: str
    embed_provider: str        # local | openai | voyage
    embed_dim: int
    openai_api_key: str
    voyage_api_key: str

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_token)


def load_settings() -> Settings:
    return Settings(
        database_url=_clean(os.environ.get("DATABASE_URL"))
        or "postgresql://memory:memory@localhost:5432/memory",
        auth_token=_clean(os.environ.get("MEMORY_AUTH_TOKEN")),
        extraction_backend=_clean(os.environ.get("EXTRACTION_BACKEND")) or "auto",
        extraction_model=_clean(os.environ.get("EXTRACTION_MODEL")) or "claude-opus-4-8",
        anthropic_api_key=_clean(os.environ.get("ANTHROPIC_API_KEY")),
        embed_provider=(_clean(os.environ.get("EMBED_PROVIDER")) or "local").lower(),
        embed_dim=int(_clean(os.environ.get("EMBED_DIM")) or "384"),
        openai_api_key=_clean(os.environ.get("OPENAI_API_KEY")),
        voyage_api_key=_clean(os.environ.get("VOYAGE_API_KEY")),
    )


settings = load_settings()
