"""Lightweight, dependency-free token budgeting.

The contract only asks that ``/recall`` respect ``max_tokens`` approximately
("don't blow past it by 2x"). We use a conservative chars-per-token heuristic
rather than pulling in a tokenizer: it never makes a network call, is stable
across models, and over-estimates slightly so we stay *under* budget.
"""
from __future__ import annotations

# English averages ~4 chars/token. We divide by 3.6 to bias toward
# over-counting, which keeps assembled context comfortably within budget.
_CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def fits(text: str, budget_tokens: int) -> bool:
    return estimate_tokens(text) <= budget_tokens


def truncate_to_tokens(text: str, budget_tokens: int) -> str:
    """Hard-truncate ``text`` so its estimate fits ``budget_tokens``."""
    if budget_tokens <= 0:
        return ""
    if estimate_tokens(text) <= budget_tokens:
        return text
    max_chars = int(budget_tokens * _CHARS_PER_TOKEN)
    if max_chars <= 1:
        return ""
    return text[: max_chars - 1].rstrip() + "…"
