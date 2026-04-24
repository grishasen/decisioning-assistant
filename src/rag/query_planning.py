from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryRewriteConfig:
    """Store conversation-aware retrieval query rewrite settings."""

    enabled: bool = False
    max_tokens: int = 96
    temperature: float = 0.0


def build_query_rewrite_prompt(question: str, history: str) -> str:
    """Build the local-model prompt for standalone retrieval query rewriting."""
    return (
        "Rewrite the current user question into a standalone search query for "
        "retrieving product documentation and team discussion archives.\n"
        "Use the conversation history only to resolve references like it, this, "
        "that, they, the previous answer, or the product/module being discussed.\n"
        "Preserve exact product names, acronyms, versions, dates, error text, and "
        "technical terms. Do not answer the question.\n"
        "Return only the rewritten search query, with no labels or explanation.\n\n"
        f"Conversation history:\n{history or 'No prior conversation.'}\n\n"
        f"Current user question:\n{question}\n\n"
        "Standalone retrieval query:"
    )


def _json_query(value: str) -> str:
    """Return a query field from JSON output when the model emits JSON."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("query", "retrieval_query", "standalone_query", "question"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def clean_rewritten_query(output: str, fallback: str) -> str:
    """Normalize model output into a single retrieval query."""
    cleaned = str(output or "").strip()
    fallback_text = str(fallback or "").strip()
    if not cleaned:
        return fallback_text

    json_query = _json_query(cleaned)
    if json_query:
        cleaned = json_query

    cleaned = re.sub(r"^```(?:\w+)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(
        r"^(standalone\s+)?(retrieval\s+)?(search\s+)?query\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return fallback_text

    lowered = cleaned.casefold()
    if lowered in {"same as above", "unchanged", "n/a", "none"}:
        return fallback_text
    return cleaned


def rewrite_retrieval_query(
    generator: Any,
    *,
    question: str,
    history: str,
    config: QueryRewriteConfig,
) -> str:
    """Rewrite the current question for retrieval, falling back to the original."""
    fallback = str(question or "").strip()
    if not config.enabled or not fallback or not str(history or "").strip():
        return fallback

    prompt = build_query_rewrite_prompt(fallback, history)
    try:
        rewritten = generator.generate(
            prompt=prompt,
            max_tokens=max(16, int(config.max_tokens)),
            temperature=max(0.0, float(config.temperature)),
        )
    except Exception:
        return fallback
    return clean_rewritten_query(str(rewritten or ""), fallback)
