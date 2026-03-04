from __future__ import annotations

import re
from typing import Any


_TRUNCATION_MARKER = "\n...[truncated]"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0

    piece_count = len(_TOKEN_RE.findall(cleaned))
    char_estimate = max(1, len(cleaned) // 4)
    return max(piece_count, char_estimate)


def clip_text(text: str, max_chars: int, marker: str = _TRUNCATION_MARKER) -> str:
    cleaned = (text or "").strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned

    if max_chars <= len(marker):
        return cleaned[:max_chars]

    return cleaned[: max_chars - len(marker)].rstrip() + marker


def _trim_to_token_limit(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""

    candidate = (text or "").strip()
    while candidate and estimate_tokens(candidate) > max_tokens:
        overflow = estimate_tokens(candidate) - max_tokens
        step = max(8, overflow * 3)
        candidate = candidate[:-step].rstrip()
    return candidate


def clip_text_to_tokens(
    text: str, max_tokens: int, marker: str = _TRUNCATION_MARKER
) -> str:
    cleaned = (text or "").strip()
    if max_tokens <= 0 or estimate_tokens(cleaned) <= max_tokens:
        return cleaned

    marker_text = marker or ""
    marker_tokens = estimate_tokens(marker_text) if marker_text else 0
    if marker_tokens >= max_tokens:
        marker_text = ""
        marker_tokens = 0

    estimated = estimate_tokens(cleaned)
    target_chars = max(1, int(len(cleaned) * (max_tokens / float(max(estimated, 1)))))
    candidate = cleaned[:target_chars].rstrip()
    candidate = _trim_to_token_limit(
        candidate,
        max_tokens - marker_tokens if marker_tokens > 0 else max_tokens,
    )

    if marker_text and candidate:
        candidate = _trim_to_token_limit(candidate, max_tokens - marker_tokens)
        if candidate:
            return f"{candidate}{marker_text}"

    if marker_text and marker_tokens <= max_tokens:
        return marker_text.strip() or marker_text

    return candidate


def select_context_rows(
    rows: list[dict[str, Any]],
    max_rows: int,
    max_chunk_chars: int,
    max_total_chars: int,
    max_chunk_tokens: int = 0,
    max_total_tokens: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_chars = 0
    used_tokens = 0

    row_limit = max_rows if max_rows > 0 else len(rows)

    for row in rows:
        if len(selected) >= row_limit:
            break

        text = str(row.get("text") or "").strip()
        if not text:
            continue

        if max_chunk_chars > 0:
            text = clip_text(text, max_chunk_chars)

        if max_chunk_tokens > 0:
            text = clip_text_to_tokens(text, max_chunk_tokens)

        if max_total_chars > 0:
            remaining_chars = max_total_chars - used_chars
            if remaining_chars <= 0:
                break
            text = clip_text(text, remaining_chars, marker="")

        if max_total_tokens > 0:
            remaining_tokens = max_total_tokens - used_tokens
            if remaining_tokens <= 0:
                break
            text = clip_text_to_tokens(text, remaining_tokens, marker="")

        if not text:
            continue

        copied = dict(row)
        copied["text"] = text
        selected.append(copied)
        used_chars += len(text)
        used_tokens += estimate_tokens(text)

    return selected


def format_context(rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, row in enumerate(rows, start=1):
        source_ref = row.get("source_ref") or "unknown"
        text = row.get("text") or ""
        blocks.append(f"[Context {idx}] ({source_ref})\\n{text}")
    return "\\n\\n".join(blocks)


def format_history(
    messages: list[dict[str, Any]],
    max_turns: int,
    max_chars: int,
    max_tokens: int = 0,
) -> str:
    chat_messages = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if max_turns > 0:
        selected = chat_messages[-max_turns * 2 :]
    else:
        selected = chat_messages

    lines: list[str] = []
    for msg in selected:
        role = str(msg["role"]).upper()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    history = clip_text("\\n".join(lines), max_chars)
    if max_tokens > 0:
        history = clip_text_to_tokens(history, max_tokens)
    return history


def build_rag_prompt(question: str, context: str, history: str) -> str:
    return (
        "You are a technical assistant for product documentation and team discussion archives.\\n"
        "Use only the retrieved context to answer factual questions.\\n"
        "If the answer is not present in context, say you do not know.\\n\\n"
        "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.\\n\\n"
        f"Conversation history:\\n{history or 'No prior conversation.'}\\n\\n"
        f"Retrieved context:\\n{context or 'No context retrieved.'}\\n\\n"
        f"Question:\\n{question}"
    )
