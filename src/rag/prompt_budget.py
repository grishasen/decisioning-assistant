from __future__ import annotations

from typing import Any


_TRUNCATION_MARKER = "\n...[truncated]"


def clip_text(text: str, max_chars: int, marker: str = _TRUNCATION_MARKER) -> str:
    cleaned = (text or "").strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned

    if max_chars <= len(marker):
        return cleaned[:max_chars]

    return cleaned[: max_chars - len(marker)].rstrip() + marker


def select_context_rows(
    rows: list[dict[str, Any]],
    max_rows: int,
    max_chunk_chars: int,
    max_total_chars: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_chars = 0

    row_limit = max_rows if max_rows > 0 else len(rows)

    for row in rows:
        if len(selected) >= row_limit:
            break

        text = str(row.get("text") or "").strip()
        if not text:
            continue

        if max_chunk_chars > 0:
            text = clip_text(text, max_chunk_chars)

        if max_total_chars > 0:
            remaining = max_total_chars - used_chars
            if remaining <= 0:
                break
            text = clip_text(text, remaining, marker="")

        if not text:
            continue

        copied = dict(row)
        copied["text"] = text
        selected.append(copied)
        used_chars += len(text)

    return selected


def format_context(rows: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, row in enumerate(rows, start=1):
        source_ref = row.get("source_ref") or "unknown"
        text = row.get("text") or ""
        blocks.append(f"[Context {idx}] ({source_ref})\\n{text}")
    return "\\n\\n".join(blocks)


def format_history(messages: list[dict[str, Any]], max_turns: int, max_chars: int) -> str:
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

    return clip_text("\\n".join(lines), max_chars)


def build_rag_prompt(question: str, context: str, history: str) -> str:
    return (
        "You are a technical assistant for product documentation and team discussion archives.\\n"
        "Use only the retrieved context to answer factual questions.\\n"
        "If the answer is not present in context, say you do not know.\\n\\n"
        f"Conversation history:\\n{history or 'No prior conversation.'}\\n\\n"
        f"Retrieved context:\\n{context or 'No context retrieved.'}\\n\\n"
        f"Question:\\n{question}"
    )
