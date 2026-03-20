from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from common.text_utils import normalize_whitespace


@dataclass(frozen=True)
class WebexThreadLine:
    timestamp: str
    author: str
    message_text: str


def parse_webex_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_webex_thread_message_line(
    *,
    author: str,
    message_text: str,
    created: datetime | None,
) -> str:
    cleaned_message = str(message_text or "").strip()
    if not cleaned_message:
        return ""

    cleaned_author = str(author or "unknown-user").strip() or "unknown-user"
    timestamp = created.isoformat() if isinstance(created, datetime) else "unknown-time"
    return f"[{timestamp}] {cleaned_author}: {cleaned_message}"


def parse_webex_thread_message_line(line: str) -> WebexThreadLine | None:
    stripped = str(line or "").strip()
    if not stripped:
        return None

    timestamp = "unknown-time"
    body = stripped
    if stripped.startswith("[") and "] " in stripped:
        timestamp, body = stripped[1:].split("] ", 1)
        timestamp = timestamp.strip() or "unknown-time"

    if ":" in body:
        author, message_text = body.split(":", 1)
        cleaned_author = author.strip() or "unknown-user"
        cleaned_message = normalize_whitespace(message_text)
    else:
        cleaned_author = "unknown-user"
        cleaned_message = normalize_whitespace(body)

    if not cleaned_message:
        return None

    return WebexThreadLine(
        timestamp=timestamp,
        author=cleaned_author,
        message_text=cleaned_message,
    )


def parse_webex_thread_lines(text: str) -> list[WebexThreadLine]:
    parsed: list[WebexThreadLine] = []
    for raw_line in str(text or "").splitlines():
        item = parse_webex_thread_message_line(raw_line)
        if item is not None:
            parsed.append(item)
    return parsed
