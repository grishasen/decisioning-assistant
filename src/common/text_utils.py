from __future__ import annotations

import hashlib
import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    normalized = text.strip()
    if not normalized:
        return

    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            split_at = normalized.rfind("\n", start, end)
            if split_at == -1:
                split_at = normalized.rfind(" ", start, end)
            if split_at > start + chunk_size // 3:
                end = split_at

        chunk = normalized[start:end].strip()
        if chunk:
            yield chunk

        if end >= length:
            break
        start = max(0, end - chunk_overlap)
