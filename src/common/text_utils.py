from __future__ import annotations

import hashlib
import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_INITIALISM_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")
_NO_SPLIT_ABBREVIATIONS = {
    "al.",
    "approx.",
    "art.",
    "cf.",
    "ch.",
    "co.",
    "corp.",
    "dept.",
    "dr.",
    "e.g.",
    "eq.",
    "etc.",
    "fig.",
    "i.e.",
    "inc.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "p.",
    "pg.",
    "pp.",
    "prof.",
    "sec.",
    "sr.",
    "st.",
    "vs.",
}


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_newlines(text).strip()
    if not normalized:
        return []

    paragraphs: list[str] = []
    for block in _BLANK_LINE_RE.split(normalized):
        lines = [normalize_whitespace(line) for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        paragraph = " ".join(lines).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def _looks_like_abbreviation(token: str) -> bool:
    cleaned = token.rstrip('"\')]}').lower()
    if cleaned in _NO_SPLIT_ABBREVIATIONS:
        return True
    return bool(_INITIALISM_RE.fullmatch(token.rstrip('"\')]}')))


def split_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    sentences: list[str] = []
    start = 0
    idx = 0
    length = len(normalized)
    while idx < length:
        char = normalized[idx]
        if char not in ".!?":
            idx += 1
            continue

        end = idx + 1
        while end < length and normalized[end] in '"\')]}':
            end += 1

        candidate = normalized[start:end].strip()
        if not candidate:
            idx += 1
            continue

        token = candidate.split()[-1]
        if _looks_like_abbreviation(token):
            idx += 1
            continue

        next_idx = end
        while next_idx < length and normalized[next_idx].isspace():
            next_idx += 1

        if next_idx >= length:
            break

        next_char = normalized[next_idx]
        if next_char.isupper() or next_char.isdigit() or next_char in '"\'([{':
            sentences.append(candidate)
            start = next_idx
            idx = next_idx
            continue

        idx += 1

    remainder = normalized[start:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences or [normalized]


def _group_sentences(sentences: list[str], target_chars: int) -> list[str]:
    if not sentences:
        return []
    if target_chars <= 0:
        return [" ".join(sentences).strip()]

    groups: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if not current:
            current = [sentence]
            continue

        candidate = " ".join([*current, sentence]).strip()
        if len(candidate) <= target_chars:
            current.append(sentence)
            continue

        groups.append(" ".join(current).strip())
        current = [sentence]

    if current:
        groups.append(" ".join(current).strip())
    return [group for group in groups if group]


def pack_paragraphs(
    paragraphs: list[str],
    target_chars: int,
    min_chunk_chars: int,
) -> list[str]:
    if min_chunk_chars < 0:
        raise ValueError("min_chunk_chars must be >= 0")

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        combined = "\n\n".join(current).strip()
        if not combined:
            current = []
            return

        if len(combined) < min_chunk_chars and chunks:
            chunks[-1] = f"{chunks[-1]}\n\n{combined}".strip()
        else:
            chunks.append(combined)
        current = []

    for paragraph in paragraphs:
        units = [paragraph]
        if target_chars > 0 and len(paragraph) > target_chars:
            sentences = split_sentences(paragraph)
            if len(sentences) > 1:
                units = _group_sentences(sentences, target_chars)

        for unit in units:
            if not current:
                current = [unit]
                continue

            candidate = "\n\n".join([*current, unit]).strip()
            if target_chars > 0 and len(candidate) > target_chars:
                flush()
            current.append(unit)

            if target_chars > 0 and len("\n\n".join(current)) >= target_chars:
                flush()

    flush()
    return chunks
