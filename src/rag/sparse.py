from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from qdrant_client.models import SparseVector

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_./:][A-Za-z0-9]+)*")
_SPLIT_RE = re.compile(r"[-_./:]+")


@dataclass(frozen=True)
class SparseTextConfig:
    """Store sparse lexical vectorization settings."""

    max_terms: int = 512
    min_token_chars: int = 2


def _stable_term_index(term: str) -> int:
    """Return a stable unsigned 32-bit index for a lexical term."""
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _iter_terms(text: str, min_token_chars: int) -> Iterable[str]:
    """Yield normalized exact and separator-split lexical terms."""
    minimum = max(1, int(min_token_chars))
    for raw_token in _TOKEN_RE.findall(str(text or "")):
        token = raw_token.casefold()
        if len(token) >= minimum:
            yield token

        for part in _SPLIT_RE.split(token):
            if part and part != token and len(part) >= minimum:
                yield part


def sparse_indices_values(text: str, config: SparseTextConfig) -> tuple[list[int], list[float]]:
    """Convert text into sorted sparse-vector indices and sublinear term-frequency values."""
    counts: Counter[int] = Counter()
    for term in _iter_terms(text, config.min_token_chars):
        counts[_stable_term_index(term)] += 1

    if not counts:
        return [], []

    terms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if config.max_terms > 0:
        terms = terms[: config.max_terms]

    sorted_terms = sorted(terms, key=lambda item: item[0])
    indices = [index for index, _ in sorted_terms]
    values = [1.0 + math.log(float(count)) for _, count in sorted_terms]
    return indices, values


def sparse_vector_from_text(text: str, config: SparseTextConfig) -> SparseVector:
    """Build a Qdrant sparse vector from text."""
    indices, values = sparse_indices_values(text, config)
    return SparseVector(indices=indices, values=values)
