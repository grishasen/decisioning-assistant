"""Shared helpers for retrieval and answering evaluation runs.

The evaluation JSONL schema is intentionally lightweight so it can be maintained
alongside the project configs. These helpers keep matching and abstention rules
consistent across retrieval-only and end-to-end answer benchmarks.
"""

from __future__ import annotations

from typing import Any

from common.io_utils import iter_jsonl

_ABSTAIN_HINTS = (
    "i do not know",
    "i don't know",
    "cannot determine from the retrieved context",
    "can't determine from the retrieved context",
    "not enough information in the retrieved context",
    "insufficient information in the retrieved context",
    "not present in the retrieved context",
    "not available in the retrieved context",
    "not enough evidence in the retrieved context",
)


def load_eval_cases(path: str) -> list[dict[str, Any]]:
    """Signature: def load_eval_cases(path: str) -> list[dict[str, Any]].

    Load eval cases.
    """
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path), start=1):
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"Eval row {index} is missing a question")
        case = dict(row)
        case.setdefault("query_id", f"case-{index}")
        case["question"] = question
        cases.append(case)
    return cases


def expected_source_refs(case: dict[str, Any]) -> set[str]:
    """Signature: def expected_source_refs(case: dict[str, Any]) -> set[str].

    Expected source refs.
    """
    values = case.get("expected_source_refs") or []
    return {str(value).strip() for value in values if str(value).strip()}


def expected_chunk_ids(case: dict[str, Any]) -> set[str]:
    """Signature: def expected_chunk_ids(case: dict[str, Any]) -> set[str].

    Expected chunk ids.
    """
    values = case.get("expected_chunk_ids") or []
    return {str(value).strip() for value in values if str(value).strip()}


def expected_answer_phrases(case: dict[str, Any]) -> list[str]:
    """Signature: def expected_answer_phrases(case: dict[str, Any]) -> list[str].

    Expected answer phrases.
    """
    values = case.get("expected_answer_contains") or []
    return [str(value).strip() for value in values if str(value).strip()]


def has_retrieval_labels(case: dict[str, Any]) -> bool:
    """Signature: def has_retrieval_labels(case: dict[str, Any]) -> bool.

    Return whether retrieval labels.
    """
    return bool(expected_source_refs(case) or expected_chunk_ids(case))


def row_matches_case(row: dict[str, Any], case: dict[str, Any]) -> bool:
    """Signature: def row_matches_case(row: dict[str, Any], case: dict[str, Any]) -> bool.

    Row matches case.
    """
    source_refs = expected_source_refs(case)
    chunk_ids = expected_chunk_ids(case)

    source_ref = str(row.get("source_ref") or "").strip()
    chunk_id = str(row.get("chunk_id") or "").strip()
    metadata_raw = row.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    linked_chunk_id = str(metadata.get("linked_chunk_id") or "").strip()

    if source_refs and source_ref in source_refs:
        return True
    if chunk_ids and (chunk_id in chunk_ids or linked_chunk_id in chunk_ids):
        return True
    return False


def best_match_rank(rows: list[dict[str, Any]], case: dict[str, Any]) -> int | None:
    """Signature: def best_match_rank(rows: list[dict[str, Any]], case: dict[str, Any]) -> int | None.

    Return the best match rank.
    """
    if not has_retrieval_labels(case):
        return None
    for index, row in enumerate(rows, start=1):
        if row_matches_case(row, case):
            return index
    return None


def is_abstained(answer: str) -> bool:
    """Signature: def is_abstained(answer: str) -> bool.

    Return whether abstained.
    """
    normalized = " ".join(str(answer or "").lower().split())
    if not normalized:
        return True
    return any(pattern in normalized for pattern in _ABSTAIN_HINTS)


def phrase_hits(answer: str, phrases: list[str]) -> list[str]:
    """Signature: def phrase_hits(answer: str, phrases: list[str]) -> list[str].

    Return hits.
    """
    normalized_answer = str(answer or "").lower()
    return [phrase for phrase in phrases if phrase.lower() in normalized_answer]
