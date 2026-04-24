from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from common.io_utils import append_jsonl


def source_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Return compact source metadata for feedback and eval review."""
    metadata_raw = row.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    return {
        "source_ref": row.get("source_ref"),
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "source_type": row.get("source_type"),
        "record_type": row.get("record_type"),
        "score": float(row.get("score") or 0.0),
        "qdrant_score": (
            float(row["qdrant_score"])
            if isinstance(row.get("qdrant_score"), (int, float))
            else None
        ),
        "rerank_score": (
            float(row["rerank_score"])
            if isinstance(row.get("rerank_score"), (int, float))
            else None
        ),
        "title": metadata.get("pdf_title") or metadata.get("title") or metadata.get("room_title"),
        "section": metadata.get("section_path") or metadata.get("section_title"),
    }


def build_feedback_row(
    *,
    message_id: str,
    rating: str,
    feedback_text: str,
    question: str,
    answer: str,
    retrieval_query: str,
    sources: list[dict[str, Any]],
    answer_time_seconds: float | None,
) -> dict[str, Any]:
    """Build one JSONL feedback row."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message_id": message_id,
        "rating": rating,
        "feedback_text": str(feedback_text or "").strip(),
        "question": question,
        "answer": answer,
        "retrieval_query": retrieval_query,
        "answer_time_seconds": answer_time_seconds,
        "selected_sources": [source_summary(row) for row in sources],
    }


def append_feedback(path: str, row: dict[str, Any]) -> None:
    """Append one feedback row to a JSONL file."""
    append_jsonl(path, [row])
