from __future__ import annotations

import argparse
import json
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from common.io_utils import read_yaml
from common.vector_utils import dot_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k passages from local Qdrant.")
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_bucket(row: dict[str, Any]) -> str:
    metadata_raw = row.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    room_id = str(metadata.get("room_id") or "").strip()
    thread_id = str(metadata.get("thread_id") or "").strip()
    if room_id and thread_id:
        return f"webex::{room_id}::thread::{thread_id}"

    linked_chunk_id = str(metadata.get("linked_chunk_id") or "").strip()
    if linked_chunk_id:
        return f"qa-link::{linked_chunk_id}"

    doc_id = str(row.get("doc_id") or "").strip()
    if doc_id:
        return f"doc::{doc_id}"

    source_ref = str(row.get("source_ref") or "").strip()
    if source_ref:
        return f"source::{source_ref.split(':block=')[0]}"

    chunk_id = str(row.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk::{chunk_id}"

    return "unknown"


def _embedding_cosine_scores(
    question: str,
    rows: list[dict[str, Any]],
    embedder: SentenceTransformer,
    normalize_embeddings: bool,
) -> list[float]:
    if not rows:
        return []

    query_vector = embedder.encode([question], normalize_embeddings=normalize_embeddings)[0]
    doc_vectors = embedder.encode(
        [str(row.get("text") or "") for row in rows],
        normalize_embeddings=normalize_embeddings,
    )

    scores: list[float] = []
    for doc_vector in doc_vectors:
        scores.append(dot_score(query_vector, doc_vector))
    return scores


def _cross_encoder_scores(
    question: str,
    rows: list[dict[str, Any]],
    reranker_model: str,
    cross_encoder: Any | None,
) -> list[float]:
    if not rows:
        return []

    model = cross_encoder
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(reranker_model)

    pairs = [(question, str(row.get("text") or "")) for row in rows]
    raw_scores = model.predict(pairs, show_progress_bar=False)
    return [float(score) for score in raw_scores]


def _apply_source_cap(
    rows: list[dict[str, Any]],
    max_per_source: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if max_per_source <= 0:
        return rows[:top_k]

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    source_counts: dict[str, int] = {}

    for row in rows:
        bucket = _source_bucket(row)
        if source_counts.get(bucket, 0) >= max_per_source:
            continue

        row_key = "||".join(
            [
                str(row.get("chunk_id") or ""),
                str(row.get("source_ref") or ""),
                str(row.get("record_type") or ""),
            ]
        )
        if row_key in selected_keys:
            continue

        selected.append(row)
        selected_keys.add(row_key)
        source_counts[bucket] = source_counts.get(bucket, 0) + 1
        if len(selected) >= top_k:
            return selected

    for row in rows:
        row_key = "||".join(
            [
                str(row.get("chunk_id") or ""),
                str(row.get("source_ref") or ""),
                str(row.get("record_type") or ""),
            ]
        )
        if row_key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(row_key)
        if len(selected) >= top_k:
            break

    return selected


def postprocess_retrieval_rows(
    question: str,
    rows: list[dict[str, Any]],
    *,
    embedder: SentenceTransformer,
    normalize_embeddings: bool,
    top_k: int,
    score_threshold: float = 0.0,
    rerank_mode: str = "cross_encoder",
    reranker_model: str = "BAAI/bge-reranker-base",
    rerank_alpha: float = 0.65,
    max_per_source: int = 0,
    qa_pair_score_boost: float = 0.0,
    cross_encoder: Any | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    candidates: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        qdrant_score = _as_float(row.get("qdrant_score", row.get("score", 0.0)))
        if score_threshold > 0.0 and qdrant_score < score_threshold:
            continue

        copied = dict(row)
        copied["text"] = text
        copied["qdrant_score"] = qdrant_score
        candidates.append(copied)

    if not candidates:
        return []

    mode = (rerank_mode or "none").strip().lower()
    rerank_scores: list[float]

    if mode == "embedding_cosine":
        rerank_scores = _embedding_cosine_scores(
            question,
            candidates,
            embedder,
            normalize_embeddings=normalize_embeddings,
        )
    elif mode == "cross_encoder":
        try:
            rerank_scores = _cross_encoder_scores(
                question=question,
                rows=candidates,
                reranker_model=reranker_model,
                cross_encoder=cross_encoder,
            )
        except Exception:
            rerank_scores = [row["qdrant_score"] for row in candidates]
    else:
        rerank_scores = [row["qdrant_score"] for row in candidates]

    alpha = max(0.0, min(1.0, _as_float(rerank_alpha, 0.65)))

    scored: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        qdrant_score = row["qdrant_score"]
        rerank_score = rerank_scores[idx] if idx < len(rerank_scores) else qdrant_score

        final_score = ((1.0 - alpha) * qdrant_score) + (alpha * float(rerank_score))
        if str(row.get("record_type") or "").lower() == "qa_pair":
            final_score += _as_float(qa_pair_score_boost, 0.0)

        row["rerank_score"] = float(rerank_score)
        row["score"] = float(final_score)
        scored.append(row)

    scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return _apply_source_cap(scored, max_per_source=max_per_source, top_k=top_k)


def run_retrieval(
    question: str,
    cfg: dict[str, Any],
    top_k_override: int = 0,
) -> list[dict[str, Any]]:
    qdrant_path = str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = str(cfg.get("collection_name", "docs"))
    embedding_model = str(cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))

    top_k = int(top_k_override or cfg.get("top_k", 6))
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    fetch_k = int(cfg.get("fetch_k", max(top_k * 3, top_k)))
    fetch_k = max(fetch_k, top_k)

    score_threshold = _as_float(cfg.get("score_threshold", 0.0), 0.0)
    rerank_mode = str(cfg.get("rerank_mode", "cross_encoder"))
    reranker_model = str(cfg.get("reranker_model", "BAAI/bge-reranker-base"))
    rerank_alpha = _as_float(cfg.get("rerank_alpha", 0.65), 0.65)
    max_per_source = int(cfg.get("max_per_source", 0))
    qa_pair_score_boost = _as_float(cfg.get("qa_pair_score_boost", 0.0), 0.0)

    embedder = SentenceTransformer(embedding_model)
    vector = embedder.encode([question], normalize_embeddings=normalize_embeddings)[0].tolist()

    client = QdrantClient(path=qdrant_path)
    resp = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=fetch_k,
        with_payload=True,
    )

    rows: list[dict[str, Any]] = []
    for point in resp.points:
        payload = point.payload or {}
        qdrant_score = _as_float(point.score, 0.0)
        rows.append(
            {
                "score": qdrant_score,
                "qdrant_score": qdrant_score,
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "source_ref": payload.get("source_ref"),
                "source_type": payload.get("source_type"),
                "record_type": payload.get("record_type", "chunk"),
                "text": payload.get("text") or "",
                "metadata": payload.get("metadata", {}),
            }
        )

    return postprocess_retrieval_rows(
        question=question,
        rows=rows,
        embedder=embedder,
        normalize_embeddings=normalize_embeddings,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank_mode=rerank_mode,
        reranker_model=reranker_model,
        rerank_alpha=rerank_alpha,
        max_per_source=max_per_source,
        qa_pair_score_boost=qa_pair_score_boost,
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    rows = run_retrieval(args.question, cfg, args.top_k)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
