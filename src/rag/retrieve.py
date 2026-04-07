from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from common.io_utils import read_yaml
from common.vector_utils import dot_score
from common.webex_utils import parse_webex_datetime


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse command-line arguments for the standalone retrieval CLI.
    """
    parser = argparse.ArgumentParser(description="Retrieve top-k passages from local Qdrant.")
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def _as_float(value: Any, default: float = 0.0) -> float:
    """Signature: def _as_float(value: Any, default: float = 0.0) -> float.

    Return the value converted to float when possible.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_WEBEX_RECENCY_HINT_TERMS = (
    "latest",
    "recent",
    "recently",
    "current",
    "updated",
    "today",
    "yesterday",
    "this week",
    "this month",
    "last week",
    "last month",
)


def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Signature: def _metadata_dict(row: dict[str, Any]) -> dict[str, Any].

    Return the metadata mapping stored on a retrieval row.
    """
    metadata_raw = row.get("metadata")
    return metadata_raw if isinstance(metadata_raw, dict) else {}


def _row_rerank_text(row: dict[str, Any]) -> str:
    # retrieval_text keeps metadata context for retrieval while raw text stays cleaner for prompting.
    """Signature: def _row_rerank_text(row: dict[str, Any]) -> str.

    Return the text field used for second-stage retrieval reranking.
    """
    retrieval_text = str(row.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    return str(row.get("text") or "").strip()


def _row_source_type(row: dict[str, Any]) -> str:
    """Signature: def _row_source_type(row: dict[str, Any]) -> str.

    Return the normalized source type for a retrieval row.
    """
    source_type = str(row.get("source_type") or "").strip().lower()
    if source_type:
        return source_type

    metadata = _metadata_dict(row)
    meta_source = str(metadata.get("source_type") or "").strip().lower()
    if meta_source:
        return meta_source

    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref.startswith("webex::"):
        return "webex"
    if source_ref.startswith("pdf::"):
        return "pdf"
    return ""


def _question_prefers_recent_content(question: str) -> bool:
    """Signature: def _question_prefers_recent_content(question: str) -> bool.

    Return whether the question asks for recent or current information.
    """
    normalized = " ".join(str(question or "").lower().split())
    if not normalized:
        return False
    return any(term in normalized for term in _WEBEX_RECENCY_HINT_TERMS)


def _webex_timestamp(
    row: dict[str, Any],
    preferred_field: str,
) -> tuple[datetime | None, str]:
    """Signature: def _webex_timestamp(row: dict[str, Any], preferred_field: str) -> tuple[datetime | None, str].

    Return the best available Webex timestamp and the field it came from.
    """
    metadata = _metadata_dict(row)
    fields = [preferred_field] + [
        field for field in ("updated_at", "created_at", "ingested_at") if field != preferred_field
    ]

    for field in fields:
        value = metadata.get(field)
        parsed: datetime | None
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = parse_webex_datetime(value)
        if parsed is None:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed, field

    return None, ""


def _webex_recency_bonus(
    question: str,
    row: dict[str, Any],
    *,
    enabled: bool,
    preferred_field: str,
    half_life_days: float,
    max_bonus: float,
    apply_to_qa_pairs: bool,
    query_adaptive: bool,
    recent_half_life_days: float,
    recent_max_bonus: float,
    now_utc: datetime | None = None,
) -> tuple[float, float, float, str]:
    """Signature: def _webex_recency_bonus(question: str, row: dict[str, Any], *, enabled: bool, preferred_field: str, half_life_days: float, max_bonus: float, apply_to_qa_pairs: bool, query_adaptive: bool, recent_half_life_days: float, recent_max_bonus: float, now_utc: datetime | None = None) -> tuple[float, float, float, str].

    Compute the additive recency bonus for a Webex retrieval row.
    """
    if not enabled or _row_source_type(row) != "webex":
        return 0.0, 0.0, 0.0, ""

    if not apply_to_qa_pairs and str(row.get("record_type") or "").strip().lower() == "qa_pair":
        return 0.0, 0.0, 0.0, ""

    timestamp, timestamp_field = _webex_timestamp(row, preferred_field)
    if timestamp is None:
        return 0.0, 0.0, 0.0, ""

    effective_half_life = max(0.1, float(half_life_days))
    effective_max_bonus = max(0.0, float(max_bonus))

    if query_adaptive and _question_prefers_recent_content(question):
        if float(recent_half_life_days) > 0:
            effective_half_life = max(0.1, float(recent_half_life_days))
        if float(recent_max_bonus) > 0:
            effective_max_bonus = max(0.0, float(recent_max_bonus))

    if effective_max_bonus <= 0.0:
        return 0.0, 0.0, 0.0, timestamp_field

    current_time = now_utc or datetime.now(timezone.utc)
    age_days = max(0.0, (current_time - timestamp).total_seconds() / 86400.0)
    recency_score = math.exp(-math.log(2.0) * age_days / effective_half_life)
    return recency_score, effective_max_bonus * recency_score, age_days, timestamp_field


def _source_bucket(row: dict[str, Any]) -> str:
    """Signature: def _source_bucket(row: dict[str, Any]) -> str.

    Build the diversification bucket key for a retrieval row.
    """
    metadata = _metadata_dict(row)

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
    """Signature: def _embedding_cosine_scores(question: str, rows: list[dict[str, Any]], embedder: SentenceTransformer, normalize_embeddings: bool) -> list[float].

    Score retrieval rows with embedding cosine similarity.
    """
    if not rows:
        return []

    query_vector = embedder.encode([question], normalize_embeddings=normalize_embeddings)[0]
    doc_vectors = embedder.encode(
        [_row_rerank_text(row) for row in rows],
        normalize_embeddings=normalize_embeddings,
    )

    scores: list[float] = []
    for doc_vector in doc_vectors:
        scores.append(dot_score(query_vector, doc_vector))
    return scores


def load_cross_encoder_model(reranker_model: str) -> Any | None:
    """Signature: def load_cross_encoder_model(reranker_model: str) -> Any | None.

    Load the configured sentence-transformers cross-encoder reranker.
    """
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(reranker_model)
    except Exception:
        return None


def _cross_encoder_scores(
    question: str,
    rows: list[dict[str, Any]],
    reranker_model: str,
    cross_encoder: Any | None,
) -> list[float]:
    """Signature: def _cross_encoder_scores(question: str, rows: list[dict[str, Any]], reranker_model: str, cross_encoder: Any | None) -> list[float].

    Score retrieval rows with a cross-encoder reranker.
    """
    if not rows:
        return []

    model = cross_encoder or load_cross_encoder_model(reranker_model)
    if model is None:
        raise RuntimeError(f"Could not load cross-encoder reranker: {reranker_model}")

    pairs = [(question, _row_rerank_text(row)) for row in rows]
    raw_scores = model.predict(pairs, show_progress_bar=False)
    return [float(score) for score in raw_scores]


def _apply_source_cap(
    rows: list[dict[str, Any]],
    max_per_source: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Signature: def _apply_source_cap(rows: list[dict[str, Any]], max_per_source: int, top_k: int) -> list[dict[str, Any]].

    Limit how many final rows can come from the same source bucket.
    """
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
    webex_recency_enabled: bool = False,
    webex_recency_field: str = "updated_at",
    webex_recency_half_life_days: float = 45.0,
    webex_recency_max_bonus: float = 0.0,
    webex_recency_apply_to_qa_pairs: bool = True,
    webex_recency_query_adaptive: bool = True,
    webex_recency_recent_half_life_days: float = 14.0,
    webex_recency_recent_max_bonus: float = 0.0,
) -> list[dict[str, Any]]:
    """Signature: def postprocess_retrieval_rows(question: str, rows: list[dict[str, Any]], *, embedder: SentenceTransformer, normalize_embeddings: bool, top_k: int, score_threshold: float = 0.0, rerank_mode: str = 'cross_encoder', reranker_model: str = 'BAAI/bge-reranker-base', rerank_alpha: float = 0.65, max_per_source: int = 0, qa_pair_score_boost: float = 0.0, cross_encoder: Any | None = None, webex_recency_enabled: bool = False, webex_recency_field: str = 'updated_at', webex_recency_half_life_days: float = 45.0, webex_recency_max_bonus: float = 0.0, webex_recency_apply_to_qa_pairs: bool = True, webex_recency_query_adaptive: bool = True, webex_recency_recent_half_life_days: float = 14.0, webex_recency_recent_max_bonus: float = 0.0) -> list[dict[str, Any]].

    Filter, rerank, rescore, and diversify retrieved rows before prompt building.
    """
    if top_k <= 0:
        return []

    candidates: list[dict[str, Any]] = []
    for row in rows:
        ranking_text = _row_rerank_text(row)
        if not ranking_text:
            continue

        display_text = str(row.get("text") or "").strip() or ranking_text

        qdrant_score = _as_float(row.get("qdrant_score", row.get("score", 0.0)))
        if score_threshold > 0.0 and qdrant_score < score_threshold:
            continue

        copied = dict(row)
        copied["text"] = display_text
        copied["retrieval_text"] = ranking_text
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
            try:
                rerank_scores = _embedding_cosine_scores(
                    question,
                    candidates,
                    embedder,
                    normalize_embeddings=normalize_embeddings,
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

        recency_score, recency_bonus, recency_age_days, recency_field = _webex_recency_bonus(
            question,
            row,
            enabled=webex_recency_enabled,
            preferred_field=webex_recency_field,
            half_life_days=webex_recency_half_life_days,
            max_bonus=webex_recency_max_bonus,
            apply_to_qa_pairs=webex_recency_apply_to_qa_pairs,
            query_adaptive=webex_recency_query_adaptive,
            recent_half_life_days=webex_recency_recent_half_life_days,
            recent_max_bonus=webex_recency_recent_max_bonus,
        )
        final_score += recency_bonus

        row["rerank_score"] = float(rerank_score)
        row["score"] = float(final_score)
        if recency_field:
            row["recency_field"] = recency_field
            row["recency_score"] = float(recency_score)
            row["recency_bonus"] = float(recency_bonus)
            row["recency_age_days"] = float(recency_age_days)
        scored.append(row)

    scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return _apply_source_cap(scored, max_per_source=max_per_source, top_k=top_k)


class LocalRetriever:
    """Run local Qdrant retrieval, reranking, and recency-aware scoring."""
    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embedding_model: str,
        normalize_embeddings: bool,
        fetch_k: int,
        score_threshold: float,
        rerank_mode: str,
        reranker_model: str,
        rerank_alpha: float,
        max_per_source: int,
        qa_pair_score_boost: float,
        webex_recency_enabled: bool,
        webex_recency_field: str,
        webex_recency_half_life_days: float,
        webex_recency_max_bonus: float,
        webex_recency_apply_to_qa_pairs: bool,
        webex_recency_query_adaptive: bool,
        webex_recency_recent_half_life_days: float,
        webex_recency_recent_max_bonus: float,
    ) -> None:
        """Signature: def __init__(self, qdrant_path: str, collection_name: str, embedding_model: str, normalize_embeddings: bool, fetch_k: int, score_threshold: float, rerank_mode: str, reranker_model: str, rerank_alpha: float, max_per_source: int, qa_pair_score_boost: float, webex_recency_enabled: bool, webex_recency_field: str, webex_recency_half_life_days: float, webex_recency_max_bonus: float, webex_recency_apply_to_qa_pairs: bool, webex_recency_query_adaptive: bool, webex_recency_recent_half_life_days: float, webex_recency_recent_max_bonus: float) -> None.

        Load the embedding model, local Qdrant client, and reranker state.
        """
        self._collection_name = collection_name
        self._normalize_embeddings = normalize_embeddings
        self._embedder = SentenceTransformer(embedding_model)
        self._client = QdrantClient(path=qdrant_path)

        self._fetch_k = max(1, int(fetch_k))
        self._score_threshold = max(0.0, float(score_threshold))
        self._rerank_mode = str(rerank_mode or "none").strip().lower()
        self._reranker_model = str(reranker_model)
        self._rerank_alpha = max(0.0, min(1.0, float(rerank_alpha)))
        self._max_per_source = max(0, int(max_per_source))
        self._qa_pair_score_boost = float(qa_pair_score_boost)
        self._webex_recency_enabled = bool(webex_recency_enabled)
        self._webex_recency_field = str(webex_recency_field or "updated_at").strip() or "updated_at"
        self._webex_recency_half_life_days = max(0.1, float(webex_recency_half_life_days))
        self._webex_recency_max_bonus = max(0.0, float(webex_recency_max_bonus))
        self._webex_recency_apply_to_qa_pairs = bool(webex_recency_apply_to_qa_pairs)
        self._webex_recency_query_adaptive = bool(webex_recency_query_adaptive)
        self._webex_recency_recent_half_life_days = max(0.1, float(webex_recency_recent_half_life_days))
        self._webex_recency_recent_max_bonus = max(0.0, float(webex_recency_recent_max_bonus))

        self._cross_encoder: Any | None = None
        if self._rerank_mode == "cross_encoder":
            self._cross_encoder = load_cross_encoder_model(self._reranker_model)
            if self._cross_encoder is None:
                self._rerank_mode = "embedding_cosine"

    @property
    def embedder(self) -> SentenceTransformer:
        """Signature: def embedder(self) -> SentenceTransformer.

        Return the embedding model used for local retrieval.
        """
        return self._embedder

    @property
    def normalize_embeddings(self) -> bool:
        """Signature: def normalize_embeddings(self) -> bool.

        Return whether embedding vectors are normalized for scoring.
        """
        return self._normalize_embeddings

    @property
    def cross_encoder(self) -> Any | None:
        """Signature: def cross_encoder(self) -> Any | None.

        Return the loaded cross-encoder reranker, if one is available.
        """
        return self._cross_encoder

    @property
    def reranker_model(self) -> str:
        """Signature: def reranker_model(self) -> str.

        Return the configured cross-encoder model name.
        """
        return self._reranker_model

    @property
    def rerank_mode(self) -> str:
        """Signature: def rerank_mode(self) -> str.

        Return the active retrieval rerank mode.
        """
        return self._rerank_mode

    def ensure_cross_encoder(self, reranker_model: str | None = None) -> Any | None:
        """Signature: def ensure_cross_encoder(self, reranker_model: str | None = None) -> Any | None.

        Load or reuse the configured cross-encoder reranker.
        """
        if self._cross_encoder is not None:
            return self._cross_encoder

        model_name = str(reranker_model or self._reranker_model)
        loaded = load_cross_encoder_model(model_name)
        if loaded is None:
            return None

        if model_name == self._reranker_model:
            self._cross_encoder = loaded
        return loaded

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        """Signature: def search(self, question: str, top_k: int) -> list[dict[str, Any]].

        Retrieve, rerank, and score the best rows for a question.
        """
        requested_top_k = max(1, int(top_k))
        fetch_k = max(requested_top_k, self._fetch_k)

        vector = self._embedder.encode(
            [question],
            normalize_embeddings=self._normalize_embeddings,
        )[0].tolist()

        resp = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=fetch_k,
            with_payload=True,
        )

        rows: list[dict[str, Any]] = []
        for point in resp.points:
            payload = point.payload or {}
            qdrant_score = float(point.score or 0.0)
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
                    "retrieval_text": payload.get("retrieval_text") or payload.get("text") or "",
                    "metadata": payload.get("metadata", {}),
                }
            )

        return postprocess_retrieval_rows(
            question=question,
            rows=rows,
            embedder=self._embedder,
            normalize_embeddings=self._normalize_embeddings,
            top_k=requested_top_k,
            score_threshold=self._score_threshold,
            rerank_mode=self._rerank_mode,
            reranker_model=self._reranker_model,
            rerank_alpha=self._rerank_alpha,
            max_per_source=self._max_per_source,
            qa_pair_score_boost=self._qa_pair_score_boost,
            cross_encoder=self._cross_encoder,
            webex_recency_enabled=self._webex_recency_enabled,
            webex_recency_field=self._webex_recency_field,
            webex_recency_half_life_days=self._webex_recency_half_life_days,
            webex_recency_max_bonus=self._webex_recency_max_bonus,
            webex_recency_apply_to_qa_pairs=self._webex_recency_apply_to_qa_pairs,
            webex_recency_query_adaptive=self._webex_recency_query_adaptive,
            webex_recency_recent_half_life_days=self._webex_recency_recent_half_life_days,
            webex_recency_recent_max_bonus=self._webex_recency_recent_max_bonus,
        )


def resolve_top_k(cfg: dict[str, Any], top_k_override: int = 0) -> int:
    """Signature: def resolve_top_k(cfg: dict[str, Any], top_k_override: int = 0) -> int.

    Resolve the effective top-k value from config and overrides.
    """
    top_k = int(top_k_override or cfg.get("top_k", 6))
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    return top_k


def build_local_retriever(cfg: dict[str, Any]) -> LocalRetriever:
    """Signature: def build_local_retriever(cfg: dict[str, Any]) -> LocalRetriever.

    Build a LocalRetriever from rag.yaml settings.
    """
    top_k = resolve_top_k(cfg, 0)
    fetch_k = int(cfg.get("fetch_k", max(top_k * 3, top_k)))

    return LocalRetriever(
        qdrant_path=str(cfg.get("qdrant_path", "data/rag/vectordb")),
        collection_name=str(cfg.get("collection_name", "docs")),
        embedding_model=str(cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")),
        normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
        fetch_k=max(fetch_k, top_k),
        score_threshold=_as_float(cfg.get("score_threshold", 0.0), 0.0),
        rerank_mode=str(cfg.get("rerank_mode", "cross_encoder")),
        reranker_model=str(cfg.get("reranker_model", "BAAI/bge-reranker-base")),
        rerank_alpha=_as_float(cfg.get("rerank_alpha", 0.65), 0.65),
        max_per_source=int(cfg.get("max_per_source", 0)),
        qa_pair_score_boost=_as_float(cfg.get("qa_pair_score_boost", 0.0), 0.0),
        webex_recency_enabled=bool(cfg.get("webex_recency_enabled", True)),
        webex_recency_field=str(cfg.get("webex_recency_field", "updated_at")),
        webex_recency_half_life_days=_as_float(cfg.get("webex_recency_half_life_days", 45.0), 45.0),
        webex_recency_max_bonus=_as_float(cfg.get("webex_recency_max_bonus", 0.04), 0.04),
        webex_recency_apply_to_qa_pairs=bool(cfg.get("webex_recency_apply_to_qa_pairs", True)),
        webex_recency_query_adaptive=bool(cfg.get("webex_recency_query_adaptive", True)),
        webex_recency_recent_half_life_days=_as_float(
            cfg.get("webex_recency_recent_half_life_days", 14.0),
            14.0,
        ),
        webex_recency_recent_max_bonus=_as_float(
            cfg.get("webex_recency_recent_max_bonus", 0.1),
            0.1,
        ),
    )


def resolve_answer_rerank_resources(
    *,
    retriever: LocalRetriever,
    sample_count: int,
    rerank_mode: str,
    reranker_model: str,
) -> tuple[SentenceTransformer | None, Any | None]:
    """Signature: def resolve_answer_rerank_resources(*, retriever: LocalRetriever, sample_count: int, rerank_mode: str, reranker_model: str) -> tuple[SentenceTransformer | None, Any | None].

    Resolve the embedder or cross-encoder needed for answer reranking.
    """
    if sample_count <= 1:
        return None, None

    mode = str(rerank_mode or "none").strip().lower()
    answer_cross_encoder: Any | None = None
    answer_embedder: SentenceTransformer | None = None

    if mode == "cross_encoder":
        answer_cross_encoder = retriever.ensure_cross_encoder(reranker_model)
        if answer_cross_encoder is None:
            answer_embedder = retriever.embedder
    elif mode == "embedding_cosine":
        answer_embedder = retriever.embedder

    return answer_embedder, answer_cross_encoder


def run_retrieval(
    question: str,
    cfg: dict[str, Any],
    top_k_override: int = 0,
) -> list[dict[str, Any]]:
    """Signature: def run_retrieval(question: str, cfg: dict[str, Any], top_k_override: int = 0) -> list[dict[str, Any]].

    Run the end-to-end local retrieval pipeline for one question.
    """
    retriever = build_local_retriever(cfg)
    return retriever.search(question, resolve_top_k(cfg, top_k_override))


def main() -> None:
    """Signature: def main() -> None.

    Run the standalone retrieval CLI entrypoint.
    """
    args = parse_args()
    cfg = read_yaml(args.config)
    rows = run_retrieval(args.question, cfg, args.top_k)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
