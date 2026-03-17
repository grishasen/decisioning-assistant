from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer

from common.vector_utils import dot_score


@dataclass(frozen=True)
class AnswerSelectionConfig:
    sample_count: int = 1
    rerank_mode: str = "none"
    rerank_alpha: float = 0.7
    support_top_k: int = 3


def _unique_candidates(candidates: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if not cleaned or cleaned in seen:
            continue
        unique.append(cleaned)
        seen.add(cleaned)
    return unique


def generate_answer_candidates(
        generator: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        sample_count: int,
) -> list[str]:
    requested = max(1, int(sample_count))
    attempts = max(requested, 1)
    if requested > 1:
        attempts = max(requested * 2, requested + 2)

    raw_candidates: list[str] = []
    for _ in range(attempts):
        raw_candidates.append(
            generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        unique = _unique_candidates(raw_candidates)
        if len(unique) >= requested:
            return unique[:requested]

    unique = _unique_candidates(raw_candidates)
    if not unique:
        raise RuntimeError("No non-empty answers were generated")
    return unique[:requested]


def _embedding_relevance_scores(
        question: str,
        candidates: list[str],
        embedder: SentenceTransformer,
        normalize_embeddings: bool,
) -> list[float]:
    query_vector = embedder.encode([question], normalize_embeddings=normalize_embeddings)[0]
    answer_vectors = embedder.encode(candidates, normalize_embeddings=normalize_embeddings)
    return [dot_score(query_vector, vector) for vector in answer_vectors]


def _embedding_support_scores(
        question: str,
        candidates: list[str],
        context_rows: list[dict[str, Any]],
        support_top_k: int,
        embedder: SentenceTransformer,
        normalize_embeddings: bool,
) -> list[float]:
    support_rows = [
        str(row.get("text") or "").strip()
        for row in context_rows[: max(1, int(support_top_k))]
        if str(row.get("text") or "").strip()
    ]
    if not support_rows:
        return [0.0 for _ in candidates]

    support_queries = [
        f"Question: {question}\n\nContext: {row_text}" for row_text in support_rows
    ]
    query_vectors = embedder.encode(
        support_queries,
        normalize_embeddings=normalize_embeddings,
    )
    answer_vectors = embedder.encode(candidates, normalize_embeddings=normalize_embeddings)

    scores: list[float] = []
    for answer_vector in answer_vectors:
        pair_scores = [dot_score(query_vector, answer_vector) for query_vector in query_vectors]
        scores.append(max(pair_scores) if pair_scores else 0.0)
    return scores


def _cross_encoder_relevance_scores(
        question: str,
        candidates: list[str],
        cross_encoder: Any,
) -> list[float]:
    pairs = [(question, candidate) for candidate in candidates]
    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    return [float(score) for score in raw_scores]


def _cross_encoder_support_scores(
        question: str,
        candidates: list[str],
        context_rows: list[dict[str, Any]],
        support_top_k: int,
        cross_encoder: Any,
) -> list[float]:
    support_rows = [
        str(row.get("text") or "").strip()
        for row in context_rows[: max(1, int(support_top_k))]
        if str(row.get("text") or "").strip()
    ]
    if not support_rows:
        return [0.0 for _ in candidates]

    pairs: list[tuple[str, str]] = []
    for candidate in candidates:
        for row_text in support_rows:
            pairs.append((f"Question: {question}\n\nContext: {row_text}", candidate))

    raw_scores = [float(score) for score in cross_encoder.predict(pairs, show_progress_bar=False)]
    stride = len(support_rows)
    return [
        max(raw_scores[index: index + stride]) if stride > 0 else 0.0
        for index in range(0, len(raw_scores), stride)
    ]


def rerank_answer_candidates(
        question: str,
        candidates: list[str],
        context_rows: list[dict[str, Any]],
        *,
        config: AnswerSelectionConfig,
        embedder: SentenceTransformer | None,
        normalize_embeddings: bool,
        cross_encoder: Any | None,
) -> list[dict[str, Any]]:
    unique_candidates = _unique_candidates(candidates)
    if not unique_candidates:
        return []

    mode = str(config.rerank_mode or "none").strip().lower()
    alpha = max(0.0, min(1.0, float(config.rerank_alpha)))
    support_top_k = max(1, int(config.support_top_k))

    if len(unique_candidates) == 1 or mode == "none":
        return [
            {
                "answer": answer,
                "score": 0.0,
                "relevance_score": 0.0,
                "support_score": 0.0,
            }
            for answer in unique_candidates
        ]

    if mode == "cross_encoder" and cross_encoder is not None:
        relevance_scores = _cross_encoder_relevance_scores(question, unique_candidates, cross_encoder)
        support_scores = _cross_encoder_support_scores(
            question,
            unique_candidates,
            context_rows,
            support_top_k,
            cross_encoder,
        )
    elif embedder is not None:
        relevance_scores = _embedding_relevance_scores(
            question,
            unique_candidates,
            embedder,
            normalize_embeddings,
        )
        support_scores = _embedding_support_scores(
            question,
            unique_candidates,
            context_rows,
            support_top_k,
            embedder,
            normalize_embeddings,
        )
    else:
        return [
            {
                "answer": answer,
                "score": 0.0,
                "relevance_score": 0.0,
                "support_score": 0.0,
            }
            for answer in unique_candidates
        ]

    ranked: list[dict[str, Any]] = []
    for index, answer in enumerate(unique_candidates):
        relevance_score = relevance_scores[index] if index < len(relevance_scores) else 0.0
        support_score = support_scores[index] if index < len(support_scores) else 0.0
        final_score = ((1.0 - alpha) * float(relevance_score)) + (alpha * float(support_score))
        ranked.append(
            {
                "answer": answer,
                "score": float(final_score),
                "relevance_score": float(relevance_score),
                "support_score": float(support_score),
            }
        )

    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked


def generate_best_answer(
        generator: Any,
        prompt: str,
        question: str,
        context_rows: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        config: AnswerSelectionConfig,
        embedder: SentenceTransformer | None,
        normalize_embeddings: bool,
        cross_encoder: Any | None,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = generate_answer_candidates(
        generator=generator,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        sample_count=config.sample_count,
    )
    ranked_candidates = rerank_answer_candidates(
        question=question,
        candidates=candidates,
        context_rows=context_rows,
        config=config,
        embedder=embedder,
        normalize_embeddings=normalize_embeddings,
        cross_encoder=cross_encoder,
    )
    if not ranked_candidates:
        raise RuntimeError("No answer candidates available for selection")
    return str(ranked_candidates[0]["answer"]), ranked_candidates
