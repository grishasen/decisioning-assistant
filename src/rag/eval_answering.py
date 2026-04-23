"""CLI entrypoint for end-to-end RAG answer evaluation.

This benchmark exercises the real answer path: retrieval, context selection,
prompt construction, answer generation, and optional Best-of-N answer reranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.io_utils import read_yaml, write_json
from common.mlx_utils import MLXLoadedGenerator, mlx_generation_options_from_config
from rag.answer_selection import AnswerSelectionConfig, generate_best_answer
from rag.eval_common import (
    best_match_rank,
    expected_answer_phrases,
    is_abstained,
    load_eval_cases,
    phrase_hits,
)
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    clip_text_to_tokens,
    format_context,
    normalize_prompt_mode,
    select_context_rows,
)
from rag.retrieve import build_local_retriever, resolve_answer_rerank_resources, resolve_top_k


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for eval answering.
    """
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG answering evaluation on a labeled JSONL query set."
    )
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--eval-path", default="configs/rag_eval.sample.jsonl")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--output-path",
        default="data/eval/reports/answering_report.json",
        help="Where to write the answering evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    """Signature: def main() -> None.

    Run the eval answering entrypoint.
    """
    args = parse_args()
    rag_cfg = read_yaml(args.rag_config)
    model_cfg = read_yaml(args.models_config)
    prompt_mode = normalize_prompt_mode(str(rag_cfg.get("prompt_mode", "grounded")))

    top_k = resolve_top_k(rag_cfg, args.top_k)
    cases = load_eval_cases(args.eval_path)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    retriever = build_local_retriever(rag_cfg)
    answer_cfg: dict[str, Any] = model_cfg.get("answer_model", {})
    generator = MLXLoadedGenerator(
        model=str(answer_cfg.get("model")),
        adapter_path=(args.adapter_path or answer_cfg.get("adapter_path") or None),
        trust_remote_code=bool(answer_cfg.get("trust_remote_code", True)),
        provider=str(answer_cfg.get("provider", "mlx")),
        **mlx_generation_options_from_config(answer_cfg),
    )

    answer_sel_cfg = AnswerSelectionConfig(
        sample_count=max(1, int(rag_cfg.get("answer_sample_count", 4))),
        rerank_mode=str(rag_cfg.get("answer_rerank_mode", "cross_encoder")).strip().lower(),
        rerank_alpha=max(0.0, min(1.0, float(rag_cfg.get("answer_rerank_alpha", 0.7)))),
        support_top_k=max(1, int(rag_cfg.get("answer_rerank_support_top_k", 3))),
    )
    reranker_model = str(rag_cfg.get("reranker_model", "BAAI/bge-reranker-base"))
    answer_embedder, answer_cross_encoder = resolve_answer_rerank_resources(
        retriever=retriever,
        sample_count=answer_sel_cfg.sample_count,
        rerank_mode=answer_sel_cfg.rerank_mode,
        reranker_model=reranker_model,
    )

    max_context_chunks = int(rag_cfg.get("max_context_chunks", 4))
    max_chunk_chars = int(rag_cfg.get("max_chunk_chars", 1200))
    max_total_context_chars = int(rag_cfg.get("max_total_context_chars", 5000))
    max_prompt_chars = int(rag_cfg.get("max_prompt_chars", 12000))
    max_chunk_tokens = int(rag_cfg.get("max_chunk_tokens", 0))
    max_total_context_tokens = int(rag_cfg.get("max_total_context_tokens", 0))
    max_prompt_tokens = int(rag_cfg.get("max_prompt_tokens", 0))
    max_tokens = int(answer_cfg.get("max_tokens", 256))
    temperature = float(answer_cfg.get("temperature", 0.2))

    results: list[dict[str, Any]] = []
    total_answerable = 0
    total_unanswerable = 0
    abstain_true_positive = 0
    abstain_false_positive = 0
    phrase_labeled_cases = 0
    phrase_full_hits = 0

    for case in cases:
        question = str(case["question"])
        rows = retriever.search(question, top_k)
        selected_rows = select_context_rows(
            rows=rows,
            max_rows=max_context_chunks,
            max_chunk_chars=max_chunk_chars,
            max_total_chars=max_total_context_chars,
            max_chunk_tokens=max_chunk_tokens,
            max_total_tokens=max_total_context_tokens,
        )
        context = format_context(selected_rows)
        prompt = build_rag_prompt(question, context=context, history="", prompt_mode=prompt_mode)
        prompt = clip_text(prompt, max_prompt_chars)
        prompt = clip_text_to_tokens(prompt, max_prompt_tokens)
        answer, ranked_candidates = generate_best_answer(
            generator=generator,
            prompt=prompt,
            question=question,
            context_rows=selected_rows,
            max_tokens=max_tokens,
            temperature=temperature,
            config=answer_sel_cfg,
            embedder=answer_embedder,
            normalize_embeddings=retriever.normalize_embeddings,
            cross_encoder=answer_cross_encoder,
        )

        expected_phrases = expected_answer_phrases(case)
        hits = phrase_hits(answer, expected_phrases)
        abstained = is_abstained(answer)
        retrieval_rank = best_match_rank(rows, case)
        answerable = case.get("answerable")
        if answerable is True:
            total_answerable += 1
            if abstained:
                abstain_false_positive += 1
        elif answerable is False:
            total_unanswerable += 1
            if abstained:
                abstain_true_positive += 1

        if expected_phrases:
            phrase_labeled_cases += 1
            if len(hits) == len(expected_phrases):
                phrase_full_hits += 1

        results.append(
            {
                "query_id": case.get("query_id"),
                "question": question,
                "answer": answer,
                "abstained": abstained,
                "answerable": answerable,
                "retrieval_match_rank": retrieval_rank,
                "expected_answer_contains": expected_phrases,
                "phrase_hits": hits,
                "selected_sources": [row.get("source_ref") for row in selected_rows],
                "candidate_count": len(ranked_candidates),
            }
        )

    summary = {
        "eval_path": str(Path(args.eval_path).resolve()),
        "top_k": top_k,
        "prompt_mode": prompt_mode,
        "total_cases": len(cases),
        "answerable_cases": total_answerable,
        "unanswerable_cases": total_unanswerable,
        "unanswerable_abstain_rate": (
            abstain_true_positive / total_unanswerable if total_unanswerable else None
        ),
        "answerable_false_abstain_rate": (
            abstain_false_positive / total_answerable if total_answerable else None
        ),
        "phrase_full_hit_rate": (
            phrase_full_hits / phrase_labeled_cases if phrase_labeled_cases else None
        ),
    }

    report = {
        "summary": summary,
        "cases": results,
    }
    write_json(args.output_path, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
