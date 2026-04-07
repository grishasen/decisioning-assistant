from __future__ import annotations

import argparse
from typing import Any

from common.io_utils import read_yaml
from common.mlx_utils import MLXLoadedGenerator
from rag.answer_selection import AnswerSelectionConfig, generate_best_answer
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    clip_text_to_tokens,
    format_context,
    normalize_prompt_mode,
    select_context_rows,
)
from rag.retrieve import (
    build_local_retriever,
    resolve_answer_rerank_resources,
    resolve_top_k,
)


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for chat local.
    """
    parser = argparse.ArgumentParser(
        description="Ask a local model with local retrieval context."
    )
    parser.add_argument("question")
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Signature: def main() -> None.

    Run the chat local entrypoint.
    """
    args = parse_args()

    rag_cfg = read_yaml(args.rag_config)
    model_cfg = read_yaml(args.models_config)

    retriever = build_local_retriever(rag_cfg)
    rows = retriever.search(args.question, resolve_top_k(rag_cfg, args.top_k))

    max_context_chunks = int(rag_cfg.get("max_context_chunks", 4))
    max_chunk_chars = int(rag_cfg.get("max_chunk_chars", 1200))
    max_total_context_chars = int(rag_cfg.get("max_total_context_chars", 5000))
    max_prompt_chars = int(rag_cfg.get("max_prompt_chars", 12000))

    max_chunk_tokens = int(rag_cfg.get("max_chunk_tokens", 0))
    max_total_context_tokens = int(rag_cfg.get("max_total_context_tokens", 0))
    max_prompt_tokens = int(rag_cfg.get("max_prompt_tokens", 0))

    selected_rows = select_context_rows(
        rows=rows,
        max_rows=max_context_chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_context_chars,
        max_chunk_tokens=max_chunk_tokens,
        max_total_tokens=max_total_context_tokens,
    )
    context = format_context(selected_rows)

    answer_cfg: dict[str, Any] = model_cfg.get("answer_model", {})
    model_name = str(answer_cfg.get("model"))
    max_tokens = int(answer_cfg.get("max_tokens", 256))
    temperature = float(answer_cfg.get("temperature", 0.2))
    trust_remote_code = bool(answer_cfg.get("trust_remote_code", True))

    answer_sel_cfg = AnswerSelectionConfig(
        sample_count=max(1, int(rag_cfg.get("answer_sample_count", 4))),
        rerank_mode=str(rag_cfg.get("answer_rerank_mode", "cross_encoder")).strip().lower(),
        rerank_alpha=max(0.0, min(1.0, float(rag_cfg.get("answer_rerank_alpha", 0.7)))),
        support_top_k=max(1, int(rag_cfg.get("answer_rerank_support_top_k", 3))),
    )
    reranker_model = str(rag_cfg.get("reranker_model", "BAAI/bge-reranker-base"))
    prompt_mode = normalize_prompt_mode(str(rag_cfg.get("prompt_mode", "grounded")))

    adapter_path = args.adapter_path or answer_cfg.get("adapter_path") or None

    prompt = build_rag_prompt(args.question, context=context, history="", prompt_mode=prompt_mode)
    prompt = clip_text(prompt, max_prompt_chars)
    prompt = clip_text_to_tokens(prompt, max_prompt_tokens)

    generator = MLXLoadedGenerator(
        model=model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
        trust_remote_code=trust_remote_code,
    )

    answer_embedder, answer_cross_encoder = resolve_answer_rerank_resources(
        retriever=retriever,
        sample_count=answer_sel_cfg.sample_count,
        rerank_mode=answer_sel_cfg.rerank_mode,
        reranker_model=reranker_model,
    )

    answer, ranked_candidates = generate_best_answer(
        generator=generator,
        prompt=prompt,
        question=args.question,
        context_rows=selected_rows,
        max_tokens=max_tokens,
        temperature=temperature,
        config=answer_sel_cfg,
        embedder=answer_embedder,
        normalize_embeddings=retriever.normalize_embeddings,
        cross_encoder=answer_cross_encoder,
    )

    print("Answer:\n")
    print(answer)
    if len(ranked_candidates) > 1:
        print(
            f"\nAnswer selection: generated {len(ranked_candidates)} unique candidate(s), "
            f"mode={answer_sel_cfg.rerank_mode}, alpha={answer_sel_cfg.rerank_alpha:.2f}"
        )
    print("\nSources:")
    for row in selected_rows:
        record_type = row.get("record_type") or "chunk"
        score = float(row.get("score") or 0.0)
        qdrant_score = row.get("qdrant_score")
        if isinstance(qdrant_score, (int, float)):
            print(
                f"- {row.get('source_ref')} [{record_type}] "
                f"(score={score:.4f}, vec={float(qdrant_score):.4f})"
            )
        else:
            print(f"- {row.get('source_ref')} [{record_type}] (score={score:.4f})")


if __name__ == "__main__":
    main()
