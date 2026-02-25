from __future__ import annotations

import argparse
from typing import Any

from common.io_utils import read_yaml
from common.mlx_utils import MLXLoadedGenerator
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    format_context,
    select_context_rows,
)
from rag.retrieve import run_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Gemma with local retrieval context."
    )
    parser.add_argument("question")
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--adapter-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rag_cfg = read_yaml(args.rag_config)
    model_cfg = read_yaml(args.models_config)

    rows = run_retrieval(args.question, rag_cfg)
    max_context_chunks = int(rag_cfg.get("max_context_chunks", 4))
    max_chunk_chars = int(rag_cfg.get("max_chunk_chars", 1200))
    max_total_context_chars = int(rag_cfg.get("max_total_context_chars", 5000))
    max_prompt_chars = int(rag_cfg.get("max_prompt_chars", 12000))

    selected_rows = select_context_rows(
        rows=rows,
        max_rows=max_context_chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_chars=max_total_context_chars,
    )
    context = format_context(selected_rows)

    answer_cfg: dict[str, Any] = model_cfg.get("answer_model", {})
    model_name = str(answer_cfg.get("model"))
    max_tokens = int(answer_cfg.get("max_tokens", 256))
    temperature = float(answer_cfg.get("temperature", 0.2))
    trust_remote_code = bool(answer_cfg.get("trust_remote_code", True))

    adapter_path = args.adapter_path or answer_cfg.get("adapter_path") or None

    prompt = build_rag_prompt(args.question, context=context, history="")
    prompt = clip_text(prompt, max_prompt_chars)

    generator = MLXLoadedGenerator(
        model=model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
        trust_remote_code=trust_remote_code,
    )
    answer = generator.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for row in selected_rows:
        print(f"- {row.get('source_ref')} (score={row.get('score'):.4f})")


if __name__ == "__main__":
    main()
