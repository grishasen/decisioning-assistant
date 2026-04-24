"""CLI entrypoint for retrieval evaluation on labeled RAG queries.

This script measures whether the expected source or chunk appears in the top-k
retrieval results after vector search, reranking, QA boosting, and Webex recency
weighting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.io_utils import read_yaml, write_json
from rag.eval_common import best_match_rank, has_retrieval_labels, load_eval_cases
from rag.retrieve import build_local_retriever, resolve_top_k


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for eval retrieval.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval ranking on a labeled JSONL query set."
    )
    parser.add_argument("--rag-config", default="configs/rag.yaml")
    parser.add_argument("--eval-path", default="configs/rag_eval.sample.jsonl")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--output-path",
        default="data/eval/reports/retrieval_report.json",
        help="Where to write the retrieval evaluation report.",
    )
    return parser.parse_args()


def _summarize_case(case: dict[str, Any], rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Signature: def _summarize_case(case: dict[str, Any], rows: list[dict[str, Any]], top_k: int) -> dict[str, Any].

    Summarize case.
    """
    match_rank = best_match_rank(rows, case)
    top_rows = [
        {
            "rank": idx,
            "source_ref": row.get("source_ref"),
            "chunk_id": row.get("chunk_id"),
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
            "recency_bonus": (
                float(row["recency_bonus"])
                if isinstance(row.get("recency_bonus"), (int, float))
                else None
            ),
        }
        for idx, row in enumerate(rows[:top_k], start=1)
    ]
    return {
        "query_id": case.get("query_id"),
        "question": case.get("question"),
        "match_rank": match_rank,
        "hit_at_k": bool(match_rank is not None and match_rank <= top_k),
        "top_rows": top_rows,
    }


def main() -> None:
    """Signature: def main() -> None.

    Run the eval retrieval entrypoint.
    """
    args = parse_args()
    rag_cfg = read_yaml(args.rag_config)
    top_k = resolve_top_k(rag_cfg, args.top_k)
    cases = load_eval_cases(args.eval_path)

    retriever = build_local_retriever(rag_cfg)

    results: list[dict[str, Any]] = []
    labeled_cases = 0
    hit_count = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        rows = retriever.search(str(case["question"]), top_k)
        case_result = _summarize_case(case, rows, top_k)
        results.append(case_result)

        if has_retrieval_labels(case):
            labeled_cases += 1
            match_rank = case_result.get("match_rank")
            if isinstance(match_rank, int) and match_rank > 0:
                hit_count += 1
                reciprocal_rank_sum += 1.0 / float(match_rank)

    summary = {
        "eval_path": str(Path(args.eval_path).resolve()),
        "top_k": top_k,
        "total_cases": len(cases),
        "labeled_cases": labeled_cases,
        "hit_at_k": hit_count,
        "recall_at_k": (hit_count / labeled_cases) if labeled_cases else None,
        "mrr": (reciprocal_rank_sum / labeled_cases) if labeled_cases else None,
    }

    report = {
        "summary": summary,
        "cases": results,
    }
    write_json(args.output_path, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
