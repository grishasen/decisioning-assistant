from __future__ import annotations

import argparse
import json
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from common.io_utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k passages from local Qdrant.")
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def run_retrieval(question: str, cfg: dict[str, Any], top_k_override: int = 0) -> list[dict[str, Any]]:
    qdrant_path = str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = str(cfg.get("collection_name", "docs"))
    embedding_model = str(cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
    top_k = int(top_k_override or cfg.get("top_k", 6))

    embedder = SentenceTransformer(embedding_model)
    vector = embedder.encode([question], normalize_embeddings=normalize_embeddings)[0].tolist()

    client = QdrantClient(path=qdrant_path)
    resp = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    rows: list[dict[str, Any]] = []
    for point in resp.points:
        rows.append(
            {
                "score": point.score,
                "chunk_id": point.payload.get("chunk_id"),
                "source_ref": point.payload.get("source_ref"),
                "text": point.payload.get("text"),
                "metadata": point.payload.get("metadata", {}),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    rows = run_retrieval(args.question, cfg, args.top_k)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
