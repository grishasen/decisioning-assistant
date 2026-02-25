from __future__ import annotations

import argparse
import hashlib
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml
from common.logging_utils import get_logger
from common.schemas import ChunkRecord

logger = get_logger(__name__)


def create_uuid_from_string(val: str) -> uuid.UUID:
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Qdrant index from chunk JSONL."
    )
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override embedding/index batch size. If 0, uses rag.yaml index_batch_size.",
    )
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def _load_chunks(path: str) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for row in iter_jsonl(path):
        chunks.append(ChunkRecord.model_validate(row))
    return chunks


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = read_yaml(args.config)

    chunks_path = str(cfg.get("chunks_path", "data/staging/chunks/chunks.jsonl"))
    qdrant_path = str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = str(cfg.get("collection_name", "docs"))
    embedding_model = str(cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
    batch_size = int(args.batch_size or cfg.get("index_batch_size", 128))
    if batch_size <= 0:
        raise ValueError("index batch size must be > 0")

    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise RuntimeError(f"No chunks found at {chunks_path}")

    embedder = SentenceTransformer(embedding_model)
    sample_vector = embedder.encode(
        [chunks[0].text], normalize_embeddings=normalize_embeddings
    )[0]
    vector_size = int(sample_vector.shape[0])

    client = QdrantClient(path=qdrant_path)
    if args.recreate:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        collections = {c.name for c in client.get_collections().collections}
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    for start in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        vectors = embedder.encode(texts, normalize_embeddings=normalize_embeddings)

        points: list[PointStruct] = []
        for chunk, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=create_uuid_from_string(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "source_ref": chunk.source_ref,
                        "source_type": chunk.source_type,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    logger.info(
        "Indexed %s chunks into collection '%s' at %s (batch_size=%s)",
        len(chunks),
        collection_name,
        qdrant_path,
        batch_size,
    )


if __name__ == "__main__":
    main()
