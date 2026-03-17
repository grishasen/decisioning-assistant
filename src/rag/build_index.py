from __future__ import annotations

import argparse
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml
from common.logging_utils import get_logger
from common.schemas import ChunkRecord, QARecord

logger = get_logger(__name__)


def create_uuid_from_string(val: str) -> uuid.UUID:
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Qdrant index from chunk + QA JSONL (hybrid retrieval corpus)."
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


def _load_qa(path: str) -> list[QARecord]:
    qa_rows: list[QARecord] = []
    for row in iter_jsonl(path):
        qa_rows.append(QARecord.model_validate(row))
    return qa_rows


def _build_qa_text(qa: QARecord, text_mode: str, max_answer_chars: int) -> str:
    question = qa.question.strip()
    answer = qa.answer.strip()

    if max_answer_chars > 0 and len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars].rstrip()

    if text_mode == "question_only":
        return f"Question: {question}"
    if text_mode == "answer_only":
        return f"Answer: {answer}"
    if text_mode == "question_answer":
        return f"Question: {question}\nAnswer: {answer}"

    raise ValueError(
        "qa_text_mode must be one of: question_answer, question_only, answer_only"
    )


def _compact_qa_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}

    chunk_metadata_raw = metadata.get("chunk_metadata")
    if isinstance(chunk_metadata_raw, dict):
        for key in (
            "title",
            "product",
            "doc_version",
            "doc_type",
            "pdf_title",
            "section_title",
            "section_path",
            "active_heading",
            "page",
            "page_start",
            "page_end",
            "room_id",
            "room_title",
            "thread_id",
            "message_count",
            "webex_parent_message_link",
            "created_at",
            "updated_at",
            "ingested_at",
        ):
            value = chunk_metadata_raw.get(key)
            if value is not None and value != "":
                compact[key] = value

    for key, value in metadata.items():
        if key in {"chunk_text", "chunk_metadata"}:
            continue
        compact[key] = value

    return compact


@dataclass(frozen=True)
class IndexRecord:
    point_id: str
    text: str
    payload: dict[str, Any]


def _build_chunk_index_records(chunks: list[ChunkRecord]) -> list[IndexRecord]:
    rows: list[IndexRecord] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        rows.append(
            IndexRecord(
                point_id=chunk.chunk_id,
                text=text,
                payload={
                    "record_type": "chunk",
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_ref": chunk.source_ref,
                    "source_type": chunk.source_type,
                    "text": text,
                    "metadata": {**chunk.metadata, "record_type": "chunk"},
                },
            )
        )
    return rows


def _build_qa_index_records(
    qa_rows: list[QARecord],
    qa_text_mode: str,
    max_qa_answer_chars: int,
) -> list[IndexRecord]:
    rows: list[IndexRecord] = []
    for qa in qa_rows:
        text = _build_qa_text(qa, qa_text_mode, max_qa_answer_chars).strip()
        if not text:
            continue

        qa_metadata = _compact_qa_metadata(qa.metadata)
        rows.append(
            IndexRecord(
                point_id=qa.qa_id,
                text=text,
                payload={
                    "record_type": "qa_pair",
                    "chunk_id": f"qa::{qa.qa_id}",
                    "doc_id": qa.doc_id,
                    "source_ref": qa.source_ref,
                    "source_type": qa.source_type,
                    "text": text,
                    "metadata": {
                        **qa_metadata,
                        "record_type": "qa_pair",
                        "qa_id": qa.qa_id,
                        "qa_question": qa.question,
                        "linked_chunk_id": qa.chunk_id,
                        "qa_answer_chars": len(qa.answer),
                    },
                },
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = read_yaml(args.config)

    chunks_path = str(cfg.get("chunks_path", "data/staging/chunks/chunks.jsonl"))
    include_qa = bool(cfg.get("include_qa", True))
    qa_path = str(cfg.get("qa_path", "data/qa/qa_clean.jsonl"))
    qa_text_mode = str(cfg.get("qa_text_mode", "question_answer")).strip().lower()
    max_qa_answer_chars = int(cfg.get("max_qa_answer_chars", 700))
    qdrant_path = str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = str(cfg.get("collection_name", "docs"))
    embedding_model = str(cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
    batch_size = int(args.batch_size or cfg.get("index_batch_size", 128))
    if batch_size <= 0:
        raise ValueError("index batch size must be > 0")
    if max_qa_answer_chars < 0:
        raise ValueError("max_qa_answer_chars must be >= 0")

    chunks = _load_chunks(chunks_path)
    chunk_records = _build_chunk_index_records(chunks)

    qa_rows: list[QARecord] = []
    if include_qa:
        qa_file = Path(qa_path)
        if qa_file.exists():
            qa_rows = _load_qa(qa_path)
        else:
            logger.warning(
                "QA indexing enabled but QA file was not found at %s. Continuing with chunks only.",
                qa_path,
            )

    qa_records = _build_qa_index_records(
        qa_rows=qa_rows,
        qa_text_mode=qa_text_mode,
        max_qa_answer_chars=max_qa_answer_chars,
    )

    records = [*chunk_records, *qa_records]
    if not records:
        raise RuntimeError(
            f"No indexable records found. chunks_path={chunks_path}, qa_path={qa_path}"
        )

    embedder = SentenceTransformer(embedding_model)
    sample_vector = embedder.encode(
        [records[0].text], normalize_embeddings=normalize_embeddings
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

    for start in tqdm(range(0, len(records), batch_size), desc="Indexing"):
        batch = records[start : start + batch_size]
        texts = [item.text for item in batch]
        vectors = embedder.encode(texts, normalize_embeddings=normalize_embeddings)

        points: list[PointStruct] = []
        for item, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=create_uuid_from_string(item.point_id),
                    vector=vector.tolist(),
                    payload=item.payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    chunk_count = len(chunk_records)
    qa_count = len(qa_records)
    logger.info(
        "Indexed %s records (%s chunks, %s QA pairs) into collection '%s' at %s (batch_size=%s)",
        len(records),
        chunk_count,
        qa_count,
        collection_name,
        qdrant_path,
        batch_size,
    )


if __name__ == "__main__":
    main()
