"""Build the local Qdrant index used by the DecisioningAssistant RAG flow.

This module now supports contextualized retrieval text at index time. The core
idea is to preserve the original chunk/QA text for prompting and UI display,
while storing a second compact `retrieval_text` payload that prepends metadata
such as document title, section path, page range, room title, or thread topic.

That gives embeddings and rerankers more context without making the retrieved
source text noisier for the answer model or the user-facing popups.
"""

from __future__ import annotations

import argparse
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml
from common.logging_utils import get_logger
from common.schemas import ChunkRecord, QARecord
from common.text_utils import normalize_whitespace
from rag.sparse import SparseTextConfig, sparse_vector_from_text

logger = get_logger(__name__)


@dataclass(frozen=True)
class ContextualizationConfig:
    """Store contextualized indexing settings for retrieval text."""
    enabled: bool
    max_prefix_chars: int
    contextualize_pdf_chunks: bool
    contextualize_webex_chunks: bool
    contextualize_qa_pairs: bool


@dataclass(frozen=True)
class HybridIndexConfig:
    """Store dense+sparse hybrid index settings."""

    enabled: bool
    dense_vector_name: str
    sparse_vector_name: str
    sparse_modifier: str
    sparse_max_terms: int
    sparse_min_token_chars: int


@dataclass(frozen=True)
class IndexRecord:
    """Represent one vector payload ready for Qdrant indexing."""
    point_id: str
    text: str
    payload: dict[str, Any]


def create_uuid_from_string(val: str) -> uuid.UUID:
    """Signature: def create_uuid_from_string(val: str) -> uuid.UUID.

    Create uuid from string.
    """
    hex_string = hashlib.md5(val.encode("utf-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for build index.
    """
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
    """Signature: def _load_chunks(path: str) -> list[ChunkRecord].

    Load chunks.
    """
    return [ChunkRecord.model_validate(row) for row in iter_jsonl(path)]


def _load_qa(path: str) -> list[QARecord]:
    """Signature: def _load_qa(path: str) -> list[QARecord].

    Load qa.
    """
    return [QARecord.model_validate(row) for row in iter_jsonl(path)]


def _build_qa_text(qa: QARecord, text_mode: str, max_answer_chars: int) -> str:
    """Signature: def _build_qa_text(qa: QARecord, text_mode: str, max_answer_chars: int) -> str.

    Build qa text.
    """
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
    """Signature: def _compact_qa_metadata(metadata: dict[str, Any]) -> dict[str, Any].

    Build a compact representation of qa metadata.
    """
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


def _clean_metadata_text(value: Any) -> str:
    """Signature: def _clean_metadata_text(value: Any) -> str.

    Clean metadata text.
    """
    if value is None:
        return ""
    return normalize_whitespace(str(value))


def _page_span(metadata: dict[str, Any]) -> str:
    """Signature: def _page_span(metadata: dict[str, Any]) -> str.

    Page span.
    """
    page_start = metadata.get("page_start") or metadata.get("page")
    page_end = metadata.get("page_end")
    if page_start is None:
        return ""
    if page_end is None or str(page_end) == str(page_start):
        return str(page_start)
    return f"{page_start}-{page_end}"


def _clip_prefix(prefix: str, max_chars: int) -> str:
    """Signature: def _clip_prefix(prefix: str, max_chars: int) -> str.

    Clip prefix.
    """
    cleaned = normalize_whitespace(prefix)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _contextualize_text(
    base_text: str,
    prefix_lines: list[str],
    max_prefix_chars: int,
    *,
    body_label: str,
) -> str:
    """Signature: def _contextualize_text(base_text: str, prefix_lines: list[str], max_prefix_chars: int, *, body_label: str) -> str.

    Contextualize text.
    """
    cleaned_text = str(base_text or "").strip()
    if not cleaned_text:
        return ""

    prefix = _clip_prefix("\n".join(line for line in prefix_lines if line), max_prefix_chars)
    if not prefix:
        return cleaned_text
    return f"{prefix}\n\n{body_label}:\n{cleaned_text}"


def _build_chunk_prefix_lines(
    chunk: ChunkRecord,
    cfg: ContextualizationConfig,
) -> list[str]:
    """Signature: def _build_chunk_prefix_lines(chunk: ChunkRecord, cfg: ContextualizationConfig) -> list[str].

    Build chunk prefix lines.
    """
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    source_type = str(chunk.source_type or "").strip().lower()

    if source_type == "pdf" and cfg.contextualize_pdf_chunks:
        pages = _page_span(metadata)
        return [
            f"Document: {_clean_metadata_text(metadata.get('pdf_title') or metadata.get('title') or metadata.get('file_name'))}",
            f"Product: {_clean_metadata_text(metadata.get('product'))}",
            f"Version: {_clean_metadata_text(metadata.get('doc_version'))}",
            f"Type: {_clean_metadata_text(metadata.get('doc_type'))}",
            f"Section: {_clean_metadata_text(metadata.get('section_path') or metadata.get('section_title') or metadata.get('active_heading'))}",
            f"Pages: {pages}" if pages else "",
        ]

    if source_type == "webex" and cfg.contextualize_webex_chunks:
        updated = _clean_metadata_text(
            metadata.get("updated_at") or metadata.get("created_at")
        )
        return [
            "Source: Webex discussion",
            f"Room: {_clean_metadata_text(metadata.get('room_title') or metadata.get('title'))}",
            f"Thread: {_clean_metadata_text(metadata.get('thread_id'))}",
            f"Messages: {_clean_metadata_text(metadata.get('message_count'))}",
            f"Updated: {updated}" if updated else "",
            f"Topic: {_clean_metadata_text(metadata.get('thread_start_text'))}",
        ]

    return []


def _build_chunk_retrieval_text(
    chunk: ChunkRecord,
    cfg: ContextualizationConfig,
) -> str:
    """Signature: def _build_chunk_retrieval_text(chunk: ChunkRecord, cfg: ContextualizationConfig) -> str.

    Build metadata-enriched retrieval text for a chunk record.
    """
    base_text = str(chunk.text or "").strip()
    if not cfg.enabled:
        return base_text
    prefix_lines = _build_chunk_prefix_lines(chunk, cfg)
    return _contextualize_text(
        base_text,
        prefix_lines,
        cfg.max_prefix_chars,
        body_label="Chunk",
    )


def _build_qa_prefix_lines(
    qa: QARecord,
    qa_metadata: dict[str, Any],
    cfg: ContextualizationConfig,
) -> list[str]:
    """Signature: def _build_qa_prefix_lines(qa: QARecord, qa_metadata: dict[str, Any], cfg: ContextualizationConfig) -> list[str].

    Build qa prefix lines.
    """
    if not cfg.contextualize_qa_pairs:
        return []

    source_type = str(qa.source_type or "").strip().lower()
    lines = [
        "Source: Synthetic QA pair",
        f"Question: {_clean_metadata_text(qa.question)}",
    ]

    if source_type == "pdf":
        pages = _page_span(qa_metadata)
        lines.extend(
            [
                f"Document: {_clean_metadata_text(qa_metadata.get('pdf_title') or qa_metadata.get('title'))}",
                f"Product: {_clean_metadata_text(qa_metadata.get('product'))}",
                f"Version: {_clean_metadata_text(qa_metadata.get('doc_version'))}",
                f"Type: {_clean_metadata_text(qa_metadata.get('doc_type'))}",
                f"Section: {_clean_metadata_text(qa_metadata.get('section_path') or qa_metadata.get('section_title') or qa_metadata.get('active_heading'))}",
                f"Pages: {pages}" if pages else "",
            ]
        )
    elif source_type == "webex":
        updated = _clean_metadata_text(
            qa_metadata.get("updated_at") or qa_metadata.get("created_at")
        )
        lines.extend(
            [
                f"Room: {_clean_metadata_text(qa_metadata.get('room_title') or qa_metadata.get('title'))}",
                f"Thread: {_clean_metadata_text(qa_metadata.get('thread_id'))}",
                f"Messages: {_clean_metadata_text(qa_metadata.get('message_count'))}",
                f"Updated: {updated}" if updated else "",
            ]
        )

    return lines


def _build_qa_retrieval_text(
    qa: QARecord,
    qa_metadata: dict[str, Any],
    text_mode: str,
    max_answer_chars: int,
    cfg: ContextualizationConfig,
) -> str:
    """Signature: def _build_qa_retrieval_text(qa: QARecord, qa_metadata: dict[str, Any], text_mode: str, max_answer_chars: int, cfg: ContextualizationConfig) -> str.

    Build metadata-enriched retrieval text for a QA record.
    """
    base_text = _build_qa_text(qa, text_mode, max_answer_chars).strip()
    if not cfg.enabled:
        return base_text
    prefix_lines = _build_qa_prefix_lines(qa, qa_metadata, cfg)
    return _contextualize_text(
        base_text,
        prefix_lines,
        cfg.max_prefix_chars,
        body_label="QA Pair",
    )


def _build_chunk_index_records(
    chunks: list[ChunkRecord],
    contextualization_cfg: ContextualizationConfig,
) -> list[IndexRecord]:
    """Signature: def _build_chunk_index_records(chunks: list[ChunkRecord], contextualization_cfg: ContextualizationConfig) -> list[IndexRecord].

    Build chunk index records.
    """
    rows: list[IndexRecord] = []
    for chunk in chunks:
        raw_text = chunk.text.strip()
        if not raw_text:
            continue

        retrieval_text = _build_chunk_retrieval_text(chunk, contextualization_cfg).strip()
        if not retrieval_text:
            continue

        rows.append(
            IndexRecord(
                point_id=chunk.chunk_id,
                text=retrieval_text,
                payload={
                    "record_type": "chunk",
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_ref": chunk.source_ref,
                    "source_type": chunk.source_type,
                    # Raw text remains the source of truth for prompt construction,
                    # source popups, and export/import bundles.
                    "text": raw_text,
                    # retrieval_text is enriched with compact metadata context so
                    # embedding-based retrieval and rerankers can disambiguate short chunks.
                    "retrieval_text": retrieval_text,
                    "metadata": {**chunk.metadata, "record_type": "chunk"},
                },
            )
        )
    return rows


def _build_qa_index_records(
    qa_rows: list[QARecord],
    qa_text_mode: str,
    max_qa_answer_chars: int,
    contextualization_cfg: ContextualizationConfig,
) -> list[IndexRecord]:
    """Signature: def _build_qa_index_records(qa_rows: list[QARecord], qa_text_mode: str, max_qa_answer_chars: int, contextualization_cfg: ContextualizationConfig) -> list[IndexRecord].

    Build qa index records.
    """
    rows: list[IndexRecord] = []
    for qa in qa_rows:
        raw_text = _build_qa_text(qa, qa_text_mode, max_qa_answer_chars).strip()
        if not raw_text:
            continue

        qa_metadata = _compact_qa_metadata(qa.metadata)
        retrieval_text = _build_qa_retrieval_text(
            qa=qa,
            qa_metadata=qa_metadata,
            text_mode=qa_text_mode,
            max_answer_chars=max_qa_answer_chars,
            cfg=contextualization_cfg,
        ).strip()
        if not retrieval_text:
            continue

        rows.append(
            IndexRecord(
                point_id=qa.qa_id,
                text=retrieval_text,
                payload={
                    "record_type": "qa_pair",
                    "chunk_id": f"qa::{qa.qa_id}",
                    "doc_id": qa.doc_id,
                    "source_ref": qa.source_ref,
                    "source_type": qa.source_type,
                    "text": raw_text,
                    "retrieval_text": retrieval_text,
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


def _sparse_modifier(value: str) -> Modifier | None:
    """Return the configured Qdrant sparse-vector modifier."""
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    if normalized == "idf":
        return Modifier.IDF
    raise ValueError("sparse_modifier must be one of: idf, none")


def _vectors_config(
    *,
    vector_size: int,
    hybrid_cfg: HybridIndexConfig,
) -> tuple[Any, dict[str, SparseVectorParams] | None]:
    """Build Qdrant dense and sparse collection configs."""
    dense_params = VectorParams(size=vector_size, distance=Distance.COSINE)
    if not hybrid_cfg.enabled:
        return dense_params, None

    return (
        {hybrid_cfg.dense_vector_name: dense_params},
        {
            hybrid_cfg.sparse_vector_name: SparseVectorParams(
                index=SparseIndexParams(),
                modifier=_sparse_modifier(hybrid_cfg.sparse_modifier),
            )
        },
    )


def _collection_matches_index_shape(
    client: QdrantClient,
    collection_name: str,
    *,
    vector_size: int,
    hybrid_cfg: HybridIndexConfig,
) -> bool:
    """Return whether an existing collection can accept the configured point shape."""
    info = client.get_collection(collection_name=collection_name)
    params = info.config.params
    vectors = params.vectors
    sparse_vectors = params.sparse_vectors or {}

    if hybrid_cfg.enabled:
        if not isinstance(vectors, dict):
            return False
        dense_cfg = vectors.get(hybrid_cfg.dense_vector_name)
        if dense_cfg is None or int(dense_cfg.size) != vector_size:
            return False
        return hybrid_cfg.sparse_vector_name in sparse_vectors

    if isinstance(vectors, VectorParams):
        return int(vectors.size) == vector_size
    return False


def _point_vector(
    dense_vector: Any,
    retrieval_text: str,
    *,
    hybrid_cfg: HybridIndexConfig,
    sparse_cfg: SparseTextConfig,
) -> Any:
    """Build the Qdrant point vector payload for dense-only or hybrid collections."""
    dense_values = dense_vector.tolist()
    if not hybrid_cfg.enabled:
        return dense_values

    return {
        hybrid_cfg.dense_vector_name: dense_values,
        hybrid_cfg.sparse_vector_name: sparse_vector_from_text(retrieval_text, sparse_cfg),
    }


def main() -> None:
    """Signature: def main() -> None.

    Build or update the local Qdrant index from chunk and QA records.
    """
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
    contextualization_cfg = ContextualizationConfig(
        enabled=bool(cfg.get("contextualize_index_text", True)),
        max_prefix_chars=max(0, int(cfg.get("contextual_index_max_prefix_chars", 320))),
        contextualize_pdf_chunks=bool(cfg.get("contextualize_pdf_chunks", True)),
        contextualize_webex_chunks=bool(cfg.get("contextualize_webex_chunks", True)),
        contextualize_qa_pairs=bool(cfg.get("contextualize_qa_pairs", True)),
    )
    hybrid_cfg = HybridIndexConfig(
        enabled=bool(cfg.get("hybrid_search_enabled", False)),
        dense_vector_name=str(cfg.get("dense_vector_name", "dense")).strip() or "dense",
        sparse_vector_name=str(cfg.get("sparse_vector_name", "sparse")).strip() or "sparse",
        sparse_modifier=str(cfg.get("sparse_modifier", "idf")),
        sparse_max_terms=max(0, int(cfg.get("sparse_max_terms", 512))),
        sparse_min_token_chars=max(1, int(cfg.get("sparse_min_token_chars", 2))),
    )
    sparse_cfg = SparseTextConfig(
        max_terms=hybrid_cfg.sparse_max_terms,
        min_token_chars=hybrid_cfg.sparse_min_token_chars,
    )
    if batch_size <= 0:
        raise ValueError("index batch size must be > 0")
    if max_qa_answer_chars < 0:
        raise ValueError("max_qa_answer_chars must be >= 0")

    chunks = _load_chunks(chunks_path)
    chunk_records = _build_chunk_index_records(chunks, contextualization_cfg)

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
        contextualization_cfg=contextualization_cfg,
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
    vectors_config, sparse_vectors_config = _vectors_config(
        vector_size=vector_size,
        hybrid_cfg=hybrid_cfg,
    )
    if args.recreate:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
    else:
        collections = {c.name for c in client.get_collections().collections}
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        elif not _collection_matches_index_shape(
            client,
            collection_name,
            vector_size=vector_size,
            hybrid_cfg=hybrid_cfg,
        ):
            raise RuntimeError(
                f"Collection '{collection_name}' does not match the configured "
                "dense/sparse vector shape. Rebuild with --recreate or use a "
                "different collection_name/qdrant_path."
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
                    vector=_point_vector(
                        vector,
                        item.text,
                        hybrid_cfg=hybrid_cfg,
                        sparse_cfg=sparse_cfg,
                    ),
                    payload=item.payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    logger.info(
        "Indexed %s records into %s at %s "
        "(contextualize_index_text=%s, hybrid_search_enabled=%s)",
        len(records),
        collection_name,
        qdrant_path,
        contextualization_cfg.enabled,
        hybrid_cfg.enabled,
    )


if __name__ == "__main__":
    main()
