from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml, write_jsonl
from common.logging_utils import get_logger
from common.schemas import ChunkRecord, DocumentRecord
from common.text_utils import normalize_whitespace, split_text, stable_id

logger = get_logger(__name__)


def _load_docs(paths: list[str]) -> list[DocumentRecord]:
    docs: list[DocumentRecord] = []
    for path in paths:
        source = Path(path)
        if not source.exists():
            logger.warning("Document file missing, skipping: %s", source)
            continue
        for row in iter_jsonl(source):
            try:
                docs.append(DocumentRecord.model_validate(row))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid row in %s: %s", source, exc)
    return docs


def _chunk_docs(
    docs: list[DocumentRecord],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in tqdm(docs, desc="Chunking docs"):
        base_text = doc.markdown if doc.markdown and doc.markdown.strip() else doc.text
        text = normalize_whitespace(base_text)
        if not text:
            continue

        for idx, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap)):
            if len(chunk_text) < min_chunk_chars:
                continue
            chunk_id = stable_id(doc.doc_id, str(idx), chunk_text[:120])
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                source_ref=doc.source_ref,
                text=chunk_text,
                metadata={
                    "chunk_index": idx,
                    "title": doc.title,
                    "page": doc.page,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine + chunk documents into canonical docs/chunks JSONL.")
    parser.add_argument("--config", default="configs/sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    normalize_cfg: dict[str, Any] = config.get("normalize", {})

    inputs = normalize_cfg.get("inputs", [])
    if not isinstance(inputs, list):
        raise ValueError("normalize.inputs must be a list")

    output_documents = normalize_cfg.get("output_documents", "data/staging/documents/documents.jsonl")
    output_chunks = normalize_cfg.get("output_chunks", "data/staging/chunks/chunks.jsonl")
    chunk_size = int(normalize_cfg.get("chunk_size", 900))
    chunk_overlap = int(normalize_cfg.get("chunk_overlap", 150))
    min_chunk_chars = int(normalize_cfg.get("min_chunk_chars", 180))

    docs = _load_docs(inputs)
    doc_rows = [doc.model_dump(mode="json") for doc in docs]
    write_jsonl(output_documents, doc_rows)
    logger.info("Wrote %s normalized docs to %s", len(doc_rows), output_documents)

    chunks = _chunk_docs(docs, chunk_size, chunk_overlap, min_chunk_chars)
    chunk_rows = [chunk.model_dump(mode="json") for chunk in chunks]
    write_jsonl(output_chunks, chunk_rows)
    logger.info("Wrote %s chunks to %s", len(chunk_rows), output_chunks)


if __name__ == "__main__":
    main()
