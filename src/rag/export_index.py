from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

from common.io_utils import read_yaml, write_json
from common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local Qdrant RAG collection into portable files."
    )
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument(
        "--output-dir",
        default="data/rag/export",
        help="Directory to write export bundle (metadata + points JSONL).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of points to read per scroll batch.",
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        choices=("pdf", "webex"),
        default=[],
        help="Optional source-type filter. Repeat for multiple values (e.g. --source pdf --source webex).",
    )
    return parser.parse_args()


def _serialize_vectors_config(vectors: Any) -> dict[str, Any]:
    if isinstance(vectors, VectorParams):
        distance = vectors.distance.value if hasattr(vectors.distance, "value") else str(vectors.distance)
        return {
            "kind": "single",
            "size": int(vectors.size),
            "distance": str(distance),
        }

    if isinstance(vectors, dict):
        named: dict[str, dict[str, Any]] = {}
        for name, cfg in vectors.items():
            if not isinstance(cfg, VectorParams):
                raise ValueError(f"Unsupported named vector config for '{name}': {type(cfg)!r}")
            distance = cfg.distance.value if hasattr(cfg.distance, "value") else str(cfg.distance)
            named[name] = {
                "size": int(cfg.size),
                "distance": str(distance),
            }
        return {
            "kind": "named",
            "vectors": named,
        }

    raise ValueError(f"Unsupported vectors config type: {type(vectors)!r}")


def _json_safe_id(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_sources(sources: list[str]) -> set[str]:
    return {value.strip().lower() for value in sources if value.strip()}


def _payload_source_type(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    source_type = payload.get("source_type")
    if isinstance(source_type, str) and source_type.strip():
        return source_type.strip().lower()

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        meta_source = metadata.get("source_type")
        if isinstance(meta_source, str) and meta_source.strip():
            return meta_source.strip().lower()

    return ""


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)

    qdrant_path = str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = str(cfg.get("collection_name", "docs"))
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be > 0")

    selected_sources = _normalize_sources(args.sources)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    points_path = output_dir / "points.jsonl"

    client = QdrantClient(path=qdrant_path)
    if not client.collection_exists(collection_name=collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist at qdrant_path='{qdrant_path}'"
        )

    collection_info = client.get_collection(collection_name=collection_name)
    vectors_config = _serialize_vectors_config(collection_info.config.params.vectors)

    exported_points = 0
    scanned_points = 0
    filtered_out_points = 0
    offset: Any = None

    with points_path.open("w", encoding="utf-8") as handle:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            for point in points:
                scanned_points += 1
                payload = point.payload or {}
                if selected_sources:
                    source_type = _payload_source_type(payload)
                    if source_type not in selected_sources:
                        filtered_out_points += 1
                        continue

                row = {
                    "id": _json_safe_id(point.id),
                    "vector": point.vector,
                    "payload": payload,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                exported_points += 1

            if next_offset is None:
                break
            offset = next_offset

    metadata = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "vectors_config": vectors_config,
        "source_filter": sorted(selected_sources),
        "points_file": points_path.name,
        "points_count": exported_points,
        "scanned_points": scanned_points,
        "filtered_out_points": filtered_out_points,
    }
    write_json(metadata_path, metadata)

    filter_text = ",".join(sorted(selected_sources)) if selected_sources else "all"
    logger.info(
        "Exported %s points from collection '%s' into %s (scanned=%s, filtered_out=%s, source_filter=%s)",
        exported_points,
        collection_name,
        output_dir,
        scanned_points,
        filtered_out_points,
        filter_text,
    )


if __name__ == "__main__":
    main()
