from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Datatype,
    Distance,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from common.io_utils import iter_jsonl, read_json, read_yaml
from common.logging_utils import get_logger

logger = get_logger(__name__)


_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}

_MODIFIER_MAP = {
    "idf": Modifier.IDF,
    "none": Modifier.NONE,
}

_DATATYPE_MAP = {
    "float32": Datatype.FLOAT32,
    "float16": Datatype.FLOAT16,
    "uint8": Datatype.UINT8,
}


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for import index.
    """
    parser = argparse.ArgumentParser(
        description="Import portable Qdrant RAG bundle into local Qdrant collection."
    )
    parser.add_argument(
        "--input-dir",
        default="data/rag/export",
        help="Directory containing metadata.json and points.jsonl.",
    )
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument(
        "--qdrant-path",
        default="",
        help="Override destination qdrant path (defaults to rag.yaml qdrant_path).",
    )
    parser.add_argument(
        "--collection-name",
        default="",
        help="Override destination collection name (defaults to export metadata or rag.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of points per upsert batch.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop destination collection before import.",
    )
    return parser.parse_args()


def _distance_from_value(raw: Any) -> Distance:
    """Signature: def _distance_from_value(raw: Any) -> Distance.

    Distance from value.
    """
    text = str(raw or "").strip().lower()
    mapped = _DISTANCE_MAP.get(text)
    if mapped:
        return mapped

    # Handle enum names/values like "Cosine".
    mapped = _DISTANCE_MAP.get(text.replace("_", ""))
    if mapped:
        return mapped

    raise ValueError(f"Unsupported distance value in export metadata: {raw!r}")


def _build_vectors_config(spec: dict[str, Any]) -> Any:
    """Signature: def _build_vectors_config(spec: dict[str, Any]) -> Any.

    Build vectors config.
    """
    kind = str(spec.get("kind", "single")).strip().lower()

    if kind == "single":
        size = int(spec["size"])
        distance = _distance_from_value(spec["distance"])
        return VectorParams(size=size, distance=distance)

    if kind == "named":
        raw_vectors = spec.get("vectors")
        if not isinstance(raw_vectors, dict) or not raw_vectors:
            raise ValueError("Invalid named vectors config in export metadata")

        named: dict[str, VectorParams] = {}
        for name, cfg in raw_vectors.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Invalid vector params for '{name}'")
            named[name] = VectorParams(
                size=int(cfg["size"]),
                distance=_distance_from_value(cfg["distance"]),
            )
        return named

    raise ValueError(f"Unsupported vectors config kind: {kind!r}")


def _modifier_from_value(raw: Any) -> Modifier | None:
    """Return a Qdrant sparse-vector modifier from export metadata."""
    if raw is None or raw == "":
        return None
    text = str(raw).strip().lower()
    mapped = _MODIFIER_MAP.get(text)
    if mapped is not None:
        return mapped
    raise ValueError(f"Unsupported sparse modifier value in export metadata: {raw!r}")


def _datatype_from_value(raw: Any) -> Datatype | None:
    """Return a Qdrant datatype enum from export metadata."""
    if raw is None or raw == "":
        return None
    text = str(raw).strip().lower()
    mapped = _DATATYPE_MAP.get(text)
    if mapped is not None:
        return mapped
    raise ValueError(f"Unsupported sparse datatype value in export metadata: {raw!r}")


def _build_sparse_vectors_config(spec: Any) -> dict[str, SparseVectorParams] | None:
    """Build Qdrant sparse vector config from export metadata."""
    if not spec:
        return None
    if not isinstance(spec, dict):
        raise ValueError("metadata.sparse_vectors_config must be a mapping when present")

    sparse_vectors: dict[str, SparseVectorParams] = {}
    for name, cfg in spec.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid sparse vector params for '{name}'")
        index_raw = cfg.get("index")
        index = None
        if isinstance(index_raw, dict):
            index = SparseIndexParams(
                full_scan_threshold=index_raw.get("full_scan_threshold"),
                on_disk=index_raw.get("on_disk"),
                datatype=_datatype_from_value(index_raw.get("datatype")),
            )
        sparse_vectors[str(name)] = SparseVectorParams(
            index=index,
            modifier=_modifier_from_value(cfg.get("modifier")),
        )
    return sparse_vectors


def _build_point_vector(value: Any) -> Any:
    """Convert exported vector JSON into Qdrant PointStruct-compatible vectors."""
    if isinstance(value, dict) and "indices" in value and "values" in value:
        return SparseVector(indices=value["indices"], values=value["values"])
    if isinstance(value, dict):
        return {str(name): _build_point_vector(vector) for name, vector in value.items()}
    return value


def main() -> None:
    """Signature: def main() -> None.

    Run the import index entrypoint.
    """
    args = parse_args()
    cfg = read_yaml(args.config)

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata payload: {metadata_path}")

    points_file_name = str(metadata.get("points_file", "points.jsonl"))
    points_path = input_dir / points_file_name
    if not points_path.exists():
        raise FileNotFoundError(f"Missing points file: {points_path}")

    qdrant_path = args.qdrant_path.strip() or str(cfg.get("qdrant_path", "data/rag/vectordb"))
    collection_name = (
        args.collection_name.strip()
        or str(metadata.get("collection_name") or "").strip()
        or str(cfg.get("collection_name", "docs"))
    )
    if not collection_name:
        raise ValueError("collection name is required")

    vectors_spec = metadata.get("vectors_config")
    if not isinstance(vectors_spec, dict):
        raise ValueError("metadata.vectors_config must be a mapping")

    vectors_config = _build_vectors_config(vectors_spec)
    sparse_vectors_config = _build_sparse_vectors_config(
        metadata.get("sparse_vectors_config")
    )

    client = QdrantClient(path=qdrant_path)
    if args.recreate and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    collections = {c.name for c in client.get_collections().collections}
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    points_buffer: list[PointStruct] = []
    imported_points = 0

    for row in tqdm(iter_jsonl(points_path), desc="Importing points"):
        point_id = row.get("id")
        vector = row.get("vector")
        payload = row.get("payload") or {}

        if point_id is None:
            raise ValueError("Encountered point row without id")
        if vector is None:
            raise ValueError(f"Encountered point row without vector (id={point_id!r})")

        points_buffer.append(
            PointStruct(
                id=point_id,
                vector=_build_point_vector(vector),
                payload=payload,
            )
        )

        if len(points_buffer) >= batch_size:
            client.upsert(collection_name=collection_name, points=points_buffer)
            imported_points += len(points_buffer)
            points_buffer = []

    if points_buffer:
        client.upsert(collection_name=collection_name, points=points_buffer)
        imported_points += len(points_buffer)

    logger.info(
        "Imported %s points into collection '%s' at %s",
        imported_points,
        collection_name,
        qdrant_path,
    )


if __name__ == "__main__":
    main()
