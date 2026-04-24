from qdrant_client import QdrantClient, models

from rag.retrieve import _collection_vector_capabilities
from rag.sparse import SparseTextConfig, sparse_indices_values, sparse_vector_from_text


def test_sparse_indices_are_stable_and_include_split_terms() -> None:
    """Verify sparse vectorization is deterministic for exact and split terms."""
    config = SparseTextConfig(max_terms=128, min_token_chars=2)

    first = sparse_indices_values("Next-Best-Action action", config)
    second = sparse_indices_values("next-best-action ACTION", config)

    assert first == second
    assert len(first[0]) >= 4
    assert all(value > 0 for value in first[1])


def test_collection_vector_capabilities_detects_dense_only_fallback(tmp_path) -> None:
    """Verify dense-only collections do not activate hybrid retrieval."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name="dense_docs",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )

    dense_name, supports_hybrid = _collection_vector_capabilities(
        client,
        "dense_docs",
        "dense",
        "sparse",
    )

    assert dense_name is None
    assert supports_hybrid is False
    client.close()


def test_collection_vector_capabilities_detects_hybrid_collection(tmp_path) -> None:
    """Verify named dense+sparse collections activate hybrid retrieval."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name="hybrid_docs",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    dense_name, supports_hybrid = _collection_vector_capabilities(
        client,
        "hybrid_docs",
        "dense",
        "sparse",
    )

    assert dense_name == "dense"
    assert supports_hybrid is True
    client.close()


def test_qdrant_hybrid_rrf_prefers_sparse_match(tmp_path) -> None:
    """Verify local Qdrant can fuse dense and sparse named vectors."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    collection_name = "docs"
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    sparse_config = SparseTextConfig(max_terms=128, min_token_chars=2)
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=1,
                vector={
                    "dense": [0.0, 1.0],
                    "sparse": sparse_vector_from_text("unrelated account setup", sparse_config),
                },
                payload={"source_ref": "dense-hit"},
            ),
            models.PointStruct(
                id=2,
                vector={
                    "dense": [1.0, 0.0],
                    "sparse": sparse_vector_from_text(
                        "next-best-action arbitration", sparse_config
                    ),
                },
                payload={"source_ref": "sparse-hit"},
            ),
        ],
    )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=[0.0, 1.0], using="dense", limit=2),
            models.Prefetch(
                query=sparse_vector_from_text("next best action", sparse_config),
                using="sparse",
                limit=2,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[0.1, 3.0])),
        limit=1,
        with_payload=True,
    )

    assert response.points[0].payload["source_ref"] == "sparse-hit"
    client.close()
