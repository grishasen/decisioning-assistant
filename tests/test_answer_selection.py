from rag.answer_selection import AnswerSelectionConfig, rerank_answer_candidates


class DummyEmbedder:
    """Provide deterministic embeddings for answer-selection tests."""
    def encode(self, texts, normalize_embeddings=False):
        """Signature: def encode(self, texts, normalize_embeddings = False).

        Return deterministic vectors for the answer-selection tests.
        """
        vectors = []
        for text in texts:
            text_value = str(text)
            if "good answer" in text_value:
                vectors.append([1.0, 0.0])
            elif "bad answer" in text_value:
                vectors.append([0.0, 1.0])
            elif "Question:" in text_value and "good context" in text_value:
                vectors.append([1.0, 0.0])
            elif "Question:" in text_value and "bad context" in text_value:
                vectors.append([0.0, 1.0])
            elif "Which answer" in text_value:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors


def test_rerank_answer_candidates_prefers_supported_answer() -> None:
    """Signature: def test_rerank_answer_candidates_prefers_supported_answer() -> None.

    Verify that rerank answer candidates prefers supported answer.
    """
    ranked = rerank_answer_candidates(
        question="Which answer is grounded?",
        candidates=["good answer", "bad answer"],
        context_rows=[
            {"text": "good context"},
            {"text": "bad context"},
        ],
        config=AnswerSelectionConfig(
            sample_count=2,
            rerank_mode="embedding_cosine",
            rerank_alpha=0.7,
            support_top_k=2,
        ),
        embedder=DummyEmbedder(),
        normalize_embeddings=True,
        cross_encoder=None,
    )

    assert ranked[0]["answer"] == "good answer"
    assert ranked[0]["score"] >= ranked[1]["score"]
