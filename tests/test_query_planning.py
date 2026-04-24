from rag.query_planning import (
    QueryRewriteConfig,
    clean_rewritten_query,
    rewrite_retrieval_query,
)


class DummyGenerator:
    """Return a fixed rewrite for query planning tests."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    def generate(self, **kwargs):
        """Return the configured rewrite."""
        self.calls += 1
        return self.value


def test_clean_rewritten_query_removes_label() -> None:
    """Verify model labels are removed from rewritten retrieval queries."""
    assert (
        clean_rewritten_query("Standalone retrieval query: NBA Designer setup", "setup")
        == "NBA Designer setup"
    )


def test_rewrite_retrieval_query_uses_history_when_enabled() -> None:
    """Verify enabled query rewriting calls the generator."""
    generator = DummyGenerator("Pega CDH Paid Media Manager limits")

    query = rewrite_retrieval_query(
        generator,
        question="What about its limits?",
        history="USER: Tell me about Paid Media Manager.",
        config=QueryRewriteConfig(enabled=True),
    )

    assert query == "Pega CDH Paid Media Manager limits"
    assert generator.calls == 1


def test_rewrite_retrieval_query_skips_without_history() -> None:
    """Verify rewrite does not run when there is no prior history."""
    generator = DummyGenerator("rewritten")

    query = rewrite_retrieval_query(
        generator,
        question="What are the limits?",
        history="",
        config=QueryRewriteConfig(enabled=True),
    )

    assert query == "What are the limits?"
    assert generator.calls == 0
