from common.io_utils import iter_jsonl
from rag.feedback import append_feedback, build_feedback_row, source_summary


def test_source_summary_keeps_retrieval_scores() -> None:
    """Verify feedback source summaries keep ranking details."""
    row = {
        "source_ref": "pdf::guide#page=1",
        "chunk_id": "chunk-1",
        "score": 0.8,
        "qdrant_score": 0.7,
        "rerank_score": 0.9,
        "metadata": {"pdf_title": "Guide", "section_title": "Setup"},
    }

    summary = source_summary(row)

    assert summary["source_ref"] == "pdf::guide#page=1"
    assert summary["qdrant_score"] == 0.7
    assert summary["rerank_score"] == 0.9
    assert summary["title"] == "Guide"
    assert summary["section"] == "Setup"


def test_append_feedback_writes_jsonl(tmp_path) -> None:
    """Verify feedback rows are appended as JSONL."""
    path = tmp_path / "feedback.jsonl"
    row = build_feedback_row(
        message_id="msg-1",
        rating="needs_work",
        feedback_text="Wrong source",
        question="What changed?",
        answer="I do not know.",
        retrieval_query="What changed in Paid Media Manager?",
        sources=[{"source_ref": "webex::room#thread=1", "score": 0.4}],
        answer_time_seconds=1.25,
    )

    append_feedback(str(path), row)

    loaded = list(iter_jsonl(path))
    assert len(loaded) == 1
    assert loaded[0]["message_id"] == "msg-1"
    assert loaded[0]["rating"] == "needs_work"
    assert loaded[0]["selected_sources"][0]["source_ref"] == "webex::room#thread=1"
