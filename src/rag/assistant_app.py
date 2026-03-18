from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

from common.io_utils import read_yaml
from common.mlx_utils import MLXLoadedGenerator
from rag.answer_selection import AnswerSelectionConfig, generate_best_answer
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    clip_text_to_tokens,
    format_context,
    format_history,
    select_context_rows,
)
from rag.retrieve import LocalRetriever, resolve_answer_rerank_resources

_PAGE_REF_RE = re.compile(r"#page=(\d+)")
_SECTION_REF_RE = re.compile(r"#section=(\d+)(?:-\d+)?")


@dataclass(frozen=True)
class RetrievalConfig:
    qdrant_path: str
    collection_name: str
    embedding_model: str
    normalize_embeddings: bool
    fetch_k: int
    score_threshold: float
    rerank_mode: str
    reranker_model: str
    rerank_alpha: float
    max_per_source: int
    qa_pair_score_boost: float


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str
    max_tokens: int
    temperature: float
    adapter_path: str | None
    trust_remote_code: bool


@st.cache_resource(show_spinner=False)
def _load_retriever(cfg: RetrievalConfig) -> LocalRetriever:
    return LocalRetriever(
        qdrant_path=cfg.qdrant_path,
        collection_name=cfg.collection_name,
        embedding_model=cfg.embedding_model,
        normalize_embeddings=cfg.normalize_embeddings,
        fetch_k=cfg.fetch_k,
        score_threshold=cfg.score_threshold,
        rerank_mode=cfg.rerank_mode,
        reranker_model=cfg.reranker_model,
        rerank_alpha=cfg.rerank_alpha,
        max_per_source=cfg.max_per_source,
        qa_pair_score_boost=cfg.qa_pair_score_boost,
    )


@st.cache_resource(show_spinner=False)
def _load_generator(cfg: GenerationConfig) -> MLXLoadedGenerator:
    return MLXLoadedGenerator(
        model=cfg.model_name,
        adapter_path=cfg.adapter_path,
        trust_remote_code=cfg.trust_remote_code,
    )


def _read_configs(
    rag_config_path: str,
    models_config_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rag_cfg = read_yaml(rag_config_path)
    models_cfg = read_yaml(models_config_path)
    return rag_cfg, models_cfg


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _infer_source_type(row: dict[str, Any]) -> str:
    source_type = str(row.get("source_type") or "").strip().lower()
    if source_type:
        return source_type

    source_ref = str(row.get("source_ref") or "").strip().lower()
    if source_ref.startswith("pdf::"):
        return "pdf"
    if source_ref.startswith("webex::"):
        return "webex"
    return "unknown"


def _format_datetime(raw_value: Any) -> str:
    if raw_value is None:
        return "unknown"

    text = str(raw_value).strip()
    if not text:
        return "unknown"

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat(sep=" ", timespec="seconds")
    except ValueError:
        return text


def _extract_pdf_page(source_ref: str, metadata: dict[str, Any]) -> str:
    page_start_raw = metadata.get("page_start") or metadata.get("page")
    page_end_raw = metadata.get("page_end")

    page_start: int | None = None
    page_end: int | None = None
    if isinstance(page_start_raw, int):
        page_start = page_start_raw
    elif isinstance(page_start_raw, str) and page_start_raw.isdigit():
        page_start = int(page_start_raw)

    if isinstance(page_end_raw, int):
        page_end = page_end_raw
    elif isinstance(page_end_raw, str) and page_end_raw.isdigit():
        page_end = int(page_end_raw)

    if page_start is not None:
        if page_end is not None and page_end != page_start:
            return f"{page_start}-{page_end}"
        return str(page_start)

    page_match = _PAGE_REF_RE.search(source_ref)
    if page_match:
        return page_match.group(1)

    section_match = _SECTION_REF_RE.search(source_ref)
    if section_match:
        return section_match.group(1)

    return "unknown"


def _format_source_line(row: dict[str, Any]) -> str:
    metadata_raw = row.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    source_ref = str(row.get("source_ref") or "unknown")

    score = float(row.get("score") or 0.0)
    qdrant_score = row.get("qdrant_score")
    if isinstance(qdrant_score, (int, float)):
        score_text = f"score={score:.4f}, vec={float(qdrant_score):.4f}"
    else:
        score_text = f"score={score:.4f}"

    source_type = _infer_source_type(row)

    if source_type == "pdf":
        title = str(
            metadata.get("pdf_title")
            or metadata.get("title")
            or metadata.get("file_name")
            or "unknown"
        )
        section = str(
            metadata.get("section_title") or metadata.get("active_heading") or "unknown"
        )
        page = _extract_pdf_page(source_ref, metadata)
        return (
            f"- PDF: `{title}` | chapter: `{section}` | page: `{page}` ({score_text})"
        )

    if source_type == "webex":
        room_title = str(
            metadata.get("room_title") or metadata.get("title") or "unknown"
        )
        message_timestamp = _format_datetime(metadata.get("created_at"))
        return (
            f"- Webex Space: `{room_title}` | Timestamp: `{message_timestamp}` ({score_text})"
        )

    record_type = str(row.get("record_type") or "chunk")
    return f"- `{source_ref}` [{record_type}] ({score_text})"


def _source_popup_text(row: dict[str, Any], max_chars: int = 20000) -> str:
    value = str(row.get("text") or "").strip()
    if not value:
        return "No retrieved document text available for this source."

    if len(value) <= max_chars:
        return value

    return value[:max_chars].rstrip() + "\n\n...[truncated]"


def _webex_parent_message_link(row: dict[str, Any]) -> str:
    metadata_raw = row.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    value = metadata.get("webex_parent_message_link")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _render_source_row(row: dict[str, Any], index: int) -> None:
    left, right = st.columns([0.82, 0.18], vertical_alignment="top")

    with left:
        st.markdown(_format_source_line(row))

    with right:
        with st.popover(f"Show text", type="tertiary"):
            webex_link = _webex_parent_message_link(row)
            if webex_link:
                st.markdown(f"[Open parent message in Webex]({webex_link})")
                st.caption(webex_link)
            st.caption("Retrieved document text")
            st.text(_source_popup_text(row))


def main() -> None:
    st.set_page_config(page_title="Decisioning Assistant RAG Chat", layout="wide")
    st.title("Decisioning Assistant RAG Chat")
    st.caption("Local RAG chat using Qdrant + local MLX model")

    with st.sidebar:
        st.subheader("Settings")
        rag_config_path = st.text_input("RAG config path", value="configs/rag.yaml")
        models_config_path = st.text_input(
            "Models config path", value="configs/models.yaml"
        )

    try:
        rag_cfg, models_cfg = _read_configs(rag_config_path, models_config_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read config files: {exc}")
        st.stop()

    answer_cfg: dict[str, Any] = models_cfg.get("answer_model", {})
    with st.sidebar:
        adapter_override = st.text_input(
            "Adapter path (optional)", value=answer_cfg.get("adapter_path", "")
        )
        clear_chat = st.button("Clear Chat History 🗑️")

    model_name = str(answer_cfg.get("model"))
    max_tokens = int(answer_cfg.get("max_tokens", 256))
    temperature = float(answer_cfg.get("temperature", 0.2))
    trust_remote_code = bool(answer_cfg.get("trust_remote_code", True))

    top_k_min = int(rag_cfg.get("top_k_min", 1))
    top_k_max = int(rag_cfg.get("top_k_max", 30))
    if top_k_min > top_k_max:
        top_k_min, top_k_max = top_k_max, top_k_min

    max_ctx_min = int(rag_cfg.get("max_context_chunks_min", 1))
    max_ctx_max = int(rag_cfg.get("max_context_chunks_max", 30))
    if max_ctx_min > max_ctx_max:
        max_ctx_min, max_ctx_max = max_ctx_max, max_ctx_min

    top_k_default = _clamp(int(rag_cfg.get("top_k", 6)), top_k_min, top_k_max)
    max_ctx_default = _clamp(
        int(rag_cfg.get("max_context_chunks", 4)),
        max_ctx_min,
        max_ctx_max,
    )

    max_history_turns_default = max(0, int(rag_cfg.get("max_history_turns", 6)))
    max_history_chars_default = max(0, int(rag_cfg.get("max_history_chars", 3000)))
    max_chunk_chars_default = max(0, int(rag_cfg.get("max_chunk_chars", 1200)))
    max_total_context_chars_default = max(
        0, int(rag_cfg.get("max_total_context_chars", 5000))
    )
    max_prompt_chars_default = max(0, int(rag_cfg.get("max_prompt_chars", 12000)))

    max_history_tokens_default = max(0, int(rag_cfg.get("max_history_tokens", 0)))
    max_chunk_tokens_default = max(0, int(rag_cfg.get("max_chunk_tokens", 0)))
    max_total_context_tokens_default = max(
        0, int(rag_cfg.get("max_total_context_tokens", 0))
    )
    max_prompt_tokens_default = max(0, int(rag_cfg.get("max_prompt_tokens", 0)))

    fetch_k_default = max(
        1, int(rag_cfg.get("fetch_k", max(top_k_default * 3, top_k_default)))
    )
    score_threshold_default = max(0.0, float(rag_cfg.get("score_threshold", 0.0)))
    rerank_mode_default = str(
        rag_cfg.get("rerank_mode", "cross_encoder")
    ).strip().lower()
    reranker_model_default = str(
        rag_cfg.get("reranker_model", "BAAI/bge-reranker-base")
    )
    rerank_alpha_default = max(
        0.0, min(1.0, float(rag_cfg.get("rerank_alpha", 0.65)))
    )
    max_per_source_default = max(0, int(rag_cfg.get("max_per_source", 0)))
    qa_pair_score_boost_default = float(rag_cfg.get("qa_pair_score_boost", 0.0))

    answer_sample_count_min = max(1, int(rag_cfg.get("answer_sample_count_min", 1)))
    answer_sample_count_max = max(
        answer_sample_count_min,
        int(rag_cfg.get("answer_sample_count_max", 12)),
    )
    answer_sample_count_default = _clamp(
        int(rag_cfg.get("answer_sample_count", 4)),
        answer_sample_count_min,
        answer_sample_count_max,
    )
    answer_rerank_mode_default = str(
        rag_cfg.get("answer_rerank_mode", "cross_encoder")
    ).strip().lower()
    answer_rerank_alpha_default = max(
        0.0, min(1.0, float(rag_cfg.get("answer_rerank_alpha", 0.7)))
    )
    answer_rerank_support_top_k_default = max(
        1, int(rag_cfg.get("answer_rerank_support_top_k", 3))
    )

    retrieval_modes = ["cross_encoder", "embedding_cosine", "none"]
    if rerank_mode_default not in retrieval_modes:
        rerank_mode_default = "cross_encoder"
    if answer_rerank_mode_default not in retrieval_modes:
        answer_rerank_mode_default = "cross_encoder"

    with st.sidebar:
        top_k = int(
            st.number_input(
                "Top-k retrieval",
                min_value=top_k_min,
                max_value=top_k_max,
                value=top_k_default,
                help=(
                    "Number of results kept after reranking, filtering, and source "
                    "diversity limits. Increase for broader recall; lower for tighter, "
                    "faster retrieval."
                ),
            )
        )
        max_context_chunks = int(
            st.number_input(
                "Max context chunks",
                min_value=max_ctx_min,
                max_value=max_ctx_max,
                value=max_ctx_default,
                help=(
                    "Maximum number of retrieved rows that can be inserted into the "
                    "prompt after budget trimming. Higher values add recall but can "
                    "dilute relevance and increase latency."
                ),
            )
        )

        with st.expander("Advanced Retrieval"):
            fetch_k = int(
                st.number_input(
                    "Fetch-k candidates",
                    min_value=top_k,
                    value=max(fetch_k_default, top_k),
                    step=1,
                    help=(
                        "How many vector hits to pull from Qdrant before reranking. "
                        "Raise this when the reranker is strong or the corpus is noisy; "
                        "lower it to reduce latency."
                    ),
                )
            )
            score_threshold = float(
                st.number_input(
                    "Score threshold",
                    min_value=0.0,
                    value=score_threshold_default,
                    step=0.01,
                    format="%.3f",
                    help=(
                        "Discard raw vector hits below this Qdrant score before reranking. "
                        "Raise it to reduce noise; lower it if relevant results are being missed."
                    ),
                )
            )
            rerank_mode = st.selectbox(
                "Rerank mode",
                options=retrieval_modes,
                index=retrieval_modes.index(rerank_mode_default),
                help=(
                    "cross_encoder is the most accurate but slowest. embedding_cosine "
                    "re-embeds the retrieved text for a cheaper second pass. none keeps "
                    "the raw vector order."
                ),
            )
            reranker_model = st.text_input(
                "Reranker model",
                value=reranker_model_default,
                disabled=rerank_mode != "cross_encoder",
                help=(
                    "SentenceTransformers cross-encoder model used for pairwise reranking. "
                    "Only used when rerank mode is cross_encoder."
                ),
            )
            rerank_alpha = float(
                st.slider(
                    "Rerank alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=rerank_alpha_default,
                    step=0.05,
                    disabled=rerank_mode == "none",
                    help=(
                        "Blend between original vector score and reranker score. "
                        "0.0 uses only vector ranking; 1.0 uses only the reranker."
                    ),
                )
            )
            max_per_source = int(
                st.number_input(
                    "Max per source",
                    min_value=0,
                    value=max_per_source_default,
                    step=1,
                    help=(
                        "Limit how many rows can come from the same source bucket "
                        "(for example one PDF, one Webex thread, or one linked QA set). "
                        "Set to 0 to disable."
                    ),
                )
            )
            qa_pair_score_boost = float(
                st.number_input(
                    "QA pair score boost",
                    value=qa_pair_score_boost_default,
                    step=0.01,
                    format="%.3f",
                    help=(
                        "Small score bonus added to synthetic QA records after reranking. "
                        "Useful when QA rows should compete more strongly with raw chunks."
                    ),
                )
            )

        with st.expander("Answer Selection"):
            answer_sample_count = int(
                st.number_input(
                    "Answer candidates",
                    min_value=answer_sample_count_min,
                    max_value=answer_sample_count_max,
                    value=answer_sample_count_default,
                    help=(
                        "How many independently sampled answers to generate from the same "
                        "RAG prompt. 1 disables Best-of-N. Higher values improve robustness "
                        "but increase latency."
                    ),
                )
            )
            answer_rerank_mode = st.selectbox(
                "Answer rerank mode",
                options=retrieval_modes,
                index=retrieval_modes.index(answer_rerank_mode_default),
                help=(
                    "How to score multiple answer candidates after generation. cross_encoder "
                    "is strongest if available, embedding_cosine is lighter, and none keeps "
                    "generation order."
                ),
            )
            answer_rerank_alpha = float(
                st.slider(
                    "Answer rerank alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=answer_rerank_alpha_default,
                    step=0.05,
                    disabled=answer_rerank_mode == "none",
                    help=(
                        "Blend between question-answer relevance and support from the retrieved "
                        "context. Higher values favor grounded support over general answer fit."
                    ),
                )
            )
            answer_rerank_support_top_k = int(
                st.number_input(
                    "Answer support top-k",
                    min_value=1,
                    max_value=max(1, max_context_chunks),
                    value=min(answer_rerank_support_top_k_default, max(1, max_context_chunks)),
                    help=(
                        "How many of the top selected context rows to use when scoring how well "
                        "each answer is supported by retrieved evidence."
                    ),
                )
            )

        with st.expander("Prompt Budgets"):
            st.caption(
                "Use 0 to disable token-based limits. Character budgets still apply "
                "when set above 0."
            )
            max_history_turns = int(
                st.number_input(
                    "Max history turns",
                    min_value=0,
                    value=max_history_turns_default,
                    step=1,
                    help=(
                        "How many prior user+assistant turns to include in the prompt. "
                        "Lower this if the model starts drifting or repeats itself."
                    ),
                )
            )
            max_history_chars = int(
                st.number_input(
                    "Max history chars",
                    min_value=0,
                    value=max_history_chars_default,
                    step=100,
                    help=(
                        "Character cap for chat history before it is inserted into the prompt. "
                        "0 disables this character cap."
                    ),
                )
            )
            max_history_tokens = int(
                st.number_input(
                    "Max history tokens",
                    min_value=0,
                    value=max_history_tokens_default,
                    step=50,
                    help=(
                        "Approximate token cap for chat history. Use this when moving to larger "
                        "contexts and you want history to be bounded more predictably than chars."
                    ),
                )
            )
            max_chunk_chars = int(
                st.number_input(
                    "Max chunk chars",
                    min_value=0,
                    value=max_chunk_chars_default,
                    step=100,
                    help=(
                        "Per-context-row character cap before rows are added to the prompt. "
                        "Prevents one long chunk from crowding out the others."
                    ),
                )
            )
            max_chunk_tokens = int(
                st.number_input(
                    "Max chunk tokens",
                    min_value=0,
                    value=max_chunk_tokens_default,
                    step=50,
                    help=(
                        "Approximate per-context-row token cap. Use together with or instead of "
                        "character budgets when moving to models with larger context windows."
                    ),
                )
            )
            max_total_context_chars = int(
                st.number_input(
                    "Max total context chars",
                    min_value=0,
                    value=max_total_context_chars_default,
                    step=100,
                    help=(
                        "Total character budget across all selected context rows. This is the "
                        "main control for how much retrieved material reaches the model."
                    ),
                )
            )
            max_total_context_tokens = int(
                st.number_input(
                    "Max total context tokens",
                    min_value=0,
                    value=max_total_context_tokens_default,
                    step=50,
                    help=(
                        "Approximate token budget across all selected context rows. Useful when "
                        "switching to models with larger or smaller real context windows."
                    ),
                )
            )
            max_prompt_chars = int(
                st.number_input(
                    "Max prompt chars",
                    min_value=0,
                    value=max_prompt_chars_default,
                    step=100,
                    help=(
                        "Final hard character cap applied to the assembled prompt after history "
                        "and context are merged. Keep this above the history and context budgets."
                    ),
                )
            )
            max_prompt_tokens = int(
                st.number_input(
                    "Max prompt tokens",
                    min_value=0,
                    value=max_prompt_tokens_default,
                    step=50,
                    help=(
                        "Final approximate token cap for the whole prompt. This is the last guard "
                        "rail before generation starts."
                    ),
                )
            )

        st.markdown(f"**Model**: `{model_name}`")
        st.caption(
            "Retrieval: "
            f"fetch_k={fetch_k}, mode={rerank_mode}, threshold={score_threshold:.2f}, "
            f"alpha={rerank_alpha:.2f}, max_per_source={max_per_source}, "
            f"qa_boost={qa_pair_score_boost:.2f}"
        )
        st.caption(
            "Answers: "
            f"candidates={answer_sample_count}, mode={answer_rerank_mode}, "
            f"alpha={answer_rerank_alpha:.2f}, support_top_k={answer_rerank_support_top_k}"
        )
        st.caption(
            "Budgets(chars): "
            f"history={max_history_chars}, chunk={max_chunk_chars}, "
            f"context={max_total_context_chars}, prompt={max_prompt_chars}"
        )
        st.caption(
            "Budgets(tokens): "
            f"history={max_history_tokens}, chunk={max_chunk_tokens}, "
            f"context={max_total_context_tokens}, prompt={max_prompt_tokens}"
        )

    retr_cfg = RetrievalConfig(
        qdrant_path=str(rag_cfg.get("qdrant_path", "data/rag/vectordb")),
        collection_name=str(rag_cfg.get("collection_name", "docs")),
        embedding_model=str(rag_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")),
        normalize_embeddings=bool(rag_cfg.get("normalize_embeddings", True)),
        fetch_k=fetch_k,
        score_threshold=score_threshold,
        rerank_mode=rerank_mode,
        reranker_model=reranker_model,
        rerank_alpha=rerank_alpha,
        max_per_source=max_per_source,
        qa_pair_score_boost=qa_pair_score_boost,
    )
    answer_sel_cfg = AnswerSelectionConfig(
        sample_count=answer_sample_count,
        rerank_mode=answer_rerank_mode,
        rerank_alpha=answer_rerank_alpha,
        support_top_k=answer_rerank_support_top_k,
    )
    gen_cfg = GenerationConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        adapter_path=(
            adapter_override.strip() or answer_cfg.get("adapter_path") or None
        ),
        trust_remote_code=trust_remote_code,
    )

    try:
        with st.spinner("Loading retriever and model..."):
            retriever = _load_retriever(retr_cfg)
            generator = _load_generator(gen_cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize resources: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources")
            if sources:
                with st.expander("Sources"):
                    for idx, row in enumerate(sources, start=1):
                        _render_source_row(row, idx)

    question = st.chat_input("Ask a question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                retrieved_rows = retriever.search(question=question, top_k=top_k)
                selected_rows = select_context_rows(
                    rows=retrieved_rows,
                    max_rows=max_context_chunks,
                    max_chunk_chars=max_chunk_chars,
                    max_total_chars=max_total_context_chars,
                    max_chunk_tokens=max_chunk_tokens,
                    max_total_tokens=max_total_context_tokens,
                )
                context = format_context(selected_rows)
                history = format_history(
                    st.session_state.messages,
                    max_turns=max_history_turns,
                    max_chars=max_history_chars,
                    max_tokens=max_history_tokens,
                )
                prompt = build_rag_prompt(
                    question=question,
                    context=context,
                    history=history,
                )
                prompt = clip_text(prompt, max_prompt_chars)
                prompt = clip_text_to_tokens(prompt, max_prompt_tokens)
                answer_embedder, answer_cross_encoder = resolve_answer_rerank_resources(
                    retriever=retriever,
                    sample_count=answer_sel_cfg.sample_count,
                    rerank_mode=answer_sel_cfg.rerank_mode,
                    reranker_model=reranker_model,
                )
                answer, ranked_candidates = generate_best_answer(
                    generator=generator,
                    prompt=prompt,
                    question=question,
                    context_rows=selected_rows,
                    max_tokens=gen_cfg.max_tokens,
                    temperature=gen_cfg.temperature,
                    config=answer_sel_cfg,
                    embedder=answer_embedder,
                    normalize_embeddings=retriever.normalize_embeddings,
                    cross_encoder=answer_cross_encoder,
                )
            except Exception as exc:  # noqa: BLE001
                answer = f"Generation failed: {exc}"
                selected_rows = []
                ranked_candidates = []

        st.markdown(answer)
        if len(ranked_candidates) > 1:
            with st.expander("Answer Candidates"):
                for idx, candidate in enumerate(ranked_candidates, start=1):
                    st.markdown(
                        f"{idx}. score={float(candidate.get('score') or 0.0):.4f}, "
                        f"relevance={float(candidate.get('relevance_score') or 0.0):.4f}, "
                        f"support={float(candidate.get('support_score') or 0.0):.4f}"
                    )
                    st.text(str(candidate.get("answer") or ""))
        if selected_rows:
            with st.expander("Sources"):
                for idx, row in enumerate(selected_rows, start=1):
                    _render_source_row(row, idx)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": selected_rows,
        }
    )


if __name__ == "__main__":
    main()
