from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from common.io_utils import read_yaml
from common.mlx_utils import MLXLoadedGenerator
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    clip_text_to_tokens,
    format_context,
    format_history,
    select_context_rows,
)
from rag.retrieve import postprocess_retrieval_rows

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


class LocalRetriever:
    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embedding_model: str,
        normalize_embeddings: bool,
        fetch_k: int,
        score_threshold: float,
        rerank_mode: str,
        reranker_model: str,
        rerank_alpha: float,
        max_per_source: int,
        qa_pair_score_boost: float,
    ) -> None:
        self._collection_name = collection_name
        self._normalize_embeddings = normalize_embeddings
        self._embedder = SentenceTransformer(embedding_model)
        self._client = QdrantClient(path=qdrant_path)

        self._fetch_k = max(1, int(fetch_k))
        self._score_threshold = max(0.0, float(score_threshold))
        self._rerank_mode = str(rerank_mode or "none").strip().lower()
        self._reranker_model = str(reranker_model)
        self._rerank_alpha = max(0.0, min(1.0, float(rerank_alpha)))
        self._max_per_source = max(0, int(max_per_source))
        self._qa_pair_score_boost = float(qa_pair_score_boost)

        self._cross_encoder: Any | None = None
        if self._rerank_mode == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self._reranker_model)
            except Exception:
                # Fall back to embedding rerank if cross-encoder cannot be loaded.
                self._rerank_mode = "embedding_cosine"

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        requested_top_k = max(1, int(top_k))
        fetch_k = max(requested_top_k, self._fetch_k)

        vector = self._embedder.encode(
            [question],
            normalize_embeddings=self._normalize_embeddings,
        )[0].tolist()

        resp = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=fetch_k,
            with_payload=True,
        )

        rows: list[dict[str, Any]] = []
        for point in resp.points:
            payload = point.payload or {}
            qdrant_score = float(point.score or 0.0)
            rows.append(
                {
                    "score": qdrant_score,
                    "qdrant_score": qdrant_score,
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "source_ref": payload.get("source_ref"),
                    "source_type": payload.get("source_type"),
                    "record_type": payload.get("record_type", "chunk"),
                    "text": payload.get("text") or "",
                    "metadata": payload.get("metadata", {}),
                }
            )

        return postprocess_retrieval_rows(
            question=question,
            rows=rows,
            embedder=self._embedder,
            normalize_embeddings=self._normalize_embeddings,
            top_k=requested_top_k,
            score_threshold=self._score_threshold,
            rerank_mode=self._rerank_mode,
            reranker_model=self._reranker_model,
            rerank_alpha=self._rerank_alpha,
            max_per_source=self._max_per_source,
            qa_pair_score_boost=self._qa_pair_score_boost,
            cross_encoder=self._cross_encoder,
        )


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


def _format_date_only(raw_value: Any) -> str:
    if raw_value is None:
        return "unknown"

    text = str(raw_value).strip()
    if not text:
        return "unknown"

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
        return text


def _extract_pdf_page(source_ref: str, metadata: dict[str, Any]) -> str:
    for key in ("page", "page_start"):
        raw = metadata.get(key)
        if isinstance(raw, int):
            return str(raw)
        if isinstance(raw, str) and raw.isdigit():
            return raw

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
        message_date = _format_date_only(metadata.get("created_at"))
        return f"- Webex Space: `{room_title}` | Date: `{message_date}` ({score_text})"

    record_type = str(row.get("record_type") or "chunk")
    return f"- `{source_ref}` [{record_type}] ({score_text})"


def _source_popup_text(row: dict[str, Any], max_chars: int = 20000) -> str:
    value = str(row.get("text") or "").strip()
    if not value:
        return "No retrieved document text available for this source."

    if len(value) <= max_chars:
        return value

    return value[:max_chars].rstrip() + "\n\n...[truncated]"


def _render_source_row(row: dict[str, Any], index: int) -> None:
    left, right = st.columns([0.82, 0.18], vertical_alignment="top")

    with left:
        st.markdown(_format_source_line(row))

    with right:
        with st.popover(f"Show text", type="tertiary"):
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

    max_history_turns = max(0, int(rag_cfg.get("max_history_turns", 6)))
    max_history_chars = max(0, int(rag_cfg.get("max_history_chars", 3000)))
    max_chunk_chars = max(0, int(rag_cfg.get("max_chunk_chars", 1200)))
    max_total_context_chars = max(0, int(rag_cfg.get("max_total_context_chars", 5000)))
    max_prompt_chars = max(0, int(rag_cfg.get("max_prompt_chars", 12000)))

    max_history_tokens = max(0, int(rag_cfg.get("max_history_tokens", 0)))
    max_chunk_tokens = max(0, int(rag_cfg.get("max_chunk_tokens", 0)))
    max_total_context_tokens = max(0, int(rag_cfg.get("max_total_context_tokens", 0)))
    max_prompt_tokens = max(0, int(rag_cfg.get("max_prompt_tokens", 0)))

    fetch_k = max(1, int(rag_cfg.get("fetch_k", max(top_k_default * 3, top_k_default))))
    score_threshold = max(0.0, float(rag_cfg.get("score_threshold", 0.0)))
    rerank_mode = str(rag_cfg.get("rerank_mode", "cross_encoder")).strip().lower()
    reranker_model = str(rag_cfg.get("reranker_model", "BAAI/bge-reranker-base"))
    rerank_alpha = max(0.0, min(1.0, float(rag_cfg.get("rerank_alpha", 0.65))))
    max_per_source = max(0, int(rag_cfg.get("max_per_source", 0)))
    qa_pair_score_boost = float(rag_cfg.get("qa_pair_score_boost", 0.0))

    with st.sidebar:
        top_k = int(
            st.number_input(
                "Top-k retrieval",
                min_value=top_k_min,
                max_value=top_k_max,
                value=top_k_default,
            )
        )
        max_context_chunks = int(
            st.number_input(
                "Max context chunks",
                min_value=max_ctx_min,
                max_value=max_ctx_max,
                value=max_ctx_default,
            )
        )
        st.markdown(f"**Model**: `{model_name}`")
        st.caption(
            "Retrieval: "
            f"fetch_k={fetch_k}, mode={rerank_mode}, threshold={score_threshold:.2f}, "
            f"max_per_source={max_per_source}"
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
                answer = generator.generate(
                    prompt=prompt,
                    max_tokens=gen_cfg.max_tokens,
                    temperature=gen_cfg.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                answer = f"Generation failed: {exc}"
                selected_rows = []

        st.markdown(answer)
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
