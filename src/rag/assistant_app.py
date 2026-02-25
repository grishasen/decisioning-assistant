from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from common.io_utils import read_yaml
from common.mlx_utils import MLXLoadedGenerator
from rag.prompt_budget import (
    build_rag_prompt,
    clip_text,
    format_context,
    format_history,
    select_context_rows,
)


@dataclass(frozen=True)
class RetrievalConfig:
    qdrant_path: str
    collection_name: str
    embedding_model: str
    normalize_embeddings: bool


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
    ) -> None:
        self._collection_name = collection_name
        self._normalize_embeddings = normalize_embeddings
        self._embedder = SentenceTransformer(embedding_model)
        self._client = QdrantClient(path=qdrant_path)

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        vector = self._embedder.encode(
            [question],
            normalize_embeddings=self._normalize_embeddings,
        )[0].tolist()

        resp = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )

        rows: list[dict[str, Any]] = []
        for point in resp.points:
            payload = point.payload or {}
            rows.append(
                {
                    "score": point.score,
                    "chunk_id": payload.get("chunk_id"),
                    "source_ref": payload.get("source_ref"),
                    "text": payload.get("text") or "",
                    "metadata": payload.get("metadata", {}),
                }
            )
        return rows


@st.cache_resource(show_spinner=False)
def _load_retriever(cfg: RetrievalConfig) -> LocalRetriever:
    return LocalRetriever(
        qdrant_path=cfg.qdrant_path,
        collection_name=cfg.collection_name,
        embedding_model=cfg.embedding_model,
        normalize_embeddings=cfg.normalize_embeddings,
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


def main() -> None:
    st.set_page_config(page_title="DecisioningAssistant RAG Chat", layout="wide")
    st.title("DecisioningAssistant RAG Chat")
    st.caption("Local RAG chat using Qdrant + MLXLoadedGenerator")

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
        clear_chat = st.button("Clear Chat History")

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
            "Budget config: "
            f"history_turns={max_history_turns}, "
            f"history_chars={max_history_chars}, "
            f"chunk_chars={max_chunk_chars}, "
            f"context_chars={max_total_context_chars}, "
            f"prompt_chars={max_prompt_chars}"
        )

    retr_cfg = RetrievalConfig(
        qdrant_path=str(rag_cfg.get("qdrant_path", "data/rag/vectordb")),
        collection_name=str(rag_cfg.get("collection_name", "docs")),
        embedding_model=str(rag_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")),
        normalize_embeddings=bool(rag_cfg.get("normalize_embeddings", True)),
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
                    for row in sources:
                        source_ref = row.get("source_ref") or "unknown"
                        score = float(row.get("score") or 0.0)
                        st.markdown(f"- `{source_ref}` (score={score:.4f})")

    question = st.chat_input("Ask a question about your indexed documents")
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
                )
                context = format_context(selected_rows)
                history = format_history(
                    st.session_state.messages,
                    max_turns=max_history_turns,
                    max_chars=max_history_chars,
                )
                prompt = build_rag_prompt(
                    question=question, context=context, history=history
                )
                prompt = clip_text(prompt, max_prompt_chars)
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
                for row in selected_rows:
                    source_ref = row.get("source_ref") or "unknown"
                    score = float(row.get("score") or 0.0)
                    st.markdown(f"- `{source_ref}` (score={score:.4f})")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": selected_rows,
        }
    )


if __name__ == "__main__":
    main()
