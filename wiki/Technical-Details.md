# Technical Details

## Architecture

The codebase is organized into five main areas:

- `src/ingestion`: PDF and Webex ingestion plus normalization into common document and chunk schemas
- `src/qa`: synthetic QA generation, validation, and dataset splitting
- `src/training`: LoRA fine-tuning and evaluation helpers for MLX models
- `src/rag`: index build, retrieval, export/import, CLI chat, and Streamlit app
- `src/common`: shared schemas, prompts, IO helpers, logging, text utilities, and MLX wrappers

The package entry point is `decisioning-assistant`, defined in `/Users/seng1/Documents/DecisioningAssistant/pyproject.toml`.

## Schemas and Records

The pipeline uses three main record types:

- `DocumentRecord`: normalized source document units
- `ChunkRecord`: retrieval chunks produced from normalized documents
- `QARecord`: synthetic supervised question-answer examples linked back to a chunk

Canonical metadata fields used across the RAG pipeline include:

- `product`
- `doc_version`
- `doc_type`
- `section_path`
- `page_start`
- `page_end`
- `room_id`
- `room_title`
- `thread_id`
- `message_count`
- `created_at`
- `updated_at`
- `ingested_at`

## PDF Ingestion

PDF ingestion is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/ingestion/ingest_pdfs.py`.

Current behavior:

- uses PyMuPDF for extraction
- preserves structure such as headings, section titles, and page ranges
- extracts paragraph candidates from PDF text blocks
- packs chunks by paragraph instead of arbitrary character windows
- only falls back to sentence-level splitting when a single paragraph is too large

This is important for RAG quality because it reduces sentence fragmentation and preserves local semantic coherence inside each chunk.

## Webex Ingestion

Webex ingestion is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/ingestion/ingest_webex.py`.

Current behavior:

- derives `room_title` from the raw file name
- groups messages into threads by root and child relationship
- ensures a thread chunk always starts with the thread root message
- skips threads with fewer than two messages
- writes thread-aware metadata such as room, thread, count, and timestamps

The normalization stage preserves thread chunks rather than breaking them into generic overlapping text windows.

## QA Generation

QA generation is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/qa/generate_qa.py`.

There are two distinct generation paths:

### Source chunks

For non-Webex-thread chunks, the model generates QA pairs directly from chunk text using a prompt template in `/Users/seng1/Documents/DecisioningAssistant/src/common/prompts.py`.

### Webex thread chunks

For Webex threads, the process is specialized:

- the model generates the question
- the answer is built deterministically from child messages
- if `webex_user_name` is set, only threads where that user appears in child messages are used
- when that filter is enabled, only that user’s child messages are kept in the answer

This makes Webex supervision more grounded in actual discussion replies and reduces fabricated answers.

## Fine-Tuning

Fine-tuning is executed through `/Users/seng1/Documents/DecisioningAssistant/src/training/run_lora.py` using `mlx_lm`.

The checked-in configuration in `/Users/seng1/Documents/DecisioningAssistant/configs/finetune.yaml` currently points to:

- model: `mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16`
- LoRA rank, alpha, dropout, scale, and layer count
- sequence length, batch size, iterations, learning rate, and checkpoint settings

The training pipeline is intended for local adapter training rather than full model fine-tuning.

## RAG Index

RAG indexing is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/rag/build_index.py`.

Current design:

- local Qdrant storage
- embedding model from `configs/models.yaml` and `configs/rag.yaml`
- index can include both source chunks and cleaned QA pairs
- existing collections can be updated or fully recreated

QA inclusion is configurable through:

- `include_qa`
- `qa_path`
- `qa_text_mode`
- `max_qa_answer_chars`

## Retrieval and Reranking

Retrieval logic is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/rag/retrieve.py`.

Pipeline stages:

1. Embed the user question.
2. Fetch `fetch_k` nearest neighbors from Qdrant.
3. Filter low-scoring rows by `score_threshold`.
4. Rerank with one of:
   - `cross_encoder`
   - `embedding_cosine`
   - `none`
5. Blend vector and reranker scores with `rerank_alpha`.
6. Optionally boost QA records with `qa_pair_score_boost`.
7. Apply `max_per_source` to reduce source over-concentration.

`cross_encoder` is the default reranking mode in the checked-in RAG config.

## Prompt Budgeting

Prompt budgeting is implemented in `/Users/seng1/Documents/DecisioningAssistant/src/rag/prompt_budget.py`.

The assistant constrains:

- number of selected rows
- per-row text length
- total context length
- chat history length
- final prompt size

Both character-based and approximate token-based caps are supported. This is intended to keep the application usable across different model sizes and context windows.

## Streamlit App

The interactive UI is `/Users/seng1/Documents/DecisioningAssistant/src/rag/assistant_app.py`.

Key features:

- chat history stored in the Streamlit session
- runtime RAG controls in the sidebar
- source display for PDF and Webex results
- source text popup for retrieved rows
- model loaded once through `MLXLoadedGenerator`

The UI now exposes both basic and advanced retrieval settings, including reranking and prompt-budget controls.

## Export and Import

RAG portability is handled by:

- `/Users/seng1/Documents/DecisioningAssistant/src/rag/export_index.py`
- `/Users/seng1/Documents/DecisioningAssistant/src/rag/import_index.py`

The export format contains:

- collection metadata
- vector configuration
- points serialized as JSONL

Export can be filtered by source type, for example `pdf` or `webex`.
