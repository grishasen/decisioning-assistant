# DecisioningAssistant

DecisioningAssistant is a local-first MLX project for macOS that ingests PDFs and Webex threads, generates QA datasets, fine-tunes small instruction models, builds a hybrid local RAG index, and serves a Streamlit chat assistant with source-aware citations.

## What It Does
- Ingests PDF documentation with structure-aware paragraph chunking.
- Fetches Webex room history directly from the Webex REST API using `rooms.json` plus a YAML config.
- Groups Webex data into thread-based chunks so each chunk starts with the thread root.
- Generates English QA pairs locally with an MLX-loaded model.
- Fine-tunes MLX-compatible models with LoRA.
- Builds or updates a local Qdrant index from source chunks and optional QA pairs.
- Runs a Streamlit RAG app with chat history, retrieval controls, reranking, answer selection, citations, and source popups.

## Current Pipeline Highlights
- PDF ingestion is structure-aware: chunks are built from whole paragraphs and keep section/page metadata.
- Webex ingestion is thread-aware: threads with fewer than 2 messages are skipped, and each thread chunk keeps room and thread metadata.
- Webex metadata now includes a `webexteams://...` deep link to the parent/root message in the thread.
- Webex QA generation uses the thread start to generate the question and uses child messages as the answer.
- Webex QA can be filtered to a specific user, keeping only threads where that user appears in child messages and only that user’s child messages in the answer.
- RAG retrieval supports vector search plus reranking with `cross_encoder`, `embedding_cosine`, or `none`.
- Answer generation supports Best-of-N answer selection with reranking. Default candidate count is `4`.
- The Streamlit source popup can show the retrieved text and the Webex parent-message link when available.

## Repository Layout
```text
configs/
  sources.yaml
  models.yaml
  qa_generation.yaml
  finetune.yaml
  rag.yaml
  webex_fetch.yaml

data/
  raw/pdf/
  raw/webex/
  staging/documents/
  staging/chunks/
  qa/
  rag/vectordb/

pipelines/
  01_ingest.sh
  02_generate_qa.sh
  03_finetune.sh
  04_build_rag.sh
  05_eval.sh
  06_export_rag.sh
  07_import_rag.sh

src/
  common/
  decisioning_assistant/
  ingestion/
  qa/
  rag/
  training/

wiki/
  Home.md
  Overview.md
  Technical-Details.md
  Usage-and-Configuration.md
```

## Requirements
- macOS with Apple Silicon for MLX workflows.
- Python `>=3.10`.
- English-only source material and QA generation.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional developer tools:
```bash
pip install -e .[dev]
```

## Main Configuration Files
- `configs/sources.yaml`: PDF and Webex ingestion paths plus normalization/chunking settings.
- `configs/models.yaml`: QA generator, answer model, and embedding model settings.
- `configs/qa_generation.yaml`: QA generation, validation, split, and Webex-specific QA controls.
- `configs/finetune.yaml`: MLX LoRA fine-tuning settings.
- `configs/rag.yaml`: indexing, retrieval, reranking, answer selection, and prompt-budget settings.
- `configs/webex_fetch.yaml`: direct Webex API fetch settings.

## Typical End-to-End Workflow
1. Fetch raw Webex spaces if needed.
2. Put PDFs into `data/raw/pdf/`.
3. Run ingestion and chunking.
4. Generate QA.
5. Fine-tune if needed.
6. Build or update the RAG index.
7. Start the chat app.

Example:
```bash
#required for webex dics
decisioning-assistant webex-fetch \
  --rooms-json configs/rooms.json \
  --config configs/webex_fetch.yaml \
  --output-dir data/raw/webex

#put any pdf into pdf dir

decisioning-assistant ingest #required step
decisioning-assistant qa #optional
decisioning-assistant finetune --finetune-config configs/finetune.yaml #optional
decisioning-assistant rag-index --recreate
decisioning-assistant app --server-port 8501
```

## CLI Commands
```bash
# Ingest PDF + Webex + normalize
decisioning-assistant ingest

# Generate, validate, and split QA
decisioning-assistant qa

# Fine-tune with MLX LoRA
decisioning-assistant finetune --finetune-config configs/finetune.yaml

# Build or update the hybrid RAG index
decisioning-assistant rag-index

# Recreate the RAG collection from scratch
decisioning-assistant rag-index --recreate

# Export the RAG index
decisioning-assistant rag-export --output-dir data/rag/export

# Export only selected source types
decisioning-assistant rag-export --output-dir data/rag/export --source pdf

decisioning-assistant rag-export --output-dir data/rag/export --source webex

# Import an exported RAG bundle
decisioning-assistant rag-import --input-dir data/rag/export --recreate

# Start the Streamlit app
decisioning-assistant app --server-port 8501
```

## Direct Webex Fetch
The project no longer depends on `webexspacearchive` for fetching room history. Raw Webex exports can be created directly through the Webex API.

Example:
```bash
decisioning-assistant webex-fetch \
  --rooms-json /path/to/rooms.json \
  --config configs/webex_fetch.yaml \
  --output-dir data/raw/webex
```

Notes:
- `--room-type group` is the default.
- Output file names are derived from the room title and shortened to 80 characters.
- The fetch config only uses `token` and `max_total_messages`.

## QA Generation Notes
- QA generation is local-only.
- Short Webex chunks are skipped using `min_webex_chunk_chars`.
- Webex thread QA uses generated questions plus child-message answers.
- `webex_user_name` can restrict QA generation to replies from a specific user.
- `max_webex_thread_answer_chars` controls the separate answer cap for Webex thread answers.

## RAG Notes
- Qdrant runs locally on disk.
- The index can include raw source chunks, QA pairs, or both.
- Retrieval reranking and answer reranking are separate stages.
- The default retrieval reranker is `cross_encoder`.
- The default answer-selection candidate count is `4`.
- The Streamlit app exposes the main retrieval, reranking, and prompt-budget controls in the sidebar.

## Streamlit App
Run the app directly if needed:
```bash
PYTHONPATH=src streamlit run src/rag/assistant_app.py
```

The app provides:
- session chat history,
- configurable retrieval and prompt budgets,
- answer Best-of-N selection,
- source citations,
- source popups with retrieved text,
- Webex room timestamp display,
- Webex parent-message deep links when available.

## Export and Import
Portable RAG bundles can be moved to another machine.

Example:
```bash
# Export
decisioning-assistant rag-export --output-dir data/rag/export

# Import on another machine
decisioning-assistant rag-import --input-dir data/rag/export --recreate
```

## Notes
- The defaults in `configs/` were tuned for a MacBook Pro M3 with 24GB RAM, but the code is not hard-limited to that hardware.
- Larger future Apple Silicon systems can increase model size, retrieval depth, and prompt budgets through config.
- After changing Webex ingestion metadata, rerun ingestion, QA generation, and RAG indexing so new metadata reaches the app.
- PyMuPDF uses a dual AGPL/commercial license. Check fit for your usage.

## Documentation
See the wiki pages in `wiki/` for a fuller walkthrough:
- `wiki/Overview.md`
- `wiki/Technical-Details.md`
- `wiki/Usage-and-Configuration.md`
