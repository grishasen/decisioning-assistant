# Overview

## Purpose

DecisioningAssistant is a local-first knowledge assistant for product documentation and Webex discussion archives. It is designed to run on Apple Silicon using MLX-compatible language models and a local Qdrant vector index.

The repository supports four linked workflows:

1. Ingest source material from PDFs and Webex exports.
2. Convert those sources into normalized documents and retrieval chunks.
3. Generate synthetic QA supervision and fine-tune an instruction model.
4. Build and query a hybrid RAG index from source chunks and QA pairs.

## Core Capabilities

- PDF ingestion with structure-aware extraction and paragraph-bound chunking.
- Webex ingestion with thread-aware grouping.
- Local synthetic QA generation using a preloaded MLX model.
- LoRA fine-tuning through `mlx_lm`.
- Local hybrid RAG using Qdrant.
- Streamlit chat UI with chat history, citations, and retrieved source previews.
- Export and import of the RAG database for transfer to another machine.

## Main Data Flow

1. Raw files are placed in `data/raw/pdf/` and `data/raw/webex/`.
2. Ingestion creates normalized document records in `data/staging/documents/`.
3. Normalization creates retrieval chunks in `data/staging/chunks/chunks.jsonl`.
4. QA generation creates raw, cleaned, and split datasets under `data/qa/`.
5. Fine-tuning reads QA data and writes adapters under `data/models/`.
6. RAG indexing stores vectors and payloads in `data/rag/vectordb/`.
7. The local chat app queries Qdrant, applies reranking, builds a bounded prompt, and generates an answer with the configured MLX model.

## Source Types

### PDF

PDFs are parsed with PyMuPDF. The current pipeline extracts paragraph-like text units from PDF blocks, preserves document structure metadata, and packs whole paragraphs into chunks where possible. Page ranges, section titles, and section paths are carried forward into RAG metadata.

### Webex

Webex raw exports are grouped by thread by default. A thread starts with the root message and includes child messages ordered by creation time. Threads with fewer than two messages are skipped. The current pipeline keeps thread-level metadata such as room title, thread ID, message count, and timestamps.

## Retrieval Strategy

The index can contain:

- source chunks from PDFs and Webex threads
- cleaned synthetic QA pairs linked back to source chunks

This gives a hybrid retrieval strategy:

- source chunks preserve original evidence
- QA pairs improve retrieval for natural user questions phrased differently from the source text

## Default Operating Assumptions

- English-only corpus and QA generation.
- Local-first processing and local inference.
- Config defaults tuned for a MacBook Pro M3 with 24 GB unified memory, but not hard-limited to that hardware.
- Minimal custom RAG framework instead of a large orchestration library.

## Repository Layout

```text
configs/      Runtime and pipeline configuration
data/         Raw, staged, QA, model, and RAG data
pipelines/    Shell wrappers for common end-to-end runs
src/          Python package source
tests/        Automated tests
wiki/         Repository wiki pages
```
