# Usage and Configuration

## Installation

```bash
cd /Users/seng1/Documents/DecisioningAssistant
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[webex]
pip install -e .[pdf_markdown]
```

## Main Entry Points

You can use either the shell scripts in `pipelines/` or the Python CLI.

CLI help:

```bash
decisioning-assistant --help
```

Available workflow commands:

```bash
decisioning-assistant ingest
decisioning-assistant qa
decisioning-assistant finetune --finetune-config configs/finetune.yaml
decisioning-assistant rag-index
decisioning-assistant rag-export --output-dir data/rag/export
decisioning-assistant rag-import --input-dir data/rag/export
decisioning-assistant app --server-port 8501
```

## Typical End-to-End Flow

### 1. Add source data

- place PDFs in `data/raw/pdf/`
- place Webex dumps in `data/raw/webex/`

### 2. Ingest and normalize

```bash
decisioning-assistant ingest
```

This runs:

- PDF ingestion
- Webex ingestion
- normalization into `documents.jsonl` and `chunks.jsonl`

### 3. Generate QA data

```bash
decisioning-assistant qa
```

This runs:

- QA generation
- QA validation and filtering
- train/validation/test split preparation

### 4. Fine-tune a model

```bash
decisioning-assistant finetune --finetune-config configs/finetune.yaml
```

### 5. Build or refresh the RAG index

```bash
decisioning-assistant rag-index
```

Use `--recreate` if you want to replace the collection instead of upserting into the existing one.

### 6. Start the app

```bash
decisioning-assistant app --server-port 8501
```

## Pipeline Scripts

If you prefer shell wrappers:

```bash
./pipelines/01_ingest.sh
./pipelines/02_generate_qa.sh
./pipelines/03_finetune.sh
./pipelines/04_build_rag.sh
./pipelines/05_eval.sh
./pipelines/06_export_rag.sh
./pipelines/07_import_rag.sh
```

## Configuration Files

### `configs/sources.yaml`

Controls ingestion and normalization:

- PDF input and output paths
- Webex input and output paths
- source metadata defaults
- normalization outputs
- chunk sizing defaults

Important fields:

- `pdf.input_dir`
- `webex.raw_dir`
- `normalize.chunk_size`
- `normalize.chunk_overlap`
- `normalize.min_chunk_chars`

Note: PDF chunking is now structure-aware and paragraph-bound where possible, so normalization settings matter more for generic text than for pre-chunked PDF records.

### `configs/qa_generation.yaml`

Controls QA dataset generation and cleaning:

- `input_chunks`
- `output_raw_qa`
- `output_clean_qa`
- `qa_per_chunk`
- `min_question_chars`
- `min_answer_chars`
- `max_answer_chars`
- `min_webex_chunk_chars`
- `webex_user_name`
- `max_webex_thread_answer_chars`

Important Webex-specific behavior:

- short Webex chunks below `min_webex_chunk_chars` are skipped
- when `webex_user_name` is set, only threads with child messages from that user are used
- when `webex_user_name` is set, only that user’s child messages are kept in the Webex answer

### `configs/models.yaml`

Controls generation and embedding models:

- `qa_generator`
- `answer_model`
- `embeddings`

Important fields:

- `model`
- `adapter_path`
- `max_tokens`
- `temperature`
- `trust_remote_code`
- `embeddings.model_name`

The checked-in config currently uses `mlx-community/DeepHermes-3-Llama-3-8B-Preview-bf16` for both QA generation and answer generation.

### `configs/finetune.yaml`

Controls local MLX LoRA training:

- base model
- dataset location
- adapter output path
- batch size
- sequence length
- number of iterations
- learning rate
- LoRA settings

The current defaults are conservative for 24 GB unified memory but can be raised on larger hardware.

### `configs/rag.yaml`

Controls index content, retrieval, reranking, and prompt budgets.

Important fields:

- `include_qa`
- `qa_text_mode`
- `qdrant_path`
- `collection_name`
- `embedding_model`
- `top_k`
- `fetch_k`
- `score_threshold`
- `rerank_mode`
- `reranker_model`
- `rerank_alpha`
- `qa_pair_score_boost`
- `max_per_source`
- `max_context_chunks`
- `max_history_turns`
- `max_chunk_chars`
- `max_total_context_chars`
- `max_prompt_chars`
- token budget fields

## Streamlit App Settings

The app sidebar exposes:

- top-k retrieval
- max context chunks
- advanced retrieval controls
- prompt budget controls
- adapter override

Use these runtime controls for experiments before changing checked-in defaults in YAML.

## Common Examples

### Run ingestion from a different project root

```bash
decisioning-assistant --project-root /Users/seng1/Documents/DecisioningAssistant ingest
```

### Skip Webex during ingestion

```bash
decisioning-assistant ingest --skip-webex
```

### Rebuild the RAG collection from scratch

```bash
decisioning-assistant rag-index --recreate
```

### Export only PDF records

```bash
decisioning-assistant rag-export --output-dir data/rag/export --source pdf
```

### Export only Webex records

```bash
decisioning-assistant rag-export --output-dir data/rag/export --source webex
```

### Import an exported collection on another machine

```bash
decisioning-assistant rag-import --input-dir data/rag/export --recreate
```

## Operational Notes

- The repository is currently English-only.
- RAG quality depends heavily on chunk quality, QA filtering, and reranking settings.
- For larger models or machines with more memory, the first parameters to revisit are:
  - generation `max_tokens`
  - prompt budgets
  - `fetch_k`
  - `max_context_chunks`
  - reranker choice

## Recommended Editing Practice

Treat checked-in YAML files as baseline defaults and use app-side overrides or CLI flags for experiments first. Once a setting proves stable, move it into the relevant config file.
