# DecisioningAssistant

This project builds a local pipeline to:
1. Ingest PDFs and Webex exports.
2. Convert source text into chunked documents.
3. Generate English QA pairs locally with Gemma on MLX.
4. Fine-tune Gemma with LoRA/QLoRA via `mlx_lm`.
5. Build a local open-source RAG index (Qdrant).
6. Answer questions with retrieval-augmented local generation.

## Why This Stack
- **Model family**: Gemma (default: `mlx-community/gemma-2-2b-it-4bit`) for practical local training/inference on 24GB unified memory.
- **Vector DB**: **Qdrant local mode** for a strong open-source HNSW implementation, no cloud dependency, and reliable persistence.
- **PDF parsing**: PyMuPDF-first extraction with markdown-friendly output per page.
- **Webex**: parser for archive dumps + wrapper command for `webexspacearchive` retrieval workflow.
- **RAG**: minimal custom code (no heavy framework lock-in).

## Repository Layout
```text
configs/
  sources.yaml        # ingestion + chunking config
  models.yaml         # QA/answer model + embedding config
  qa_generation.yaml  # QA generation/clean/split config
  finetune.yaml       # MLX LoRA config
  rag.yaml            # RAG index/retrieval config

data/
  raw/pdf/            # input PDFs
  raw/webex/          # Webex archive JSON/JSONL
  staging/documents/  # normalized docs
  staging/chunks/     # chunked corpus
  qa/                 # QA datasets (raw/clean/split)
  rag/vectordb/       # local Qdrant data

src/
  ingestion/          # PDF/Webex ingestion + normalization
  qa/                 # QA generation, validation, split
  training/           # MLX LoRA run + quick eval + adapter fuse
  rag/                # index build, retrieve, local RAG chat
  common/             # schemas, IO, prompts, MLX wrapper

pipelines/
  01_ingest.sh
  02_generate_qa.sh
  03_finetune.sh
  04_build_rag.sh
  05_eval.sh
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

Optional extras:
```bash
pip install -e .[dev]
pip install -e .[webex]
```

## End-to-End Run
1. Put PDFs into `data/raw/pdf/`.
2. Put Webex archive files (`.json` / `.jsonl`) into `data/raw/webex/`.
3. Run:
```bash
./pipelines/01_ingest.sh
./pipelines/02_generate_qa.sh
./pipelines/03_finetune.sh
./pipelines/04_build_rag.sh
```

Quick query:
```bash
PYTHONPATH=src python3 -m rag.chat_local "How do we rotate API keys?"
```

## Webex Retrieval (`webexspacearchive`)
If you use `webexspacearchive` for exports, call:
```bash
PYTHONPATH=src python3 -m ingestion.fetch_webex_archive \
  --space-id <SPACE_ID> \
  --output-dir data/raw/webex \
  --command-template "webex-space-archive.py {space_id}"
```
Adjust `--command-template` to your local installation command if needed.

## Local-only QA generation vs Cloud fallback
- **Local-only QA generation**:
  - All synthetic QA is generated on your laptop.
  - Strongest privacy/compliance posture.
  - Fully reproducible offline once models are cached.
  - Tradeoff: lower throughput and possibly lower quality on very complex passages.
- **Cloud fallback for low-confidence samples**:
  - Keep local generation as default.
  - Send only low-confidence or malformed outputs (for example: invalid JSON, weak grounding score, or low answer quality) to a cloud LLM for repair/regeneration.
  - Tradeoff: better dataset quality and speed of cleanup, but introduces data governance, cost, and external dependency.

Current code is configured for **local-only** generation. You can add fallback later by inserting a second generator in `src/qa/generate_qa.py` for failed chunks.

## Notes and Constraints
- This project is currently **English-only** by design.
- `mlx_lm` CLI arguments can evolve; adjust `configs/finetune.yaml` + `src/training/run_lora.py` flags if your version differs.
- PyMuPDF licensing is dual (AGPL/commercial). Verify fit for your usage context.

## Streamlit RAG Chat UI
Run the local chat app with:
```bash
PYTHONPATH=src streamlit run src/rag/assistant_app.py
```

The app provides:
- chat history persisted in the Streamlit session,
- top-k retrieval controls,
- source citations for each assistant response,
- generation via `MLXLoadedGenerator` (model loaded once and reused).

## Python Package CLI
After installation (`pip install -e .`), use:

```bash
decisioning-assistant --help
```

Subcommands:

```bash
# 1) Ingest PDF + Webex + normalize
decisioning-assistant ingest

# 2) Generate/clean/split QA dataset
decisioning-assistant qa

# 3) Fine-tune model via MLX LoRA
decisioning-assistant finetune --finetune-config configs/finetune.yaml

# 4) Build or update local RAG index (upserts if collection exists)
decisioning-assistant rag-index

# 5) Recreate collection and rebuild from scratch
decisioning-assistant rag-index --recreate

# 6) Start Streamlit RAG assistant UI
decisioning-assistant app --server-port 8501
```

Advanced examples:

```bash
# Run from outside project root
decisioning-assistant --project-root /Users/vasya/Documents/DecisioningAssistant ingest

# Skip Webex ingestion but still normalize
decisioning-assistant ingest --skip-webex

# Run only QA split stage
decisioning-assistant qa --skip-generate --skip-validate
```
