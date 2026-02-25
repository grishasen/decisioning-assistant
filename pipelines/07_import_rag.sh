#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m rag.import_index \
  --config configs/rag.yaml \
  --input-dir data/rag/export \
  --recreate
