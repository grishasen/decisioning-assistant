#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m rag.export_index \
  --config configs/rag.yaml \
  --output-dir data/rag/export
