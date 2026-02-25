#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m rag.build_index --config configs/rag.yaml --recreate
