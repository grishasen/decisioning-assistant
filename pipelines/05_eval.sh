#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m training.evaluate_model --models-config configs/models.yaml --test-path data/qa/test.jsonl --limit 100
