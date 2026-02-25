#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python3 -m training.run_lora --config configs/finetune.yaml
