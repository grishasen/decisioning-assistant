#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python3 -m qa.generate_qa \
  --qa-config configs/qa_generation.yaml \
  --models-config configs/models.yaml

python3 -m qa.validate_qa --qa-config configs/qa_generation.yaml
python3 -m qa.split_dataset --qa-config configs/qa_generation.yaml
