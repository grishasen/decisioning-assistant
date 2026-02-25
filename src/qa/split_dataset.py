from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common.io_utils import iter_jsonl, read_yaml, write_jsonl
from common.logging_utils import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are a domain assistant. Answer accurately based on product documentation and archived team discussions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split cleaned QA into train/valid/test JSONL files.")
    parser.add_argument("--qa-config", default="configs/qa_generation.yaml")
    return parser.parse_args()


def _to_mlx_chat_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ],
        "metadata": {
            "qa_id": item.get("qa_id"),
            "chunk_id": item.get("chunk_id"),
            "source_ref": item.get("source_ref"),
            "source_type": item.get("source_type"),
        },
    }


def _split(items: list[dict[str, Any]], train_ratio: float, valid_ratio: float) -> tuple[list, list, list]:
    n = len(items)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    train = items[:train_end]
    valid = items[train_end:valid_end]
    test = items[valid_end:]
    return train, valid, test


def main() -> None:
    args = parse_args()
    qa_cfg = read_yaml(args.qa_config)

    input_path = str(qa_cfg.get("output_clean_qa", "data/qa/qa_clean.jsonl"))
    output_split_dir = Path(str(qa_cfg.get("output_split_dir", "data/qa")))
    output_split_dir.mkdir(parents=True, exist_ok=True)
    output_mlx_dir = output_split_dir / "mlx"
    output_mlx_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = float(qa_cfg.get("train_ratio", 0.9))
    valid_ratio = float(qa_cfg.get("valid_ratio", 0.08))
    seed = int(qa_cfg.get("seed", 42))

    rows = list(iter_jsonl(input_path))
    if not rows:
        logger.warning("No cleaned QA rows found at %s", input_path)
        for name in ["train", "valid", "test"]:
            write_jsonl(output_split_dir / f"{name}.jsonl", [])
            write_jsonl(output_mlx_dir / f"{name}.jsonl", [])
        return

    random.Random(seed).shuffle(rows)
    train, valid, test = _split(rows, train_ratio, valid_ratio)

    for name, subset in [("train", train), ("valid", valid), ("test", test)]:
        raw_path = output_split_dir / f"{name}.jsonl"
        mlx_path = output_mlx_dir / f"{name}.jsonl"
        write_jsonl(raw_path, subset)
        write_jsonl(mlx_path, [_to_mlx_chat_row(item) for item in subset])
        logger.info("Wrote %s rows to %s and %s", len(subset), raw_path, mlx_path)


if __name__ == "__main__":
    main()
