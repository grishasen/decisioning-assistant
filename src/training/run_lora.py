from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from common.io_utils import read_yaml
from common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLX LoRA fine-tuning with config-driven settings.")
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([f"--{key}", str(value)])


def build_command(cfg: dict[str, Any]) -> list[str]:
    cmd: list[str] = ["mlx_lm.lora"]

    _append_arg(cmd, "model", cfg.get("model"))
    if cfg.get("train", True):
        cmd.append("--train")

    _append_arg(cmd, "data", cfg.get("data_dir"))
    _append_arg(cmd, "adapter-path", cfg.get("adapter_path"))
    _append_arg(cmd, "iters", cfg.get("iters"))
    _append_arg(cmd, "batch-size", cfg.get("batch_size"))
    _append_arg(cmd, "learning-rate", cfg.get("learning_rate"))
    _append_arg(cmd, "max-seq-length", cfg.get("max_seq_length"))
    _append_arg(cmd, "num-layers", cfg.get("num_layers"))
    _append_arg(cmd, "steps-per-report", cfg.get("steps_per_report"))
    _append_arg(cmd, "steps-per-eval", cfg.get("steps_per_eval"))
    _append_arg(cmd, "save-every", cfg.get("save_every"))

    extra_args = cfg.get("extra_args")
    if isinstance(extra_args, str) and extra_args.strip():
        cmd.extend(shlex.split(extra_args))
    elif isinstance(extra_args, list):
        cmd.extend(str(x) for x in extra_args)

    return cmd


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)

    data_dir = Path(str(cfg.get("data_dir", "data/qa/mlx")))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Training dataset directory does not exist: {data_dir}. "
            "Run the QA splitting pipeline first."
        )

    command = build_command(cfg)
    logger.info("Running command: %s", " ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
