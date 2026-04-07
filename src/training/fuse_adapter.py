from __future__ import annotations

import argparse
import subprocess

from common.io_utils import read_yaml
from common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for fuse adapter.
    """
    parser = argparse.ArgumentParser(description="Fuse MLX LoRA adapter into a standalone model.")
    parser.add_argument("--config", default="configs/finetune.yaml")
    return parser.parse_args()


def main() -> None:
    """Signature: def main() -> None.

    Run the fuse adapter entrypoint.
    """
    args = parse_args()
    cfg = read_yaml(args.config)

    model = str(cfg.get("model"))
    adapter_path = str(cfg.get("adapter_path"))
    fused_path = str(cfg.get("fused_model_path", "data/models/gemma_lora_fused"))

    cmd = [
        "mlx_lm.fuse",
        "--model",
        model,
        "--adapter-path",
        adapter_path,
        "--save-path",
        fused_path,
    ]

    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
