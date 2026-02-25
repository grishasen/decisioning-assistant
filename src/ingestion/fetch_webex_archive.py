from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper that invokes a locally installed webexspacearchive command for one or more spaces. "
            "Use this if you prefer the archive tool for thread export."
        )
    )
    parser.add_argument(
        "--space-id",
        action="append",
        required=True,
        help="Webex space ID; repeat this flag for multiple spaces.",
    )
    parser.add_argument("--output-dir", default="data/raw/webex")
    parser.add_argument(
        "--command-template",
        default="webex-space-archive.py {space_id}",
        help=(
            "Command template to run. Must include {space_id}. "
            "Example: 'python webex-space-archive.py custom.ini {space_id}'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("WEBEX_ACCESS_TOKEN") or os.getenv("WEBEX_ARCHIVE_TOKEN")
    if token:
        logger.info("Webex token detected in environment.")
    else:
        logger.warning(
            "No WEBEX_ACCESS_TOKEN/WEBEX_ARCHIVE_TOKEN detected. "
            "If your archive tool requires one, export it before running."
        )

    for space_id in args.space_id:
        command = args.command_template.format(space_id=space_id)
        argv = shlex.split(command)

        logger.info("Running archive command for space: %s", space_id)
        try:
            subprocess.run(argv, cwd=output_dir, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Archive command not found. Install webexspacearchive and/or adjust --command-template."
            ) from exc

    logger.info("Webex archive command completed. Raw files should now exist under %s", output_dir)


if __name__ == "__main__":
    main()
