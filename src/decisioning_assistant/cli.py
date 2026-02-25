from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from common.io_utils import read_yaml
from common.logging_utils import get_logger

logger = get_logger("decisioning_assistant.cli")


def _resolve_path(path: str, project_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((project_root / candidate).resolve())


def _run(cmd: Sequence[str], project_root: Path) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


def _load_sources_cfg(path: str, project_root: Path) -> dict:
    cfg_path = _resolve_path(path, project_root)
    return read_yaml(cfg_path)


def cmd_ingest(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    cfg = _load_sources_cfg(args.sources_config, project_root)

    pdf_cfg = cfg.get("pdf", {}) if isinstance(cfg.get("pdf"), dict) else {}
    webex_cfg = cfg.get("webex", {}) if isinstance(cfg.get("webex"), dict) else {}

    pdf_input = args.pdf_input_dir or str(pdf_cfg.get("input_dir", "data/raw/pdf"))
    pdf_output = args.pdf_output or str(
        pdf_cfg.get("output_jsonl", "data/staging/documents/pdf_documents.jsonl")
    )
    webex_input = args.webex_input_dir or str(webex_cfg.get("raw_dir", "data/raw/webex"))
    webex_output = args.webex_output or str(
        webex_cfg.get("output_jsonl", "data/staging/documents/webex_documents.jsonl")
    )

    if not args.skip_pdf:
        _run(
            [
                sys.executable,
                "-m",
                "ingestion.ingest_pdfs",
                "--input-dir",
                _resolve_path(pdf_input, project_root),
                "--output",
                _resolve_path(pdf_output, project_root),
            ],
            project_root,
        )

    if not args.skip_webex:
        _run(
            [
                sys.executable,
                "-m",
                "ingestion.ingest_webex",
                "--input-dir",
                _resolve_path(webex_input, project_root),
                "--output",
                _resolve_path(webex_output, project_root),
            ],
            project_root,
        )

    if not args.skip_normalize:
        _run(
            [
                sys.executable,
                "-m",
                "ingestion.normalize_docs",
                "--config",
                _resolve_path(args.sources_config, project_root),
            ],
            project_root,
        )


def cmd_qa(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    qa_config = _resolve_path(args.qa_config, project_root)
    models_config = _resolve_path(args.models_config, project_root)

    if not args.skip_generate:
        _run(
            [
                sys.executable,
                "-m",
                "qa.generate_qa",
                "--qa-config",
                qa_config,
                "--models-config",
                models_config,
            ],
            project_root,
        )

    if not args.skip_validate:
        _run(
            [
                sys.executable,
                "-m",
                "qa.validate_qa",
                "--qa-config",
                qa_config,
            ],
            project_root,
        )

    if not args.skip_split:
        _run(
            [
                sys.executable,
                "-m",
                "qa.split_dataset",
                "--qa-config",
                qa_config,
            ],
            project_root,
        )


def cmd_finetune(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "training.run_lora",
        "--config",
        _resolve_path(args.finetune_config, project_root),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, project_root)


def cmd_rag_index(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "rag.build_index",
        "--config",
        _resolve_path(args.rag_config, project_root),
    ]
    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.recreate:
        cmd.append("--recreate")
    _run(cmd, project_root)


def cmd_app(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()

    from rag import assistant_app

    app_path = Path(assistant_app.__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]

    if args.server_port:
        cmd.extend(["--server.port", str(args.server_port)])
    if args.server_address:
        cmd.extend(["--server.address", args.server_address])
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    _run(cmd, project_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decisioning-assistant",
        description=(
            "CLI for ingestion, QA generation, fine-tuning, RAG indexing, and RAG app launch."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (defaults to current working directory).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingestion pipeline (PDF + Webex + normalization).",
    )
    ingest_parser.add_argument("--sources-config", default="configs/sources.yaml")
    ingest_parser.add_argument("--pdf-input-dir", default="")
    ingest_parser.add_argument("--pdf-output", default="")
    ingest_parser.add_argument("--webex-input-dir", default="")
    ingest_parser.add_argument("--webex-output", default="")
    ingest_parser.add_argument("--skip-pdf", action="store_true")
    ingest_parser.add_argument("--skip-webex", action="store_true")
    ingest_parser.add_argument("--skip-normalize", action="store_true")
    ingest_parser.set_defaults(func=cmd_ingest)

    qa_parser = subparsers.add_parser(
        "qa",
        help="Run QA pipeline (generate + validate + split).",
    )
    qa_parser.add_argument("--qa-config", default="configs/qa_generation.yaml")
    qa_parser.add_argument("--models-config", default="configs/models.yaml")
    qa_parser.add_argument("--skip-generate", action="store_true")
    qa_parser.add_argument("--skip-validate", action="store_true")
    qa_parser.add_argument("--skip-split", action="store_true")
    qa_parser.set_defaults(func=cmd_qa)

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Run MLX LoRA fine-tuning.",
    )
    finetune_parser.add_argument("--finetune-config", default="configs/finetune.yaml")
    finetune_parser.add_argument("--dry-run", action="store_true")
    finetune_parser.set_defaults(func=cmd_finetune)

    rag_index_parser = subparsers.add_parser(
        "rag-index",
        help=(
            "Build RAG vector index. By default, upserts into existing collection; "
            "use --recreate to rebuild from scratch."
        ),
    )
    rag_index_parser.add_argument("--rag-config", default="configs/rag.yaml")
    rag_index_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override indexing batch size (0 uses rag.yaml index_batch_size).",
    )
    rag_index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection before indexing.",
    )
    rag_index_parser.set_defaults(func=cmd_rag_index)

    app_parser = subparsers.add_parser(
        "app",
        help="Start the Streamlit RAG chat application.",
    )
    app_parser.add_argument("--server-port", type=int, default=0)
    app_parser.add_argument("--server-address", default="")
    app_parser.add_argument("--headless", action="store_true")
    app_parser.set_defaults(func=cmd_app)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        logger.error("Command failed with exit code %s", exc.returncode)
        return exc.returncode
    except Exception as exc:  # noqa: BLE001
        logger.exception("CLI failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
