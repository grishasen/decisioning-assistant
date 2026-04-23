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
    """Signature: def _resolve_path(path: str, project_root: Path) -> str.

    Resolve a config or data path relative to the project root.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((project_root / candidate).resolve())


def _run(cmd: Sequence[str], project_root: Path) -> None:
    """Signature: def _run(cmd: Sequence[str], project_root: Path) -> None.

    Run a subprocess command from the project root.
    """
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)


def _load_sources_cfg(path: str, project_root: Path) -> dict:
    """Signature: def _load_sources_cfg(path: str, project_root: Path) -> dict.

    Load the sources configuration file for ingestion commands.
    """
    cfg_path = _resolve_path(path, project_root)
    return read_yaml(cfg_path)


def _pick_metadata_value(override: str, cfg: dict, key: str) -> str:
    """Signature: def _pick_metadata_value(override: str, cfg: dict, key: str) -> str.

    Pick a metadata value from CLI overrides or the loaded config.
    """
    if isinstance(override, str) and override.strip():
        return override.strip()

    raw = cfg.get(key)
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _append_if_value(cmd: list[str], flag: str, value: str) -> None:
    """Signature: def _append_if_value(cmd: list[str], flag: str, value: str) -> None.

    Append a flag and value to a command when the value is non-empty.
    """
    cleaned = value.strip()
    if cleaned:
        cmd.extend([flag, cleaned])


def cmd_ingest(args: argparse.Namespace) -> None:
    """Signature: def cmd_ingest(args: argparse.Namespace) -> None.

    Run the ingest CLI command.
    """
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

    pdf_product = _pick_metadata_value(args.pdf_product, pdf_cfg, "product")
    pdf_doc_version = _pick_metadata_value(args.pdf_doc_version, pdf_cfg, "doc_version")
    pdf_doc_type = _pick_metadata_value(args.pdf_doc_type, pdf_cfg, "doc_type")

    webex_product = _pick_metadata_value(args.webex_product, webex_cfg, "product")
    webex_doc_version = _pick_metadata_value(args.webex_doc_version, webex_cfg, "doc_version")
    webex_doc_type = _pick_metadata_value(args.webex_doc_type, webex_cfg, "doc_type")
    webex_group_by_thread = bool(webex_cfg.get("include_threads", True))

    if not args.skip_pdf:
        pdf_cmd = [
            sys.executable,
            "-m",
            "ingestion.ingest_pdfs",
            "--input-dir",
            _resolve_path(pdf_input, project_root),
            "--output",
            _resolve_path(pdf_output, project_root),
        ]
        _append_if_value(pdf_cmd, "--product", pdf_product)
        _append_if_value(pdf_cmd, "--doc-version", pdf_doc_version)
        _append_if_value(pdf_cmd, "--doc-type", pdf_doc_type)
        _run(pdf_cmd, project_root)

    if not args.skip_webex:
        webex_cmd = [
            sys.executable,
            "-m",
            "ingestion.ingest_webex",
            "--input-dir",
            _resolve_path(webex_input, project_root),
            "--output",
            _resolve_path(webex_output, project_root),
        ]
        if not webex_group_by_thread:
            webex_cmd.append("--no-group-by-thread")
        _append_if_value(webex_cmd, "--product", webex_product)
        _append_if_value(webex_cmd, "--doc-version", webex_doc_version)
        _append_if_value(webex_cmd, "--doc-type", webex_doc_type)
        _run(webex_cmd, project_root)

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
    """Signature: def cmd_qa(args: argparse.Namespace) -> None.

    Run the qa CLI command.
    """
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
    """Signature: def cmd_finetune(args: argparse.Namespace) -> None.

    Run the finetune CLI command.
    """
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
    """Signature: def cmd_rag_index(args: argparse.Namespace) -> None.

    Run the rag index CLI command.
    """
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


def cmd_rag_eval_retrieval(args: argparse.Namespace) -> None:
    """Signature: def cmd_rag_eval_retrieval(args: argparse.Namespace) -> None.

    Run the rag eval retrieval CLI command.
    """
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "rag.eval_retrieval",
        "--rag-config",
        _resolve_path(args.rag_config, project_root),
        "--eval-path",
        _resolve_path(args.eval_path, project_root),
        "--output-path",
        _resolve_path(args.output_path, project_root),
    ]
    if args.top_k > 0:
        cmd.extend(["--top-k", str(args.top_k)])
    _run(cmd, project_root)


def cmd_rag_eval_answering(args: argparse.Namespace) -> None:
    """Signature: def cmd_rag_eval_answering(args: argparse.Namespace) -> None.

    Run the rag eval answering CLI command.
    """
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "rag.eval_answering",
        "--rag-config",
        _resolve_path(args.rag_config, project_root),
        "--models-config",
        _resolve_path(args.models_config, project_root),
        "--eval-path",
        _resolve_path(args.eval_path, project_root),
        "--output-path",
        _resolve_path(args.output_path, project_root),
    ]
    if args.top_k > 0:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.max_cases > 0:
        cmd.extend(["--max-cases", str(args.max_cases)])
    if args.adapter_path:
        cmd.extend(["--adapter-path", args.adapter_path])
    _run(cmd, project_root)


def cmd_rag_export(args: argparse.Namespace) -> None:
    """Signature: def cmd_rag_export(args: argparse.Namespace) -> None.

    Run the rag export CLI command.
    """
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "rag.export_index",
        "--config",
        _resolve_path(args.rag_config, project_root),
        "--output-dir",
        _resolve_path(args.output_dir, project_root),
    ]
    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])
    for source in args.source:
        cmd.extend(["--source", source])
    _run(cmd, project_root)


def cmd_rag_import(args: argparse.Namespace) -> None:
    """Signature: def cmd_rag_import(args: argparse.Namespace) -> None.

    Run the rag import CLI command.
    """
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "rag.import_index",
        "--input-dir",
        _resolve_path(args.input_dir, project_root),
        "--config",
        _resolve_path(args.rag_config, project_root),
    ]
    if args.qdrant_path:
        cmd.extend(["--qdrant-path", _resolve_path(args.qdrant_path, project_root)])
    if args.collection_name:
        cmd.extend(["--collection-name", args.collection_name])
    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.recreate:
        cmd.append("--recreate")
    _run(cmd, project_root)


def cmd_webex_fetch(args: argparse.Namespace) -> None:
    """Signature: def cmd_webex_fetch(args: argparse.Namespace) -> None.

    Run the webex fetch CLI command.
    """
    project_root = Path(args.project_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "ingestion.fetch_webex_archive",
        "--rooms-json",
        _resolve_path(args.rooms_json, project_root),
        "--config",
        _resolve_path(args.config, project_root),
        "--output-dir",
        _resolve_path(args.output_dir, project_root),
        "--room-type",
        args.room_type,
    ]
    if args.page_size > 0:
        cmd.extend(["--page-size", str(args.page_size)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    _run(cmd, project_root)


def cmd_app(args: argparse.Namespace) -> None:
    """Signature: def cmd_app(args: argparse.Namespace) -> None.

    Run the app CLI command.
    """
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


def cmd_turboquant_convert(args: argparse.Namespace) -> None:
    """Convert a HuggingFace or local MLX model to TurboQuant MLX format."""
    project_root = Path(args.project_root).resolve()
    hf_path = str(args.hf_path or "").strip()
    if not hf_path:
        models_config = _resolve_path(args.models_config, project_root)
        models_cfg = read_yaml(models_config)
        model_cfg = models_cfg.get(args.model_key, {})
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Model config key '{args.model_key}' is not a mapping")
        hf_path = str(
            model_cfg.get("base_model")
            or model_cfg.get("source_model")
            or model_cfg.get("model")
            or ""
        ).strip()

    if not hf_path:
        raise ValueError(
            "No source model found. Pass --hf-path or set model/base_model in "
            f"{args.model_key}."
        )

    try:
        from turboquant_mlx.convert import convert
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "turboquant_mlx is required for TurboQuant conversion. Install with "
            "`pip install -e .[turboquant]` or `pip install turboquant-mlx-full`."
        ) from exc

    mlx_path = _resolve_path(args.mlx_path, project_root)
    convert(
        hf_path=hf_path,
        mlx_path=mlx_path,
        bits=args.bits,
        group_size=args.group_size,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        fuse_rotations=args.fuse_rotations,
        use_qjl=args.use_qjl,
        dtype=args.dtype or None,
    )


def build_parser() -> argparse.ArgumentParser:
    """Signature: def build_parser() -> argparse.ArgumentParser.

    Build the top-level DecisioningAssistant CLI parser.
    """
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
    ingest_parser.add_argument("--pdf-product", default="")
    ingest_parser.add_argument("--pdf-doc-version", default="")
    ingest_parser.add_argument("--pdf-doc-type", default="")
    ingest_parser.add_argument("--webex-input-dir", default="")
    ingest_parser.add_argument("--webex-output", default="")
    ingest_parser.add_argument("--webex-product", default="")
    ingest_parser.add_argument("--webex-doc-version", default="")
    ingest_parser.add_argument("--webex-doc-type", default="")
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
            "Build hybrid RAG vector index (document chunks + QA pairs). By default, upserts into existing collection; "
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

    rag_eval_retrieval_parser = subparsers.add_parser(
        "rag-eval-retrieval",
        help="Evaluate retrieval ranking on a labeled JSONL query set.",
    )
    rag_eval_retrieval_parser.add_argument("--rag-config", default="configs/rag.yaml")
    rag_eval_retrieval_parser.add_argument("--eval-path", default="configs/rag_eval.sample.jsonl")
    rag_eval_retrieval_parser.add_argument("--top-k", type=int, default=0)
    rag_eval_retrieval_parser.add_argument(
        "--output-path",
        default="data/eval/reports/retrieval_report.json",
    )
    rag_eval_retrieval_parser.set_defaults(func=cmd_rag_eval_retrieval)

    rag_eval_answering_parser = subparsers.add_parser(
        "rag-eval-answering",
        help="Run end-to-end answer evaluation on a labeled JSONL query set.",
    )
    rag_eval_answering_parser.add_argument("--rag-config", default="configs/rag.yaml")
    rag_eval_answering_parser.add_argument("--models-config", default="configs/models.yaml")
    rag_eval_answering_parser.add_argument("--eval-path", default="configs/rag_eval.sample.jsonl")
    rag_eval_answering_parser.add_argument("--adapter-path", default="")
    rag_eval_answering_parser.add_argument("--top-k", type=int, default=0)
    rag_eval_answering_parser.add_argument("--max-cases", type=int, default=0)
    rag_eval_answering_parser.add_argument(
        "--output-path",
        default="data/eval/reports/answering_report.json",
    )
    rag_eval_answering_parser.set_defaults(func=cmd_rag_eval_answering)

    rag_export_parser = subparsers.add_parser(
        "rag-export",
        help="Export local RAG collection into a portable bundle for transfer.",
    )
    rag_export_parser.add_argument("--rag-config", default="configs/rag.yaml")
    rag_export_parser.add_argument("--output-dir", default="data/rag/export")
    rag_export_parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Override export batch size.",
    )
    rag_export_parser.add_argument(
        "--source",
        action="append",
        choices=["pdf", "webex"],
        default=[],
        help="Export only selected source type(s). Repeat for multiple values.",
    )
    rag_export_parser.set_defaults(func=cmd_rag_export)

    rag_import_parser = subparsers.add_parser(
        "rag-import",
        help="Import a portable RAG bundle into local Qdrant.",
    )
    rag_import_parser.add_argument("--rag-config", default="configs/rag.yaml")
    rag_import_parser.add_argument("--input-dir", default="data/rag/export")
    rag_import_parser.add_argument(
        "--qdrant-path",
        default="",
        help="Override destination Qdrant path.",
    )
    rag_import_parser.add_argument(
        "--collection-name",
        default="",
        help="Override destination collection name.",
    )
    rag_import_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Override import batch size.",
    )
    rag_import_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate destination collection before import.",
    )
    rag_import_parser.set_defaults(func=cmd_rag_import)

    webex_fetch_parser = subparsers.add_parser(
        "webex-fetch",
        help="Fetch raw Webex room messages directly from the Webex API.",
    )
    webex_fetch_parser.add_argument("--rooms-json", required=True)
    webex_fetch_parser.add_argument("--config", required=True)
    webex_fetch_parser.add_argument("--output-dir", default="data/raw/webex")
    webex_fetch_parser.add_argument(
        "--room-type",
        choices=["group", "direct", "all"],
        default="group",
        help="Which room types from rooms.json to archive (default: group).",
    )
    webex_fetch_parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Messages per Webex API page (default: 500).",
    )
    webex_fetch_parser.add_argument("--skip-existing", action="store_true")
    webex_fetch_parser.set_defaults(func=cmd_webex_fetch)

    app_parser = subparsers.add_parser(
        "app",
        help="Start the Streamlit RAG chat application.",
    )
    app_parser.add_argument("--server-port", type=int, default=0)
    app_parser.add_argument("--server-address", default="")
    app_parser.add_argument("--headless", action="store_true")
    app_parser.set_defaults(func=cmd_app)

    tq_parser = subparsers.add_parser(
        "turboquant-convert",
        help="Convert a HuggingFace or local model to TurboQuant MLX format.",
    )
    tq_parser.add_argument(
        "--hf-path",
        default="",
        help=(
            "Source HuggingFace repo or local model path. If omitted, the model "
            "is read from --models-config and --model-key."
        ),
    )
    tq_parser.add_argument("--models-config", default="configs/models.yaml")
    tq_parser.add_argument("--model-key", default="answer_model")
    tq_parser.add_argument(
        "--mlx-path",
        required=True,
        help="Output directory for the converted TurboQuant model.",
    )
    tq_parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    tq_parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        choices=[32, 64, 128],
    )
    tq_parser.add_argument(
        "--rotation",
        default="hadamard",
        choices=["hadamard", "blockwise_hadamard", "none"],
    )
    tq_parser.add_argument("--rotation-seed", type=int, default=42)
    tq_parser.add_argument(
        "--fuse-rotations",
        action="store_true",
        help="Fuse eligible rotations into norms where TurboQuant supports it.",
    )
    tq_parser.add_argument(
        "--use-qjl",
        action="store_true",
        help="Enable QJL residual correction during conversion.",
    )
    tq_parser.add_argument(
        "--dtype",
        default="",
        choices=["", "float16", "bfloat16", "float32"],
        help="Optional dtype before quantization.",
    )
    tq_parser.set_defaults(func=cmd_turboquant_convert)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Signature: def main(argv: Sequence[str] | None = None) -> int.

    Run the DecisioningAssistant CLI entrypoint.
    """
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
