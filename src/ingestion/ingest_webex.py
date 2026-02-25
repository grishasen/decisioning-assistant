from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_json, write_jsonl
from common.logging_utils import get_logger
from common.schemas import DocumentRecord
from common.text_utils import normalize_whitespace, stable_id

logger = get_logger(__name__)


MESSAGE_KEYS = ("messages", "items", "posts", "results", "thread", "replies")


def _coerce_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _message_text(msg: dict[str, Any]) -> str:
    for key in ("markdown", "text", "body", "message"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    return ""


def _looks_like_message(obj: dict[str, Any]) -> bool:
    has_id = isinstance(obj.get("id"), str)
    has_text = bool(_message_text(obj))
    return has_id and has_text


def _iter_message_candidates(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_message_candidates(item)
        return

    if not isinstance(payload, dict):
        return

    if _looks_like_message(payload):
        yield payload

    for key in MESSAGE_KEYS:
        value = payload.get(key)
        if value is not None:
            yield from _iter_message_candidates(value)


def parse_dump_file(path: Path) -> list[DocumentRecord]:
    payload: Any
    if path.suffix.lower() == ".jsonl":
        payload = list(iter_jsonl(path))
    else:
        payload = read_json(path)

    records: list[DocumentRecord] = []
    for msg in _iter_message_candidates(payload):
        text = _message_text(msg)
        if not text:
            continue

        msg_id = msg.get("id", "")
        room_id = msg.get("roomId") or msg.get("spaceId") or "unknown-room"
        room_title = msg.get("roomTitle") or msg.get("spaceTitle") or "Webex Space"
        source_ref = f"webex::{room_id}#{msg_id}"

        record = DocumentRecord(
            doc_id=stable_id("webex", source_ref),
            source_type="webex",
            source_ref=source_ref,
            source_path=str(path),
            title=room_title,
            text=text,
            markdown=msg.get("markdown") if isinstance(msg.get("markdown"), str) else None,
            created_at=_coerce_datetime(msg.get("created")),
            metadata={
                "room_id": room_id,
                "room_title": room_title,
                "message_id": msg_id,
                "parent_id": msg.get("parentId"),
                "person_email": msg.get("personEmail"),
                "person_name": msg.get("personDisplayName") or msg.get("displayName"),
                "files": msg.get("files", []),
            },
        )
        records.append(record)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Webex archive dumps into normalized JSONL docs.")
    parser.add_argument("--input-dir", default="data/raw/webex")
    parser.add_argument("--output", default="data/staging/documents/webex_documents.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        ]
    )
    if not files:
        logger.warning("No JSON files found in %s", input_dir)
        write_jsonl(args.output, [])
        return

    all_rows: list[dict[str, Any]] = []
    for path in tqdm(files, desc="Parsing Webex dumps"):
        try:
            records = parse_dump_file(path)
            all_rows.extend(record.model_dump(mode="json") for record in records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse %s: %s", path, exc)

    count = write_jsonl(args.output, all_rows)
    logger.info("Wrote %s Webex documents to %s", count, args.output)


if __name__ == "__main__":
    main()
