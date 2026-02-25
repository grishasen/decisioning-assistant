from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_json, write_jsonl
from common.logging_utils import get_logger
from common.schemas import DocumentRecord, build_metadata, normalize_doc_type
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


def _message_author(msg: dict[str, Any]) -> str:
    for key in ("personDisplayName", "displayName", "personEmail"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown-user"


def _message_id(msg: dict[str, Any]) -> str:
    value = msg.get("id")
    return value.strip() if isinstance(value, str) else ""


def _thread_id(msg: dict[str, Any]) -> str:
    parent_id = msg.get("parentId")
    if isinstance(parent_id, str) and parent_id.strip():
        return parent_id.strip()
    return _message_id(msg)


def _room_id(msg: dict[str, Any]) -> str:
    for key in ("roomId", "spaceId"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown-room"


def _room_title_from_path(path: Path) -> str:
    stem = normalize_whitespace(path.stem)
    return stem or "Webex Space"


def _sorted_thread_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        messages,
        key=lambda item: (
            item.get("created") is None,
            item.get("created") or datetime.min,
            item.get("message_id") or "",
        ),
    )


def _build_thread_records(
    path: Path,
    payload: Any,
    product: str | None,
    doc_version: str | None,
    doc_type: str | None,
    ingested_at: datetime,
) -> list[DocumentRecord]:
    room_title_from_file = _room_title_from_path(path)
    parsed_messages: list[dict[str, Any]] = []
    seen_messages: set[str] = set()

    for msg in _iter_message_candidates(payload):
        text = _message_text(msg)
        if not text:
            continue

        msg_id = _message_id(msg)
        if not msg_id:
            continue

        room_id = _room_id(msg)
        dedupe_key = f"{room_id}:{msg_id}"
        if dedupe_key in seen_messages:
            continue
        seen_messages.add(dedupe_key)

        parent_value = msg.get("parentId")
        parent_id = parent_value.strip() if isinstance(parent_value, str) and parent_value.strip() else None

        parsed_messages.append(
            {
                "message_id": msg_id,
                "room_id": room_id,
                "author": _message_author(msg),
                "person_email": msg.get("personEmail"),
                "created": _coerce_datetime(msg.get("created")),
                "updated": _coerce_datetime(msg.get("updated")),
                "text": text,
                "markdown": msg.get("markdown") if isinstance(msg.get("markdown"), str) else None,
                "files": msg.get("files", []),
                "parent_id": parent_id,
            }
        )

    roots: list[dict[str, Any]] = []
    replies_by_root: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for item in parsed_messages:
        parent_id = item.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            room_id = str(item.get("room_id") or "unknown-room")
            replies_by_root.setdefault((room_id, parent_id), []).append(item)
        else:
            roots.append(item)

    ordered_roots = sorted(
        roots,
        key=lambda item: (
            item.get("room_id") or "",
            item.get("created") is None,
            item.get("created") or datetime.min,
            item.get("message_id") or "",
        ),
    )

    records: list[DocumentRecord] = []
    for root in ordered_roots:
        room_id = str(root.get("room_id") or "unknown-room")
        thread_id = str(root.get("message_id") or "")
        if not thread_id:
            continue

        replies = _sorted_thread_messages(replies_by_root.get((room_id, thread_id), []))
        ordered = [root, *replies]

        room_title = room_title_from_file
        file_name = room_title_from_file
        lines: list[str] = []
        markdown_lines: list[str] = []
        message_ids: list[str] = []
        authors: set[str] = set()
        files: list[str] = []

        for item in ordered:
            created = item.get("created")
            author = str(item.get("author") or "unknown-user")
            ts = created.isoformat() if isinstance(created, datetime) else "unknown-time"
            message_text = str(item.get("text") or "").strip()
            if message_text:
                lines.append(f"[{ts}] {author}: {message_text}")

            markdown = item.get("markdown")
            if isinstance(markdown, str) and markdown.strip():
                markdown_lines.append(f"[{ts}] {author}: {markdown.strip()}")

            msg_id = str(item.get("message_id") or "")
            if msg_id:
                message_ids.append(msg_id)
            authors.add(author)

            msg_files = item.get("files")
            if isinstance(msg_files, list):
                files.extend(str(x) for x in msg_files)

        thread_text = "\n".join(lines).strip()
        if not thread_text:
            continue

        timestamps: list[datetime] = []
        for item in ordered:
            updated = item.get("updated")
            created = item.get("created")
            if isinstance(updated, datetime):
                timestamps.append(updated)
            elif isinstance(created, datetime):
                timestamps.append(created)

        first_created = root.get("created")
        thread_updated = max(timestamps) if timestamps else (first_created if isinstance(first_created, datetime) else None)

        source_ref = f"webex::{room_id}#thread={thread_id}"
        metadata = build_metadata(
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            room_id=room_id,
            room_title=room_title,
            thread_id=thread_id,
            message_count=len(ordered),
            created_at=first_created if isinstance(first_created, datetime) else None,
            updated_at=thread_updated,
            ingested_at=ingested_at,
        )
        metadata.update(
            {
                "file_name": file_name,
                "message_ids": message_ids,
                "authors": sorted(authors),
                "files": files,
                "is_thread_document": True,
                "webex_grouping": "thread",
            }
        )

        record = DocumentRecord(
            doc_id=stable_id("webex-thread", str(path.resolve()), room_id, thread_id),
            source_type="webex",
            source_ref=source_ref,
            source_path=str(path),
            title=room_title,
            text=thread_text,
            markdown="\n".join(markdown_lines).strip() or None,
            created_at=first_created if isinstance(first_created, datetime) else None,
            metadata=metadata,
        )
        records.append(record)

    return records


def _build_message_records(
    path: Path,
    payload: Any,
    product: str | None,
    doc_version: str | None,
    doc_type: str | None,
    ingested_at: datetime,
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    room_title_from_file = _room_title_from_path(path)
    for msg in _iter_message_candidates(payload):
        text = _message_text(msg)
        if not text:
            continue

        msg_id = _message_id(msg)
        room_id = _room_id(msg)
        room_title = room_title_from_file
        file_name = room_title_from_file
        thread_id = _thread_id(msg)
        source_ref = f"webex::{room_id}#{msg_id}"

        created_at = _coerce_datetime(msg.get("created"))
        updated_at = _coerce_datetime(msg.get("updated")) or created_at

        metadata = build_metadata(
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            room_id=room_id,
            room_title=room_title,
            thread_id=thread_id,
            message_count=1,
            created_at=created_at,
            updated_at=updated_at,
            ingested_at=ingested_at,
        )
        metadata.update(
            {
                "file_name": file_name,
                "message_id": msg_id,
                "parent_id": msg.get("parentId"),
                "person_email": msg.get("personEmail"),
                "person_name": msg.get("personDisplayName") or msg.get("displayName"),
                "files": msg.get("files", []),
                "is_thread_document": False,
                "webex_grouping": "message",
            }
        )

        record = DocumentRecord(
            doc_id=stable_id("webex", source_ref),
            source_type="webex",
            source_ref=source_ref,
            source_path=str(path),
            title=room_title,
            text=text,
            markdown=msg.get("markdown") if isinstance(msg.get("markdown"), str) else None,
            created_at=created_at,
            metadata=metadata,
        )
        records.append(record)

    return records


def parse_dump_file(
    path: Path,
    group_by_thread: bool = True,
    product: str | None = None,
    doc_version: str | None = None,
    doc_type: str | None = None,
    ingested_at: datetime | None = None,
) -> list[DocumentRecord]:
    payload: Any
    if path.suffix.lower() == ".jsonl":
        payload = list(iter_jsonl(path))
    else:
        payload = read_json(path)

    normalized_ingested_at = ingested_at or datetime.now(timezone.utc)

    if group_by_thread:
        return _build_thread_records(
            path=path,
            payload=payload,
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            ingested_at=normalized_ingested_at,
        )

    return _build_message_records(
        path=path,
        payload=payload,
        product=product,
        doc_version=doc_version,
        doc_type=doc_type,
        ingested_at=normalized_ingested_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Webex archive dumps into normalized JSONL docs.")
    parser.add_argument("--input-dir", default="data/raw/webex")
    parser.add_argument("--output", default="data/staging/documents/webex_documents.jsonl")
    parser.add_argument(
        "--group-by-thread",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group Webex messages by thread (default: true).",
    )
    parser.add_argument("--product", default="", help="Product label for indexed metadata.")
    parser.add_argument("--doc-version", default="", help="Document version label.")
    parser.add_argument(
        "--doc-type",
        default="",
        help="Document type metadata (guide, api, release-note).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    product = args.product.strip() or None
    doc_version = args.doc_version.strip() or None
    doc_type = normalize_doc_type(args.doc_type)
    ingested_at = datetime.now(timezone.utc)

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
            records = parse_dump_file(
                path=path,
                group_by_thread=args.group_by_thread,
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
                ingested_at=ingested_at,
            )
            all_rows.extend(record.model_dump(mode="json") for record in records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse %s: %s", path, exc)

    count = write_jsonl(args.output, all_rows)
    logger.info("Wrote %s Webex documents to %s", count, args.output)


if __name__ == "__main__":
    main()
