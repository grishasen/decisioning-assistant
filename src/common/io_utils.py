from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_parent_dir(path: str | Path) -> None:
    """Signature: def ensure_parent_dir(path: str | Path) -> None.

    Create the parent directory for a file path when needed.
    """
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Signature: def read_yaml(path: str | Path) -> dict[str, Any].

    Load a YAML mapping from disk.
    """
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def read_json(path: str | Path) -> Any:
    """Signature: def read_json(path: str | Path) -> Any.

    Load JSON data from disk.
    """
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    """Signature: def write_json(path: str | Path, payload: Any) -> None.

    Write JSON data to disk with UTF-8 encoding.
    """
    ensure_parent_dir(path)
    with Path(path).expanduser().open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Signature: def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]].

    Yield parsed JSON objects from a JSONL file.
    """
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_iter_jsonl(path: str | Path) -> int:
    """Signature: def count_iter_jsonl(path: str | Path) -> int.

    Count non-empty rows in a JSONL file.
    """
    count = 0
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Signature: def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int.

    Write JSONL rows to disk and return the number written.
    """
    ensure_parent_dir(path)
    count = 0
    with Path(path).expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Signature: def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int.

    Append JSONL rows to disk and return the number written.
    """
    ensure_parent_dir(path)
    count = 0
    with Path(path).expanduser().open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def repair_jsonl_tail(path: str | Path) -> int:
    """Signature: def repair_jsonl_tail(path: str | Path) -> int.

    Truncate malformed trailing JSONL lines and return how many were dropped.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return 0

    valid_lines: list[str] = []
    dropped_lines = 0
    corrupted = False
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if corrupted:
                dropped_lines += 1
                continue
            try:
                json.loads(line)
            except JSONDecodeError:
                corrupted = True
                dropped_lines += 1
                continue
            valid_lines.append(line)

    if corrupted:
        ensure_parent_dir(file_path)
        with file_path.open("w", encoding="utf-8") as handle:
            for line in valid_lines:
                handle.write(line + "\n")

    return dropped_lines
