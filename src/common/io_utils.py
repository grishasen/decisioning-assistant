from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with Path(path).expanduser().open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent_dir(path)
    count = 0
    with Path(path).expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent_dir(path)
    count = 0
    with Path(path).expanduser().open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def repair_jsonl_tail(path: str | Path) -> int:
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
