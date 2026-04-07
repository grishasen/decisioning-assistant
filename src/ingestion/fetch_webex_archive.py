from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common.io_utils import read_yaml
from common.logging_utils import get_logger
from common.webex_utils import parse_webex_datetime

logger = get_logger(__name__)

API_BASE_URL = "https://webexapis.com/v1"
DEFAULT_PAGE_SIZE = 500
DEFAULT_TOTAL_LIMIT = 10000
FILENAME_MAX_CHARS = 80
ALLOWED_CONFIG_KEYS = {"token", "max_total_messages"}
DAYS_SPEC_RE = re.compile(r"^(?P<days>\d+)d$", re.IGNORECASE)
FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


@dataclass(frozen=True)
class FetchPolicy:
    """Store time bounds and message limits for a Webex fetch run."""
    after: datetime | None = None
    before: datetime | None = None
    total_limit: int | None = DEFAULT_TOTAL_LIMIT
    sort_old_new: bool = True


@dataclass(frozen=True)
class RoomSpec:
    """Represent a Webex room selected for archiving."""
    room_id: str
    title: str


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for fetch webex archive.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fetch raw Webex space messages directly through the Webex REST API. "
            "Takes a rooms.json export and a YAML settings file, then writes one JSON file per room."
        )
    )
    parser.add_argument(
        "--rooms-json",
        required=True,
        help="Path to a rooms.json file returned by the Webex rooms API.",
    )
    parser.add_argument(
        "--config",
        "--yaml-config",
        dest="config_path",
        required=True,
        help="YAML file with Webex token and fetch settings.",
    )
    parser.add_argument("--output-dir", default="data/raw/webex")
    parser.add_argument(
        "--room-type",
        choices=["group", "direct", "all"],
        default="group",
        help="Which room types from rooms.json to archive (default: group).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Number of messages to request per Webex API page (default: 500).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rooms whose output file already exists.",
    )
    return parser.parse_args()


def _parse_webex_timestamp(value: Any) -> datetime | None:
    """Signature: def _parse_webex_timestamp(value: Any) -> datetime | None.

    Parse webex timestamp.
    """
    return parse_webex_datetime(value)


def _format_webex_timestamp(value: datetime) -> str:
    """Signature: def _format_webex_timestamp(value: datetime) -> str.

    Format webex timestamp.
    """
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _parse_compact_date(value: str) -> datetime:
    """Signature: def _parse_compact_date(value: str) -> datetime.

    Parse compact date.
    """
    parsed = datetime.strptime(value, "%d%m%Y")
    return parsed.replace(tzinfo=timezone.utc)


def _parse_max_total_messages(
        raw_value: str | None,
        *,
        now: datetime | None = None,
) -> FetchPolicy:
    """Signature: def _parse_max_total_messages(raw_value: str | None, *, now: datetime | None = None) -> FetchPolicy.

    Parse max total messages.
    """
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return FetchPolicy(total_limit=DEFAULT_TOTAL_LIMIT)

    if cleaned.isdigit():
        return FetchPolicy(total_limit=max(0, int(cleaned)))

    day_match = DAYS_SPEC_RE.fullmatch(cleaned)
    if day_match:
        reference = now or datetime.now(timezone.utc)
        days = int(day_match.group("days"))
        return FetchPolicy(after=reference - timedelta(days=days), total_limit=None)

    if "-" in cleaned:
        start_raw, end_raw = cleaned.split("-", 1)
        if not start_raw:
            raise ValueError(
                "max_total_messages date range must start with ddmmyyyy, for example 01052021-11062021"
            )

        after = _parse_compact_date(start_raw)
        before = None
        if end_raw:
            before = _parse_compact_date(end_raw) + timedelta(days=1)
            if before <= after:
                raise ValueError("max_total_messages end date must be on or after the start date")

        return FetchPolicy(after=after, before=before, total_limit=None)

    raise ValueError(
        "Unsupported max_total_messages value. Use empty, an integer, Nd, or ddmmyyyy-ddmmyyyy."
    )


def _load_fetch_config(path: Path) -> dict[str, Any]:
    """Signature: def _load_fetch_config(path: Path) -> dict[str, Any].

    Load fetch config.
    """
    config = read_yaml(path)
    unknown_keys = sorted(set(config) - ALLOWED_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys in Webex fetch config {path}: {', '.join(unknown_keys)}"
        )
    return config


def _resolve_token(config: dict[str, Any]) -> str:
    """Signature: def _resolve_token(config: dict[str, Any]) -> str.

    Resolve token.
    """
    token = str(config.get("token") or "").strip()
    if token:
        return token

    token = os.getenv("WEBEX_ACCESS_TOKEN") or os.getenv("WEBEX_ARCHIVE_TOKEN") or ""
    token = token.strip()
    if token:
        return token

    raise ValueError(
        "No Webex token found. Set token in the YAML config or define WEBEX_ACCESS_TOKEN."
    )


def _load_room_specs(path: Path, room_type: str = "group") -> list[RoomSpec]:
    """Signature: def _load_room_specs(path: Path, room_type: str = 'group') -> list[RoomSpec].

    Load room specs.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload

    if not isinstance(items, list):
        raise ValueError(f"rooms.json must contain a list or an object with an items list: {path}")

    rooms: list[RoomSpec] = []
    seen_room_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        room_id = str(item.get("id") or "").strip()
        if not room_id or room_id in seen_room_ids:
            continue

        item_room_type = str(item.get("type") or "").strip().lower()
        if room_type != "all" and item_room_type != room_type:
            continue

        title = str(item.get("title") or "").strip()
        rooms.append(RoomSpec(room_id=room_id, title=title))
        seen_room_ids.add(room_id)

    return rooms


def _extract_next_link(header_value: str) -> str | None:
    """Signature: def _extract_next_link(header_value: str) -> str | None.

    Extract next link.
    """
    if not header_value:
        return None

    for part in header_value.split(","):
        match = NEXT_LINK_RE.search(part)
        if match:
            return match.group(1)
    return None


def _request_json(url: str, token: str) -> tuple[Any, dict[str, str]]:
    """Signature: def _request_json(url: str, token: str) -> tuple[Any, dict[str, str]].

    Request json.
    """
    last_error: Exception | None = None
    for attempt in range(5):
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload, dict(response.headers.items())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            retry_after = exc.headers.get("Retry-After", "").strip()
            if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                delay = float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 30)
                logger.warning(
                    "Webex API request failed with %s, retrying in %.1fs: %s",
                    exc.code,
                    delay,
                    url,
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise RuntimeError(
                f"Webex API request failed with status {exc.code} for {url}: {body[:400]}"
            ) from exc
        except URLError as exc:
            if attempt < 4:
                delay = min(2 ** attempt, 30)
                logger.warning(
                    "Network error contacting Webex API, retrying in %.1fs: %s",
                    delay,
                    exc,
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise RuntimeError(f"Could not reach Webex API: {exc}") from exc

    raise RuntimeError(f"Webex API request failed after retries: {last_error}")


def _build_messages_url(room_id: str, page_size: int, before: datetime | None) -> str:
    """Signature: def _build_messages_url(room_id: str, page_size: int, before: datetime | None) -> str.

    Build messages url.
    """
    params = [("roomId", room_id), ("max", str(page_size))]
    if before is not None:
        params.append(("before", _format_webex_timestamp(before)))
    return f"{API_BASE_URL}/messages?{urlencode(params)}"


def _fetch_room_messages(
        room_id: str,
        token: str,
        policy: FetchPolicy,
        page_size: int,
) -> list[dict[str, Any]]:
    """Signature: def _fetch_room_messages(room_id: str, token: str, policy: FetchPolicy, page_size: int) -> list[dict[str, Any]].

    Fetch room messages.
    """
    effective_page_size = max(1, min(int(page_size), 500))
    if policy.total_limit is not None and policy.total_limit <= 0:
        return []

    url = _build_messages_url(room_id, effective_page_size, policy.before)
    messages: list[dict[str, Any]] = []

    while url:
        payload, headers = _request_json(url, token)
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not items:
            break

        stop_paging = False
        for item in items:
            if not isinstance(item, dict):
                continue

            created_at = _parse_webex_timestamp(item.get("created"))
            if policy.before is not None and created_at is not None and created_at >= policy.before:
                continue
            if policy.after is not None and created_at is not None and created_at < policy.after:
                stop_paging = True
                break

            messages.append(item)
            if policy.total_limit is not None and len(messages) >= policy.total_limit:
                stop_paging = True
                break

        if stop_paging:
            break
        logger.info("Fetched messages: " + str(len(messages)))

        url = _extract_next_link(headers.get("Link") or headers.get("link") or "")

    messages.sort(
        key=lambda item: str(item.get("created") or ""),
        reverse=not policy.sort_old_new,
    )
    return messages


def _normalize_room_title(title: str) -> str:
    """Signature: def _normalize_room_title(title: str) -> str.

    Normalize room title.
    """
    cleaned = " ".join(title.split()).strip()
    return cleaned or "Webex Space"


def _build_output_basename(title: str, room_id: str, used_names: set[str]) -> str:
    """Signature: def _build_output_basename(title: str, room_id: str, used_names: set[str]) -> str.

    Build output basename.
    """
    normalized_title = _normalize_room_title(title)
    base_name = FILENAME_SANITIZE_RE.sub("_", normalized_title)
    base_name = re.sub(r"_+", "_", base_name).strip("._-") or "room"

    candidate = base_name[:FILENAME_MAX_CHARS].rstrip("._-") or "room"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = "_" + hashlib.sha1(room_id.encode("utf-8")).hexdigest()[:8]
    head_limit = max(1, FILENAME_MAX_CHARS - len(suffix))
    head = base_name[:head_limit].rstrip("._-") or "room"
    candidate = f"{head}{suffix}"
    used_names.add(candidate)
    return candidate


def _resolve_room_specs(rooms_json: str, room_type: str) -> list[RoomSpec]:
    """Signature: def _resolve_room_specs(rooms_json: str, room_type: str) -> list[RoomSpec].

    Resolve room specs.
    """
    rooms = _load_room_specs(Path(rooms_json), room_type=room_type)
    if not rooms:
        raise ValueError(f"No rooms of type '{room_type}' were found in {rooms_json}.")

    return [RoomSpec(room_id=room.room_id, title=_normalize_room_title(room.title)) for room in rooms]


def main() -> None:
    """Signature: def main() -> None.

    Run the fetch webex archive entrypoint.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_fetch_config(Path(args.config_path))
    token = _resolve_token(config)
    policy = _parse_max_total_messages(str(config.get("max_total_messages") or ""))

    room_specs = _resolve_room_specs(args.rooms_json, args.room_type)
    logger.info("Resolved %s Webex spaces to archive.", len(room_specs))

    used_names: set[str] = set()
    for room in room_specs:
        basename = _build_output_basename(room.title, room.room_id, used_names)
        output_path = output_dir / f"{basename}.json"
        if args.skip_existing and output_path.exists():
            logger.info("Skipping existing archive for '%s': %s", room.title, output_path)
            continue

        logger.info("Fetching messages for '%s'", room.title)
        messages = _fetch_room_messages(
            room_id=room.room_id,
            token=token,
            policy=policy,
            page_size=args.page_size,
        )
        output_path.write_text(
            json.dumps(messages, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %s messages to %s", len(messages), output_path)


if __name__ == "__main__":
    main()
