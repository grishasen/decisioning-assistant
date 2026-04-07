from datetime import datetime, timezone

from ingestion.fetch_webex_archive import (
    ALLOWED_CONFIG_KEYS,
    FILENAME_MAX_CHARS,
    _build_output_basename,
    _load_fetch_config,
    _load_room_specs,
    _parse_max_total_messages,
)


def test_parse_max_total_messages_defaults_to_last_1000_messages() -> None:
    """Signature: def test_parse_max_total_messages_defaults_to_last_1000_messages() -> None.

    Verify that parse max total messages defaults to last 1000 messages.
    """
    policy = _parse_max_total_messages("")

    assert policy.total_limit == 1000
    assert policy.after is None
    assert policy.before is None


def test_parse_max_total_messages_days_window() -> None:
    """Signature: def test_parse_max_total_messages_days_window() -> None.

    Verify that parse max total messages days window.
    """
    reference = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)

    policy = _parse_max_total_messages("30d", now=reference)

    assert policy.total_limit is None
    assert policy.after == datetime(2026, 2, 13, 12, 0, tzinfo=timezone.utc)
    assert policy.before is None


def test_parse_max_total_messages_date_range() -> None:
    """Signature: def test_parse_max_total_messages_date_range() -> None.

    Verify that parse max total messages date range.
    """
    policy = _parse_max_total_messages("01052021-11062021")

    assert policy.total_limit is None
    assert policy.after == datetime(2021, 5, 1, 0, 0, tzinfo=timezone.utc)
    assert policy.before == datetime(2021, 6, 12, 0, 0, tzinfo=timezone.utc)


def test_build_output_basename_truncates_and_disambiguates() -> None:
    """Signature: def test_build_output_basename_truncates_and_disambiguates() -> None.

    Verify that build output basename truncates and disambiguates.
    """
    used_names: set[str] = set()
    title = "Outbound Batch Campaign -VC DF Taking long time for processing"

    first = _build_output_basename(title, "room-1", used_names)
    second = _build_output_basename(title, "room-2", used_names)

    assert len(first) <= FILENAME_MAX_CHARS
    assert len(second) <= FILENAME_MAX_CHARS
    assert first != second
    assert " " not in first


def test_load_room_specs_filters_by_room_type(tmp_path) -> None:
    """Signature: def test_load_room_specs_filters_by_room_type(tmp_path) -> None.

    Verify that load room specs filters by room type.
    """
    rooms_path = tmp_path / "rooms.json"
    rooms_path.write_text(
        """{
  "items": [
    {"id": "room-group", "title": "Group Room", "type": "group"},
    {"id": "room-direct", "title": "Direct Room", "type": "direct"}
  ]
}
""",
        encoding="utf-8",
    )

    group_rooms = _load_room_specs(rooms_path, room_type="group")
    direct_rooms = _load_room_specs(rooms_path, room_type="direct")
    all_rooms = _load_room_specs(rooms_path, room_type="all")

    assert [room.room_id for room in group_rooms] == ["room-group"]
    assert [room.room_id for room in direct_rooms] == ["room-direct"]
    assert [room.room_id for room in all_rooms] == ["room-group", "room-direct"]


def test_load_fetch_config_rejects_unknown_keys(tmp_path) -> None:
    """Signature: def test_load_fetch_config_rejects_unknown_keys(tmp_path) -> None.

    Verify that load fetch config rejects unknown keys.
    """
    config_path = tmp_path / "webex_fetch.yaml"
    config_path.write_text(
        "token: abc\nmax_total_messages: 30d\ndownload: no\n",
        encoding="utf-8",
    )

    try:
        _load_fetch_config(config_path)
    except ValueError as exc:
        assert "download" in str(exc)
    else:
        raise AssertionError("Expected unknown key validation to fail")


def test_allowed_config_keys_are_minimal() -> None:
    """Signature: def test_allowed_config_keys_are_minimal() -> None.

    Verify that allowed config keys are minimal.
    """
    assert ALLOWED_CONFIG_KEYS == {"token", "max_total_messages"}
