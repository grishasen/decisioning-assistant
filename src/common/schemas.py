from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_DOC_TYPE_ALIASES = {
    "guide": "guide",
    "guides": "guide",
    "api": "api",
    "reference": "api",
    "release-note": "release-note",
    "release-notes": "release-note",
    "releasenote": "release-note",
}


class CanonicalMetadata(BaseModel):
    product: str | None = None
    doc_version: str | None = None
    doc_type: Literal["guide", "api", "release-note"] | None = None
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    room_id: str | None = None
    room_title: str | None = None
    thread_id: str | None = None
    message_count: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    ingested_at: datetime | None = None

    model_config = ConfigDict(extra="allow")


def normalize_doc_type(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().lower()
    if not cleaned:
        return None

    normalized = _DOC_TYPE_ALIASES.get(cleaned)
    if normalized:
        return normalized

    raise ValueError(
        "doc_type must be one of: guide, api, release-note"
    )


def build_metadata(**kwargs: Any) -> dict[str, Any]:
    return CanonicalMetadata(**kwargs).model_dump(mode="json", exclude_none=True)


class DocumentRecord(BaseModel):
    doc_id: str
    source_type: str
    source_ref: str
    source_path: str | None = None
    title: str | None = None
    page: int | None = None
    text: str
    markdown: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    source_ref: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QARecord(BaseModel):
    qa_id: str
    question: str
    answer: str
    chunk_id: str
    doc_id: str
    source_ref: str
    source_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
