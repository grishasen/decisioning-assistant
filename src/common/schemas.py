from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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
