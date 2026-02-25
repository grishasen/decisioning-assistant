from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz
from tqdm import tqdm

from common.io_utils import write_jsonl
from common.logging_utils import get_logger
from common.schemas import DocumentRecord, build_metadata, normalize_doc_type
from common.text_utils import normalize_whitespace, stable_id

logger = get_logger(__name__)

_HEADING_NUMERIC_RE = re.compile(r"^(?:chapter\s+)?\d+(?:\.\d+)*\s+\S+", re.IGNORECASE)
_PDF_DATE_RE = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})?(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
)


@dataclass
class SectionRange:
    title: str
    level: int
    start_page: int
    end_page: int
    path: str


def _parse_pdf_datetime(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str):
        return None

    text = raw_value.strip()
    if not text:
        return None

    if text.startswith("D:"):
        text = text[2:]

    match = _PDF_DATE_RE.match(text)
    if not match:
        return None

    year = int(match.group("year"))
    month = int(match.group("month") or "1")
    day = int(match.group("day") or "1")
    hour = int(match.group("hour") or "0")
    minute = int(match.group("minute") or "0")
    second = int(match.group("second") or "0")

    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _extract_sections(pdf_doc: fitz.Document, default_title: str, use_toc: bool) -> list[SectionRange]:
    page_count = len(pdf_doc)
    if page_count == 0:
        return []

    if not use_toc:
        return [
            SectionRange(
                title=default_title,
                level=1,
                start_page=1,
                end_page=page_count,
                path=default_title,
            )
        ]

    toc = pdf_doc.get_toc() or []
    if not toc:
        return [
            SectionRange(
                title=default_title,
                level=1,
                start_page=1,
                end_page=page_count,
                path=default_title,
            )
        ]

    parsed: list[SectionRange] = []
    heading_stack: dict[int, str] = {}
    for item in toc:
        if not isinstance(item, list) or len(item) < 3:
            continue

        level = int(item[0]) if isinstance(item[0], (int, float)) else 1
        title = normalize_whitespace(str(item[1] or "Untitled Section"))
        start_page_raw = item[2]
        if not isinstance(start_page_raw, int):
            continue

        normalized_level = max(level, 1)
        start_page = max(1, min(page_count, start_page_raw))

        heading_stack[normalized_level] = title
        for stacked_level in sorted([k for k in heading_stack if k > normalized_level]):
            heading_stack.pop(stacked_level, None)

        path_parts = [
            heading_stack[lvl]
            for lvl in sorted(heading_stack)
            if lvl <= normalized_level and heading_stack.get(lvl)
        ]
        section_path = " > ".join(path_parts) if path_parts else title

        parsed.append(
            SectionRange(
                title=title or "Untitled Section",
                level=normalized_level,
                start_page=start_page,
                end_page=start_page,
                path=section_path or (title or "Untitled Section"),
            )
        )

    if not parsed:
        return [
            SectionRange(
                title=default_title,
                level=1,
                start_page=1,
                end_page=page_count,
                path=default_title,
            )
        ]

    # Prefer top-level chapter entries when available.
    top_level = [section for section in parsed if section.level == 1]
    candidates = top_level or parsed

    # Deduplicate by page start to avoid overlapping ranges from nested TOC items.
    by_start: dict[int, SectionRange] = {}
    for section in sorted(candidates, key=lambda s: (s.start_page, s.level, s.title.lower())):
        by_start.setdefault(section.start_page, section)

    starts = sorted(by_start.keys())
    sections: list[SectionRange] = []
    for idx, start in enumerate(starts):
        current = by_start[start]
        next_start = starts[idx + 1] if idx + 1 < len(starts) else page_count + 1
        end_page = max(current.start_page, min(page_count, next_start - 1))
        sections.append(
            SectionRange(
                title=current.title,
                level=current.level,
                start_page=current.start_page,
                end_page=end_page,
                path=current.path,
            )
        )

    return sections


def _paragraphs_from_text(text: str) -> list[str]:
    if not text:
        return []

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", normalized_newlines)

    paragraphs: list[str] = []
    for block in blocks:
        value = normalize_whitespace(block)
        if value:
            paragraphs.append(value)
    return paragraphs


def _looks_like_heading(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text:
        return False

    if _HEADING_NUMERIC_RE.match(text):
        return True

    words = text.split()
    if len(words) > 14:
        return False
    if len(text) > 120:
        return False

    if text.endswith("."):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return uppercase_ratio >= 0.65


def _build_section_records(
    pdf_path: Path,
    pdf_doc: fitz.Document,
    title: str,
    section: SectionRange,
    target_chars: int,
    min_chars: int,
    product: str | None,
    doc_version: str | None,
    doc_type: str | None,
    created_at: datetime | None,
    updated_at: datetime | None,
    ingested_at: datetime,
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []

    section_text_blocks: list[str] = []
    for page_no in range(section.start_page, section.end_page + 1):
        page_text_raw = pdf_doc[page_no - 1].get_text("text") or ""
        if page_text_raw.strip():
            section_text_blocks.append(page_text_raw)

    section_text = "\n\n".join(section_text_blocks).strip()
    if not section_text:
        return records

    paragraphs = _paragraphs_from_text(section_text)
    if not paragraphs:
        return records

    active_heading = section.title
    paragraph_buffer: list[str] = []
    block_index = 0

    def flush() -> None:
        nonlocal block_index
        combined = "\n\n".join(paragraph_buffer).strip()
        if not combined:
            return
        if len(combined) < min_chars and records:
            # Merge short tail into previous block in same section.
            previous = records[-1]
            merged_text = f"{previous.text}\n\n{combined}".strip()
            previous.text = merged_text
            previous.markdown = f"{previous.markdown or ''}\n\n{combined}".strip()
            previous.metadata["merged_short_tail"] = True
            return

        source_ref = (
            f"pdf::{pdf_path}#section={section.start_page}-{section.end_page}:block={block_index}"
        )
        doc_id = stable_id(
            "pdf-section",
            str(pdf_path.resolve()),
            str(section.start_page),
            str(section.end_page),
            str(block_index),
            active_heading,
        )

        markdown = (
            f"## {section.title}\n"
            f"### {active_heading}\n\n"
            f"{combined}\n"
        )

        section_path = section.path
        if active_heading and active_heading != section.title:
            section_path = f"{section.path} > {active_heading}"

        metadata = build_metadata(
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            section_path=section_path,
            page_start=section.start_page,
            page_end=section.end_page,
            created_at=created_at,
            updated_at=updated_at,
            ingested_at=ingested_at,
        )
        metadata.update(
            {
                "file_name": pdf_path.name,
                "pdf_title": title,
                "section_title": section.title,
                "section_level": section.level,
                "active_heading": active_heading,
                "split_mode": "chapter_paragraph",
                "block_index": block_index,
                "paragraph_count": len(paragraph_buffer),
                "page_count": len(pdf_doc),
            }
        )

        records.append(
            DocumentRecord(
                doc_id=doc_id,
                source_type="pdf",
                source_ref=source_ref,
                source_path=str(pdf_path),
                title=title,
                page=section.start_page,
                text=combined,
                markdown=markdown,
                created_at=created_at,
                metadata=metadata,
            )
        )
        block_index += 1

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            if paragraph_buffer:
                flush()
                paragraph_buffer = []
            active_heading = paragraph
            continue

        paragraph_buffer.append(paragraph)

        if target_chars > 0:
            current_length = len("\n\n".join(paragraph_buffer))
            if current_length >= target_chars:
                flush()
                paragraph_buffer = []

    if paragraph_buffer:
        flush()

    return records


def extract_pdf_records(
    pdf_path: Path,
    target_chars: int = 900,
    min_chars: int = 220,
    use_toc: bool = True,
    product: str | None = None,
    doc_version: str | None = None,
    doc_type: str | None = None,
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    ingested_at = datetime.now(timezone.utc)

    with fitz.open(pdf_path) as pdf_doc:
        metadata = pdf_doc.metadata or {}
        title = metadata.get("title") or pdf_path.stem

        created_at = _parse_pdf_datetime(metadata.get("creationDate"))
        updated_at = _parse_pdf_datetime(metadata.get("modDate")) or created_at

        sections = _extract_sections(pdf_doc, title, use_toc=use_toc)
        for section in sections:
            records.extend(
                _build_section_records(
                    pdf_path=pdf_path,
                    pdf_doc=pdf_doc,
                    title=title,
                    section=section,
                    target_chars=target_chars,
                    min_chars=min_chars,
                    product=product,
                    doc_version=doc_version,
                    doc_type=doc_type,
                    created_at=created_at,
                    updated_at=updated_at,
                    ingested_at=ingested_at,
                )
            )

        if records:
            for rec in records:
                rec.metadata["pdf_metadata"] = metadata
            return records

        # Fallback: keep page-level extraction for edge PDFs with no clean text.
        for page_index, page in enumerate(pdf_doc, start=1):
            page_text_raw = page.get_text("text") or ""
            page_text = normalize_whitespace(page_text_raw)
            if not page_text:
                continue

            source_ref = f"pdf::{pdf_path}#page={page_index}"
            doc_id = stable_id("pdf-page", str(pdf_path.resolve()), str(page_index))

            record_metadata = build_metadata(
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
                section_path=f"Page {page_index}",
                page_start=page_index,
                page_end=page_index,
                created_at=created_at,
                updated_at=updated_at,
                ingested_at=ingested_at,
            )
            record_metadata.update(
                {
                    "file_name": pdf_path.name,
                    "page_count": len(pdf_doc),
                    "pdf_metadata": metadata,
                    "split_mode": "page_fallback",
                }
            )

            records.append(
                DocumentRecord(
                    doc_id=doc_id,
                    source_type="pdf",
                    source_ref=source_ref,
                    source_path=str(pdf_path),
                    title=title,
                    page=page_index,
                    text=page_text,
                    markdown=f"## Page {page_index}\n\n{page_text}\n",
                    created_at=created_at,
                    metadata=record_metadata,
                )
            )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PDFs into structure-aware records (chapters/sections/paragraph blocks)."
    )
    parser.add_argument("--input-dir", default="data/raw/pdf")
    parser.add_argument("--output", default="data/staging/documents/pdf_documents.jsonl")
    parser.add_argument(
        "--target-chars",
        type=int,
        default=900,
        help="Target text size for each paragraph block record.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=220,
        help="Minimum text size for each paragraph block record.",
    )
    parser.add_argument(
        "--use-toc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PDF table-of-contents as chapter/section boundaries (default: true).",
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

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", input_dir)
        write_jsonl(args.output, [])
        return

    all_rows: list[dict] = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting PDFs"):
        try:
            records = extract_pdf_records(
                pdf_path=pdf_path,
                target_chars=args.target_chars,
                min_chars=args.min_chars,
                use_toc=args.use_toc,
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
            )
            all_rows.extend(record.model_dump(mode="json") for record in records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse PDF %s: %s", pdf_path, exc)

    count = write_jsonl(args.output, all_rows)
    logger.info("Wrote %s PDF structured documents to %s", count, args.output)


if __name__ == "__main__":
    main()
