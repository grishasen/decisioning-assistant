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
from common.text_utils import normalize_whitespace, split_paragraphs, split_sentences, stable_id

logger = get_logger(__name__)

_HEADING_NUMERIC_RE = re.compile(r"^(?:chapter\s+)?\d+(?:\.\d+)*\s+\S+", re.IGNORECASE)
_PDF_DATE_RE = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})?(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
)


@dataclass(frozen=True)
class SectionRange:
    """Represent a section boundary detected during PDF ingestion."""
    title: str
    level: int
    start_page: int
    end_page: int
    path: str


@dataclass(frozen=True)
class ParagraphUnit:
    """Represent a paragraph-level PDF unit with page and heading metadata."""
    text: str
    page_start: int
    page_end: int
    sentence_split: bool = False


@dataclass
class ParagraphChunkPlan:
    """Describe how paragraph units are grouped into one PDF chunk."""
    active_heading: str
    units: list[ParagraphUnit]
    sentence_split_count: int = 0


def _parse_pdf_datetime(raw_value: object) -> datetime | None:
    """Signature: def _parse_pdf_datetime(raw_value: object) -> datetime | None.

    Parse pdf datetime.
    """
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
    """Signature: def _extract_sections(pdf_doc: fitz.Document, default_title: str, use_toc: bool) -> list[SectionRange].

    Extract sections.
    """
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

    top_level = [section for section in parsed if section.level == 1]
    candidates = top_level or parsed

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


def _looks_like_heading(paragraph: str) -> bool:
    """Signature: def _looks_like_heading(paragraph: str) -> bool.

    Looks like heading.
    """
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


def _extract_page_text_blocks(page: fitz.Page) -> list[str]:
    """Signature: def _extract_page_text_blocks(page: fitz.Page) -> list[str].

    Extract page text blocks.
    """
    try:
        raw_blocks = page.get_text("blocks", sort=True) or []
    except TypeError:
        raw_blocks = page.get_text("blocks") or []

    blocks: list[str] = []
    for block in raw_blocks:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue

        block_type = block[6] if len(block) > 6 else 0
        if isinstance(block_type, (int, float)) and int(block_type) != 0:
            continue

        block_text = block[4] if isinstance(block[4], str) else ""
        if block_text.strip():
            blocks.append(block_text)

    return blocks


def _paragraph_units_from_page(page: fitz.Page, page_no: int) -> list[ParagraphUnit]:
    """Signature: def _paragraph_units_from_page(page: fitz.Page, page_no: int) -> list[ParagraphUnit].

    Paragraph units from page.
    """
    units: list[ParagraphUnit] = []
    for block_text in _extract_page_text_blocks(page):
        for paragraph in split_paragraphs(block_text):
            units.append(ParagraphUnit(text=paragraph, page_start=page_no, page_end=page_no))

    if units:
        return units

    page_text_raw = page.get_text("text") or ""
    for paragraph in split_paragraphs(page_text_raw):
        units.append(ParagraphUnit(text=paragraph, page_start=page_no, page_end=page_no))
    return units


def _extract_section_paragraph_units(
    pdf_doc: fitz.Document,
    section: SectionRange,
) -> list[ParagraphUnit]:
    """Signature: def _extract_section_paragraph_units(pdf_doc: fitz.Document, section: SectionRange) -> list[ParagraphUnit].

    Extract section paragraph units.
    """
    units: list[ParagraphUnit] = []
    for page_no in range(section.start_page, section.end_page + 1):
        units.extend(_paragraph_units_from_page(pdf_doc[page_no - 1], page_no))
    return units


def _expand_oversized_unit(unit: ParagraphUnit, target_chars: int) -> list[ParagraphUnit]:
    """Signature: def _expand_oversized_unit(unit: ParagraphUnit, target_chars: int) -> list[ParagraphUnit].

    Expand oversized unit.
    """
    if target_chars <= 0 or len(unit.text) <= target_chars:
        return [unit]

    sentences = split_sentences(unit.text)
    if len(sentences) <= 1:
        return [unit]

    expanded: list[ParagraphUnit] = []
    current: list[str] = []
    for sentence in sentences:
        if not current:
            current = [sentence]
            continue

        candidate = " ".join([*current, sentence]).strip()
        if len(candidate) <= target_chars:
            current.append(sentence)
            continue

        expanded.append(
            ParagraphUnit(
                text=" ".join(current).strip(),
                page_start=unit.page_start,
                page_end=unit.page_end,
                sentence_split=True,
            )
        )
        current = [sentence]

    if current:
        expanded.append(
            ParagraphUnit(
                text=" ".join(current).strip(),
                page_start=unit.page_start,
                page_end=unit.page_end,
                sentence_split=True,
            )
        )

    return expanded or [unit]


def _join_units(units: list[ParagraphUnit]) -> str:
    """Signature: def _join_units(units: list[ParagraphUnit]) -> str.

    Join units.
    """
    return "\n\n".join(unit.text for unit in units).strip()


def _build_chunk_plans(
    section: SectionRange,
    paragraph_units: list[ParagraphUnit],
    target_chars: int,
    min_chars: int,
) -> list[ParagraphChunkPlan]:
    """Signature: def _build_chunk_plans(section: SectionRange, paragraph_units: list[ParagraphUnit], target_chars: int, min_chars: int) -> list[ParagraphChunkPlan].

    Build chunk plans.
    """
    plans: list[ParagraphChunkPlan] = []
    current_units: list[ParagraphUnit] = []
    current_heading = section.title
    current_sentence_split_count = 0

    def flush() -> None:
        """Signature: def flush() -> None.

        Flush the current paragraph units into a chunk plan.
        """
        nonlocal current_units, current_sentence_split_count
        combined = _join_units(current_units)
        if not combined:
            current_units = []
            current_sentence_split_count = 0
            return

        plan = ParagraphChunkPlan(
            active_heading=current_heading,
            units=list(current_units),
            sentence_split_count=current_sentence_split_count,
        )
        if len(combined) < min_chars and plans and plans[-1].active_heading == current_heading:
            plans[-1].units.extend(plan.units)
            plans[-1].sentence_split_count += plan.sentence_split_count
        else:
            plans.append(plan)

        current_units = []
        current_sentence_split_count = 0

    for unit in paragraph_units:
        if _looks_like_heading(unit.text):
            flush()
            current_heading = unit.text
            continue

        for expanded in _expand_oversized_unit(unit, target_chars):
            if not current_units:
                current_units = [expanded]
                current_sentence_split_count += int(expanded.sentence_split)
                continue

            candidate = _join_units([*current_units, expanded])
            if target_chars > 0 and len(candidate) > target_chars:
                flush()

            current_units.append(expanded)
            current_sentence_split_count += int(expanded.sentence_split)

            if target_chars > 0 and len(_join_units(current_units)) >= target_chars:
                flush()

    flush()
    return plans


def _build_records_from_units(
    pdf_path: Path,
    pdf_doc: fitz.Document,
    title: str,
    section: SectionRange,
    paragraph_units: list[ParagraphUnit],
    target_chars: int,
    min_chars: int,
    split_mode: str,
    product: str | None,
    doc_version: str | None,
    doc_type: str | None,
    created_at: datetime | None,
    updated_at: datetime | None,
    ingested_at: datetime,
) -> list[DocumentRecord]:
    """Signature: def _build_records_from_units(pdf_path: Path, pdf_doc: fitz.Document, title: str, section: SectionRange, paragraph_units: list[ParagraphUnit], target_chars: int, min_chars: int, split_mode: str, product: str | None, doc_version: str | None, doc_type: str | None, created_at: datetime | None, updated_at: datetime | None, ingested_at: datetime) -> list[DocumentRecord].

    Build records from units.
    """
    plans = _build_chunk_plans(section, paragraph_units, target_chars, min_chars)
    records: list[DocumentRecord] = []

    for block_index, plan in enumerate(plans):
        combined = _join_units(plan.units)
        if not combined:
            continue

        page_numbers = sorted(
            {
                page_no
                for unit in plan.units
                for page_no in range(unit.page_start, unit.page_end + 1)
            }
        )
        if not page_numbers:
            continue

        chunk_page_start = page_numbers[0]
        chunk_page_end = page_numbers[-1]
        section_path = section.path
        if plan.active_heading and plan.active_heading != section.title:
            section_path = f"{section.path} > {plan.active_heading}"

        source_ref = (
            f"pdf::{pdf_path}#section={section.start_page}-{section.end_page}:"
            f"pages={chunk_page_start}-{chunk_page_end}:block={block_index}"
        )
        doc_id = stable_id(
            "pdf-section",
            str(pdf_path.resolve()),
            str(section.start_page),
            str(section.end_page),
            str(chunk_page_start),
            str(chunk_page_end),
            str(block_index),
            plan.active_heading,
        )

        markdown_lines = [f"## {section.title}"]
        if plan.active_heading and plan.active_heading != section.title:
            markdown_lines.append(f"### {plan.active_heading}")
        markdown_lines.extend(["", combined])
        markdown = "\n".join(markdown_lines).strip() + "\n"

        metadata = build_metadata(
            product=product,
            doc_version=doc_version,
            doc_type=doc_type,
            section_path=section_path,
            page_start=chunk_page_start,
            page_end=chunk_page_end,
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
                "active_heading": plan.active_heading,
                "split_mode": split_mode,
                "chunk_strategy": "paragraph_bound",
                "block_index": block_index,
                "paragraph_count": len(plan.units),
                "sentence_split_count": plan.sentence_split_count,
                "page_numbers": page_numbers,
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
                page=chunk_page_start,
                text=combined,
                markdown=markdown,
                created_at=created_at,
                metadata=metadata,
            )
        )

    return records


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
    """Signature: def _build_section_records(pdf_path: Path, pdf_doc: fitz.Document, title: str, section: SectionRange, target_chars: int, min_chars: int, product: str | None, doc_version: str | None, doc_type: str | None, created_at: datetime | None, updated_at: datetime | None, ingested_at: datetime) -> list[DocumentRecord].

    Build section records.
    """
    paragraph_units = _extract_section_paragraph_units(pdf_doc, section)
    if not paragraph_units:
        return []

    return _build_records_from_units(
        pdf_path=pdf_path,
        pdf_doc=pdf_doc,
        title=title,
        section=section,
        paragraph_units=paragraph_units,
        target_chars=target_chars,
        min_chars=min_chars,
        split_mode="chapter_paragraph",
        product=product,
        doc_version=doc_version,
        doc_type=doc_type,
        created_at=created_at,
        updated_at=updated_at,
        ingested_at=ingested_at,
    )


def extract_pdf_records(
    pdf_path: Path,
    target_chars: int = 900,
    min_chars: int = 220,
    use_toc: bool = True,
    product: str | None = None,
    doc_version: str | None = None,
    doc_type: str | None = None,
) -> list[DocumentRecord]:
    """Signature: def extract_pdf_records(pdf_path: Path, target_chars: int = 900, min_chars: int = 220, use_toc: bool = True, product: str | None = None, doc_version: str | None = None, doc_type: str | None = None) -> list[DocumentRecord].

    Extract pdf records.
    """
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

        for page_index, page in enumerate(pdf_doc, start=1):
            paragraph_units = _paragraph_units_from_page(page, page_index)
            if not paragraph_units:
                page_text = normalize_whitespace(page.get_text("text") or "")
                if not page_text:
                    continue
                paragraph_units = [
                    ParagraphUnit(text=page_text, page_start=page_index, page_end=page_index)
                ]

            page_section = SectionRange(
                title=f"Page {page_index}",
                level=1,
                start_page=page_index,
                end_page=page_index,
                path=f"Page {page_index}",
            )
            page_records = _build_records_from_units(
                pdf_path=pdf_path,
                pdf_doc=pdf_doc,
                title=title,
                section=page_section,
                paragraph_units=paragraph_units,
                target_chars=target_chars,
                min_chars=min_chars,
                split_mode="page_paragraph",
                product=product,
                doc_version=doc_version,
                doc_type=doc_type,
                created_at=created_at,
                updated_at=updated_at,
                ingested_at=ingested_at,
            )
            records.extend(page_records)

        for rec in records:
            rec.metadata["pdf_metadata"] = metadata

    return records


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for ingest pdfs.
    """
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
    """Signature: def main() -> None.

    Run the ingest pdfs entrypoint.
    """
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
