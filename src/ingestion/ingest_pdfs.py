from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from tqdm import tqdm

from common.io_utils import write_jsonl
from common.logging_utils import get_logger
from common.schemas import DocumentRecord
from common.text_utils import normalize_whitespace, stable_id

logger = get_logger(__name__)


def _markdown_from_page_text(page_text: str, page_number: int) -> str:
    cleaned = page_text.strip()
    if not cleaned:
        return ""
    return f"## Page {page_number}\n\n{cleaned}\n"


def extract_pdf_records(pdf_path: Path) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    with fitz.open(pdf_path) as pdf_doc:
        metadata = pdf_doc.metadata or {}
        title = metadata.get("title") or pdf_path.stem

        for page_index, page in enumerate(pdf_doc, start=1):
            page_text_raw = page.get_text("text") or ""
            page_text = normalize_whitespace(page_text_raw)
            if not page_text:
                continue

            markdown = _markdown_from_page_text(page_text_raw, page_index)
            source_ref = f"pdf::{pdf_path}#page={page_index}"
            doc_id = stable_id("pdf", str(pdf_path.resolve()), str(page_index))

            record = DocumentRecord(
                doc_id=doc_id,
                source_type="pdf",
                source_ref=source_ref,
                source_path=str(pdf_path),
                title=title,
                page=page_index,
                text=page_text,
                markdown=markdown,
                metadata={
                    "file_name": pdf_path.name,
                    "page_count": len(pdf_doc),
                    "pdf_metadata": metadata,
                },
            )
            records.append(record)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF pages into normalized document JSONL.")
    parser.add_argument("--input-dir", default="data/raw/pdf")
    parser.add_argument("--output", default="data/staging/documents/pdf_documents.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", input_dir)
        write_jsonl(args.output, [])
        return

    all_rows: list[dict] = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting PDFs"):
        try:
            records = extract_pdf_records(pdf_path)
            all_rows.extend(record.model_dump(mode="json") for record in records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse PDF %s: %s", pdf_path, exc)

    count = write_jsonl(args.output, all_rows)
    logger.info("Wrote %s PDF page documents to %s", count, args.output)


if __name__ == "__main__":
    main()
