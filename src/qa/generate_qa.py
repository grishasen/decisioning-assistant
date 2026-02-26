from __future__ import annotations

import argparse
from typing import Any

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml, write_jsonl
from common.logging_utils import get_logger
from common.mlx_utils import MLXLoadedGenerator, extract_first_json_object
from common.prompts import qa_generation_prompt
from common.schemas import ChunkRecord, QARecord
from common.text_utils import normalize_whitespace, stable_id

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA from chunked documents."
    )
    parser.add_argument("--qa-config", default="configs/qa_generation.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    return parser.parse_args()


def _load_chunks(path: str, limit: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for row in iter_jsonl(path):
        chunks.append(ChunkRecord.model_validate(row))
        if limit > 0 and len(chunks) >= limit:
            break
    return chunks


def _extract_qa_pairs(model_output: str) -> list[dict[str, str]]:
    payload = extract_first_json_object(model_output)
    if not payload:
        return []
    pairs = payload.get("qa_pairs")
    if not isinstance(pairs, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        if isinstance(q, str) and isinstance(a, str):
            q = normalize_whitespace(q)
            a = normalize_whitespace(a)
            if q and a:
                cleaned.append({"question": q, "answer": a})
    return cleaned


def _is_webex_chunk(chunk: ChunkRecord) -> bool:
    if chunk.source_type.strip().lower() == "webex":
        return True
    return chunk.source_ref.strip().lower().startswith("webex::")


def main() -> None:
    args = parse_args()
    qa_cfg = read_yaml(args.qa_config)
    model_cfg = read_yaml(args.models_config)

    input_chunks = str(qa_cfg.get("input_chunks", "data/staging/chunks/chunks.jsonl"))
    output_raw_qa = str(qa_cfg.get("output_raw_qa", "data/qa/qa_raw.jsonl"))
    qa_per_chunk = int(qa_cfg.get("qa_per_chunk", 2))
    max_chunks = int(qa_cfg.get("max_chunks", 0))

    min_question_chars = int(qa_cfg.get("min_question_chars", 12))
    min_answer_chars = int(qa_cfg.get("min_answer_chars", 24))
    max_answer_chars = int(qa_cfg.get("max_answer_chars", 1800))
    min_webex_chunk_chars = int(qa_cfg.get("min_webex_chunk_chars", 200))
    if min_webex_chunk_chars < 0:
        raise ValueError("min_webex_chunk_chars must be >= 0")

    qa_model_cfg: dict[str, Any] = model_cfg.get("qa_generator", {})
    model_name = str(qa_model_cfg.get("model", "mlx-community/gemma-2-2b-it-4bit"))
    adapter_path = qa_model_cfg.get("adapter_path")
    trust_remote_code = bool(qa_model_cfg.get("trust_remote_code", True))
    max_tokens = int(qa_model_cfg.get("max_tokens", 320))
    temperature = float(qa_model_cfg.get("temperature", 0.2))

    chunks = _load_chunks(input_chunks, max_chunks)
    if not chunks:
        logger.warning("No chunks available at %s", input_chunks)
        write_jsonl(output_raw_qa, [])
        return

    logger.info("Loading QA model once for this process: %s", model_name)
    generator = MLXLoadedGenerator(
        model=model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
        trust_remote_code=trust_remote_code,
    )

    rows: list[dict[str, Any]] = []
    skipped_short_webex_chunks = 0
    for chunk in tqdm(chunks, desc="Generating QA"):
        chunk_text = chunk.text.strip()
        if _is_webex_chunk(chunk) and len(chunk_text) < min_webex_chunk_chars:
            skipped_short_webex_chunks += 1
            logger.debug(
                "Skipping webex chunk %s because %s chars < min_webex_chunk_chars=%s",
                chunk.chunk_id,
                len(chunk_text),
                min_webex_chunk_chars,
            )
            continue

        prompt = qa_generation_prompt(chunk_text, qa_per_chunk)
        try:
            model_output = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed for chunk %s: %s", chunk.chunk_id, exc)
            continue

        qa_pairs = _extract_qa_pairs(model_output)
        if not qa_pairs:
            logger.debug("No valid QA parsed for chunk %s", chunk.chunk_id)
            continue

        for idx, pair in enumerate(qa_pairs):
            question = pair["question"]
            answer = pair["answer"]
            if len(question) < min_question_chars:
                continue
            if len(answer) < min_answer_chars or len(answer) > max_answer_chars:
                continue

            qa_id = stable_id(chunk.chunk_id, str(idx), question)
            row = QARecord(
                qa_id=qa_id,
                question=question,
                answer=answer,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_ref=chunk.source_ref,
                source_type=chunk.source_type,
                metadata={
                    "chunk_text": chunk_text,
                    "chunk_metadata": chunk.metadata,
                },
            ).model_dump(mode="json")
            rows.append(row)

    if skipped_short_webex_chunks > 0:
        logger.info(
            "Skipped %s webex chunks shorter than %s characters.",
            skipped_short_webex_chunks,
            min_webex_chunk_chars,
        )

    count = write_jsonl(output_raw_qa, rows)
    logger.info("Wrote %s raw QA rows to %s", count, output_raw_qa)


if __name__ == "__main__":
    main()
