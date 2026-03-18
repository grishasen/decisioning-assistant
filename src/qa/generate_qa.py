from __future__ import annotations

import argparse
from typing import Any

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml, write_jsonl
from common.logging_utils import get_logger
from common.mlx_utils import MLXLoadedGenerator, extract_first_json_object
from common.prompts import qa_generation_prompt, webex_thread_question_prompt
from common.schemas import ChunkRecord, QARecord
from common.text_utils import normalize_whitespace, stable_id
from common.webex_utils import WebexThreadLine, parse_webex_thread_lines, parse_webex_thread_message_line

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


def _extract_question(model_output: str) -> str:
    payload = extract_first_json_object(model_output)
    if isinstance(payload, dict):
        question = payload.get("question")
        if isinstance(question, str):
            cleaned = normalize_whitespace(question)
            if cleaned:
                return cleaned

    for line in model_output.splitlines():
        cleaned = normalize_whitespace(line)
        if cleaned:
            return cleaned.strip('"')
    return ""


def _is_webex_chunk(chunk: ChunkRecord) -> bool:
    if chunk.source_type.strip().lower() == "webex":
        return True
    return chunk.source_ref.strip().lower().startswith("webex::")


def _is_webex_thread_chunk(chunk: ChunkRecord) -> bool:
    if not _is_webex_chunk(chunk):
        return False
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    if bool(metadata.get("is_thread_document")):
        return True
    return str(metadata.get("webex_grouping", "")).strip().lower() == "thread"


def _normalize_match_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_whitespace(value).lower()


def _webex_thread_messages(chunk: ChunkRecord) -> list[WebexThreadLine]:
    return parse_webex_thread_lines(chunk.text)


def _webex_thread_replies(chunk: ChunkRecord) -> list[WebexThreadLine]:
    lines = _webex_thread_messages(chunk)
    if len(lines) <= 1:
        return []
    return lines[1:]


def _webex_chunk_matches_user(chunk: ChunkRecord, normalized_user_name: str) -> bool:
    if not normalized_user_name:
        return True

    if _is_webex_thread_chunk(chunk):
        for item in _webex_thread_replies(chunk):
            if item.author and normalized_user_name in _normalize_match_value(item.author):
                return True
        return False

    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    candidates: list[str] = []
    for key in ("person_name", "person_email", "author"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)

    for candidate in candidates:
        if normalized_user_name in _normalize_match_value(candidate):
            return True

    for line in chunk.text.splitlines():
        parsed_line = parse_webex_thread_message_line(line.strip())
        if parsed_line and normalized_user_name in _normalize_match_value(parsed_line.author):
            return True

    return False


def _build_webex_thread_answer(
    chunk: ChunkRecord,
    normalized_user_name: str = "",
) -> tuple[str, str, int] | None:
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    thread_start_text = normalize_whitespace(str(metadata.get("thread_start_text") or ""))

    thread_lines = _webex_thread_messages(chunk)
    if not thread_lines:
        return None

    reply_lines = _webex_thread_replies(chunk)
    if not thread_start_text:
        thread_start_text = thread_lines[0].message_text
    if not thread_start_text or not reply_lines:
        return None

    selected_reply_messages: list[str] = []
    for line in reply_lines:
        if normalized_user_name and normalized_user_name not in _normalize_match_value(line.author):
            continue

        if line.message_text:
            selected_reply_messages.append(line.message_text)

    answer = "\n\n".join(selected_reply_messages).strip()
    if not answer:
        return None

    return thread_start_text, answer, len(selected_reply_messages)


def _generate_webex_thread_question(
    generator: MLXLoadedGenerator,
    chunk: ChunkRecord,
    max_tokens: int,
    temperature: float,
    normalized_user_name: str = "",
) -> dict[str, str] | None:
    thread_parts = _build_webex_thread_answer(
        chunk,
        normalized_user_name=normalized_user_name,
    )
    if thread_parts is None:
        return None

    thread_start_text, answer, reply_count = thread_parts
    prompt = webex_thread_question_prompt(thread_start=thread_start_text, replies=answer)
    model_output = generator.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    question = _extract_question(model_output)
    if not question:
        return None

    return {
        "question": question,
        "answer": answer,
        "reply_count": str(reply_count),
    }


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
    max_webex_thread_answer_chars = int(
        qa_cfg.get("max_webex_thread_answer_chars", max_answer_chars)
    )
    min_webex_chunk_chars = int(qa_cfg.get("min_webex_chunk_chars", 200))
    if min_webex_chunk_chars < 0:
        raise ValueError("min_webex_chunk_chars must be >= 0")
    if max_webex_thread_answer_chars < 0:
        raise ValueError("max_webex_thread_answer_chars must be >= 0")

    webex_user_name = normalize_whitespace(str(qa_cfg.get("webex_user_name", ""))).strip()
    normalized_webex_user_name = _normalize_match_value(webex_user_name)

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

    if normalized_webex_user_name:
        logger.info(
            "Webex QA user filter enabled. Only threads with child messages from user '%s' will be processed, and only those child messages will be kept in answers.",
            webex_user_name,
        )

    logger.info("Loading QA model once for this process: %s", model_name)
    generator = MLXLoadedGenerator(
        model=model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
        trust_remote_code=trust_remote_code,
    )

    rows: list[dict[str, Any]] = []
    skipped_short_webex_chunks = 0
    skipped_webex_user_filter_chunks = 0
    skipped_webex_thread_without_answer = 0
    generated_webex_thread_qas = 0
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

        if (
            _is_webex_chunk(chunk)
            and normalized_webex_user_name
            and not _webex_chunk_matches_user(chunk, normalized_webex_user_name)
        ):
            skipped_webex_user_filter_chunks += 1
            logger.debug(
                "Skipping webex chunk %s because user '%s' was not found in child messages.",
                chunk.chunk_id,
                webex_user_name,
            )
            continue

        try:
            if _is_webex_thread_chunk(chunk):
                webex_pair = _generate_webex_thread_question(
                    generator=generator,
                    chunk=chunk,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    normalized_user_name=normalized_webex_user_name,
                )
                if not webex_pair:
                    skipped_webex_thread_without_answer += 1
                    logger.debug(
                        "Skipping webex thread chunk %s because no eligible child-message answer could be built.",
                        chunk.chunk_id,
                    )
                    continue
                qa_pairs = [webex_pair]
                generated_webex_thread_qas += 1
            else:
                prompt = qa_generation_prompt(chunk_text, qa_per_chunk)
                model_output = generator.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                qa_pairs = _extract_qa_pairs(model_output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed for chunk %s: %s", chunk.chunk_id, exc)
            continue

        if not qa_pairs:
            logger.debug("No valid QA parsed for chunk %s", chunk.chunk_id)
            continue

        for idx, pair in enumerate(qa_pairs):
            question = pair["question"]
            answer = pair["answer"]
            if len(question) < min_question_chars:
                continue

            allowed_max_answer_chars = max_answer_chars
            if _is_webex_thread_chunk(chunk):
                allowed_max_answer_chars = max_webex_thread_answer_chars

            if len(answer) < min_answer_chars or (
                allowed_max_answer_chars > 0 and len(answer) > allowed_max_answer_chars
            ):
                continue

            qa_id = stable_id(chunk.chunk_id, str(idx), question)
            row_metadata = {
                "chunk_text": chunk_text,
                "chunk_metadata": chunk.metadata,
            }
            if _is_webex_thread_chunk(chunk):
                row_metadata.update(
                    {
                        "qa_generation_mode": "webex_thread_question_child_messages_answer",
                        "thread_start_text": str(
                            chunk.metadata.get("thread_start_text") or ""
                        ).strip(),
                        "thread_reply_count": int(pair.get("reply_count") or 0),
                        "thread_reply_user_filter": webex_user_name,
                    }
                )

            row = QARecord(
                qa_id=qa_id,
                question=question,
                answer=answer,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_ref=chunk.source_ref,
                source_type=chunk.source_type,
                metadata=row_metadata,
            ).model_dump(mode="json")
            rows.append(row)

    if skipped_short_webex_chunks > 0:
        logger.info(
            "Skipped %s webex chunks shorter than %s characters.",
            skipped_short_webex_chunks,
            min_webex_chunk_chars,
        )

    if skipped_webex_user_filter_chunks > 0:
        logger.info(
            "Skipped %s webex chunks because user '%s' was not found in child messages.",
            skipped_webex_user_filter_chunks,
            webex_user_name,
        )

    if skipped_webex_thread_without_answer > 0:
        logger.info(
            "Skipped %s webex thread chunks because no eligible child-message answer could be built.",
            skipped_webex_thread_without_answer,
        )

    if generated_webex_thread_qas > 0:
        logger.info(
            "Generated %s webex thread QA rows using generated-question + child-messages-answer mode.",
            generated_webex_thread_qas,
        )

    count = write_jsonl(output_raw_qa, rows)
    logger.info("Wrote %s raw QA rows to %s", count, output_raw_qa)


if __name__ == "__main__":
    main()
