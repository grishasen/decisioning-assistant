from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from common.io_utils import append_jsonl, iter_jsonl, read_yaml, repair_jsonl_tail, write_jsonl, count_iter_jsonl
from common.logging_utils import get_logger
from common.mlx_utils import MLXLoadedGenerator, extract_first_json_object
from common.prompts import qa_generation_prompt, webex_thread_question_prompt
from common.schemas import ChunkRecord, QARecord
from common.text_utils import normalize_whitespace, stable_id
from common.webex_utils import (
    WebexThreadLine,
    parse_webex_thread_lines,
    parse_webex_thread_message_line,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Signature: def parse_args() -> argparse.Namespace.

    Parse CLI arguments for generate qa.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA from chunked documents."
    )
    parser.add_argument("--qa-config", default="configs/qa_generation.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    return parser.parse_args()


class _ChunkIterator:
    """Iterate over validated chunk records with a stable reported length."""
    def __init__(self, path: str, limit: int):
        """Signature: def __init__(self, path: str, limit: int).

        Initialize the instance state.
        """
        self.path = path
        self.limit = limit
        self._length: int | None = None

    def __iter__(self) -> Iterator[ChunkRecord]:
        """Signature: def __iter__(self) -> Iterator[ChunkRecord].

        Iterate over the items exposed by this object.
        """
        count = 0
        for row in iter_jsonl(self.path):
            yield ChunkRecord.model_validate(row)
            count += 1
            if self.limit > 0 and count >= self.limit:
                break

    def __len__(self) -> int:
        """Signature: def __len__(self) -> int.

        Return the number of items exposed by this object.
        """
        if self._length is None:
            total = count_iter_jsonl(self.path)
            if self.limit > 0:
                total = min(total, self.limit)
            self._length = total
        return self._length


def _iter_chunks(path: str, limit: int) -> _ChunkIterator:
    """Signature: def _iter_chunks(path: str, limit: int) -> _ChunkIterator.

    Iterate over chunks.
    """
    return _ChunkIterator(path, limit)


def _iter_jsonl_rows_safe(path: str):
    """Signature: def _iter_jsonl_rows_safe(path: str).

    Iterate over jsonl rows safe.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSONL line %s in %s while loading resume state.",
                    line_number,
                    file_path,
                )


def _load_processed_chunk_ids(output_raw_qa: str, qa_progress_path: str) -> set[str]:
    """Signature: def _load_processed_chunk_ids(output_raw_qa: str, qa_progress_path: str) -> set[str].

    Load processed chunk IDs from raw QA output and progress state files.
    """
    processed_from_output: set[str] = set()
    processed_from_progress: set[str] = set()

    for row in _iter_jsonl_rows_safe(output_raw_qa) or []:
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id.strip():
            processed_from_output.add(chunk_id.strip())

    for row in _iter_jsonl_rows_safe(qa_progress_path) or []:
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id.strip():
            processed_from_progress.add(chunk_id.strip())

    if processed_from_output:
        logger.info(
            "Recovered %s processed chunks from existing raw QA output.",
            len(processed_from_output),
        )
    if processed_from_progress:
        logger.info(
            "Recovered %s processed chunks from QA progress state.",
            len(processed_from_progress),
        )

    return processed_from_output | processed_from_progress


def _reset_resume_artifacts(output_raw_qa: str, qa_progress_path: str) -> None:
    """Signature: def _reset_resume_artifacts(output_raw_qa: str, qa_progress_path: str) -> None.

    Reset resume artifacts.
    """
    for path in (output_raw_qa, qa_progress_path):
        file_path = Path(path).expanduser()
        if file_path.exists():
            file_path.unlink()


def _has_pending_chunks(
        input_chunks: str,
        limit: int,
        processed_chunk_ids: set[str],
) -> bool:
    """Signature: def _has_pending_chunks(input_chunks: str, limit: int, processed_chunk_ids: set[str]) -> bool.

    Return whether pending chunks.
    """
    for chunk in _iter_chunks(input_chunks, limit):
        if chunk.chunk_id not in processed_chunk_ids:
            return True
    return False


def _record_chunk_progress(
        qa_progress_path: str,
        chunk: ChunkRecord,
        status: str,
        qa_row_count: int,
) -> None:
    """Signature: def _record_chunk_progress(qa_progress_path: str, chunk: ChunkRecord, status: str, qa_row_count: int) -> None.

    Record chunk progress.
    """
    append_jsonl(
        qa_progress_path,
        [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_ref": chunk.source_ref,
                "source_type": chunk.source_type,
                "status": status,
                "qa_row_count": int(qa_row_count),
            }
        ],
    )


def _extract_qa_pairs(model_output: str) -> list[dict[str, str]]:
    """Signature: def _extract_qa_pairs(model_output: str) -> list[dict[str, str]].

    Extract normalized QA pairs from the model output JSON payload.
    """
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
    """Signature: def _extract_question(model_output: str) -> str.

    Extract the generated question text from the model output.
    """
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
    """Signature: def _is_webex_chunk(chunk: ChunkRecord) -> bool.

    Return whether webex chunk.
    """
    if chunk.source_type.strip().lower() == "webex":
        return True
    return chunk.source_ref.strip().lower().startswith("webex::")


def _is_webex_thread_chunk(chunk: ChunkRecord) -> bool:
    """Signature: def _is_webex_thread_chunk(chunk: ChunkRecord) -> bool.

    Return whether webex thread chunk.
    """
    if not _is_webex_chunk(chunk):
        return False
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    if bool(metadata.get("is_thread_document")):
        return True
    return str(metadata.get("webex_grouping", "")).strip().lower() == "thread"


def _normalize_match_value(value: Any) -> str:
    """Signature: def _normalize_match_value(value: Any) -> str.

    Normalize match value.
    """
    if not isinstance(value, str):
        return ""
    return normalize_whitespace(value).lower()


def _webex_thread_messages(chunk: ChunkRecord) -> list[WebexThreadLine]:
    """Signature: def _webex_thread_messages(chunk: ChunkRecord) -> list[WebexThreadLine].

    Webex thread messages.
    """
    return parse_webex_thread_lines(chunk.text)


def _webex_thread_replies(chunk: ChunkRecord) -> list[WebexThreadLine]:
    """Signature: def _webex_thread_replies(chunk: ChunkRecord) -> list[WebexThreadLine].

    Webex thread replies.
    """
    lines = _webex_thread_messages(chunk)
    if len(lines) <= 1:
        return []
    return lines[1:]


def _webex_chunk_matches_user(chunk: ChunkRecord, normalized_user_name: str) -> bool:
    """Signature: def _webex_chunk_matches_user(chunk: ChunkRecord, normalized_user_name: str) -> bool.

    Webex chunk matches user.
    """
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
    """Signature: def _build_webex_thread_answer(chunk: ChunkRecord, normalized_user_name: str = '') -> tuple[str, str, int] | None.

    Build a Webex thread answer from the matching reply messages.
    """
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
    """Signature: def _generate_webex_thread_question(generator: MLXLoadedGenerator, chunk: ChunkRecord, max_tokens: int, temperature: float, normalized_user_name: str = '') -> dict[str, str] | None.

    Generate webex thread question.
    """
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
    """Signature: def main() -> None.

    Generate QA pairs for chunks and write resumable progress files.
    """
    args = parse_args()
    qa_cfg = read_yaml(args.qa_config)
    model_cfg = read_yaml(args.models_config)

    input_chunks = str(qa_cfg.get("input_chunks", "data/staging/chunks/chunks.jsonl"))
    output_raw_qa = str(qa_cfg.get("output_raw_qa", "data/qa/qa_raw.jsonl"))
    qa_progress_path = str(
        qa_cfg.get("qa_progress_path", f"{output_raw_qa}.progress.jsonl")
    )
    resume_generation = bool(qa_cfg.get("resume_generation", True))
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

    input_chunks_path = Path(input_chunks).expanduser()
    output_raw_qa_path = Path(output_raw_qa).expanduser()
    qa_progress_file_path = Path(qa_progress_path).expanduser()
    if not input_chunks_path.exists():
        logger.warning("Input chunks file not found: %s", input_chunks)
        if not output_raw_qa_path.exists():
            write_jsonl(output_raw_qa, [])
        return

    if resume_generation:
        if output_raw_qa_path.exists() or qa_progress_file_path.exists():
            logger.info(
                "Resuming QA generation from prior progress. raw_qa=%s progress=%s",
                output_raw_qa,
                qa_progress_path,
            )
        else:
            logger.info(
                "Resume is enabled but no prior QA progress files were found. Starting the first pass."
            )
        repaired_raw_rows = repair_jsonl_tail(output_raw_qa)
        repaired_progress_rows = repair_jsonl_tail(qa_progress_path)
        if repaired_raw_rows > 0:
            logger.warning(
                "Recovered raw QA output by truncating %s malformed trailing rows from %s.",
                repaired_raw_rows,
                output_raw_qa,
            )
        if repaired_progress_rows > 0:
            logger.warning(
                "Recovered QA progress state by truncating %s malformed trailing rows from %s.",
                repaired_progress_rows,
                qa_progress_path,
            )
        processed_chunk_ids = _load_processed_chunk_ids(output_raw_qa, qa_progress_path)
    else:
        logger.info("Resume disabled. Starting QA generation from scratch.")
        _reset_resume_artifacts(output_raw_qa, qa_progress_path)
        write_jsonl(output_raw_qa, [])
        processed_chunk_ids = set()

    if processed_chunk_ids:
        logger.info(
            "Resume is enabled. %s chunks are already marked as processed and will be skipped.",
            len(processed_chunk_ids),
        )

    if resume_generation and not _has_pending_chunks(
            input_chunks,
            max_chunks,
            processed_chunk_ids,
    ):
        if not output_raw_qa_path.exists():
            write_jsonl(output_raw_qa, [])
        logger.info("No pending chunks left to process. Raw QA output is already up to date.")
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

    resumed_skipped_chunks = 0
    processed_this_run = 0
    written_rows_this_run = 0
    skipped_short_webex_chunks = 0
    skipped_webex_user_filter_chunks = 0
    skipped_webex_thread_without_answer = 0
    generated_webex_thread_qas = 0

    chunks = _iter_chunks(input_chunks, max_chunks)
    for chunk in tqdm(
            chunks,
            desc="Generating QA",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    ):
        if chunk.chunk_id in processed_chunk_ids:
            resumed_skipped_chunks += 1
            continue

        chunk_text = chunk.text.strip()
        chunk_rows: list[dict[str, Any]] = []
        chunk_status = "completed"

        if _is_webex_chunk(chunk) and len(chunk_text) < min_webex_chunk_chars:
            skipped_short_webex_chunks += 1
            chunk_status = "skipped_short_webex"
            logger.debug(
                "Skipping webex chunk %s because %s chars < min_webex_chunk_chars=%s",
                chunk.chunk_id,
                len(chunk_text),
                min_webex_chunk_chars,
            )
        elif (
                _is_webex_chunk(chunk)
                and normalized_webex_user_name
                and not _webex_chunk_matches_user(chunk, normalized_webex_user_name)
        ):
            skipped_webex_user_filter_chunks += 1
            chunk_status = "skipped_webex_user_filter"
            logger.debug(
                "Skipping webex chunk %s because user '%s' was not found in child messages.",
                chunk.chunk_id,
                webex_user_name,
            )
        else:
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
                        chunk_status = "skipped_webex_thread_without_answer"
                        logger.debug(
                            "Skipping webex thread chunk %s because no eligible child-message answer could be built.",
                            chunk.chunk_id,
                        )
                        _record_chunk_progress(
                            qa_progress_path,
                            chunk,
                            chunk_status,
                            qa_row_count=0,
                        )
                        processed_chunk_ids.add(chunk.chunk_id)
                        processed_this_run += 1
                        continue
                    qa_pairs = [webex_pair]
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
                chunk_status = "no_valid_qa"
            else:
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
                    chunk_rows.append(row)

                if not chunk_rows:
                    chunk_status = "filtered_after_validation"

        if chunk_rows:
            appended = append_jsonl(output_raw_qa, chunk_rows)
            written_rows_this_run += appended
            if _is_webex_thread_chunk(chunk):
                generated_webex_thread_qas += appended

        _record_chunk_progress(
            qa_progress_path,
            chunk,
            chunk_status,
            qa_row_count=len(chunk_rows),
        )
        processed_chunk_ids.add(chunk.chunk_id)
        processed_this_run += 1

    if not output_raw_qa_path.exists():
        write_jsonl(output_raw_qa, [])

    if resumed_skipped_chunks > 0:
        logger.info(
            "Skipped %s chunks because they were already processed in an earlier run.",
            resumed_skipped_chunks,
        )

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

    logger.info(
        "QA generation complete. Processed %s chunks in this run, appended %s raw QA rows to %s, and updated resume state in %s.",
        processed_this_run,
        written_rows_this_run,
        output_raw_qa,
        qa_progress_path,
    )


if __name__ == "__main__":
    main()
