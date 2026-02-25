from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any

from common.io_utils import iter_jsonl, read_yaml, write_jsonl
from common.logging_utils import get_logger
from common.schemas import QARecord
from common.text_utils import normalize_whitespace

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and clean generated QA data.")
    parser.add_argument("--qa-config", default="configs/qa_generation.yaml")
    return parser.parse_args()


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= 4}


def _overlap_ratio(answer: str, context: str) -> float:
    a = _tokens(answer)
    c = _tokens(context)
    if not a or not c:
        return 0.0
    return len(a.intersection(c)) / float(len(a))


def main() -> None:
    args = parse_args()
    qa_cfg = read_yaml(args.qa_config)

    input_path = str(qa_cfg.get("output_raw_qa", "data/qa/qa_raw.jsonl"))
    output_path = str(qa_cfg.get("output_clean_qa", "data/qa/qa_clean.jsonl"))

    min_question_chars = int(qa_cfg.get("min_question_chars", 12))
    min_answer_chars = int(qa_cfg.get("min_answer_chars", 24))
    max_answer_chars = int(qa_cfg.get("max_answer_chars", 1800))

    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    drops = Counter()

    for raw in iter_jsonl(input_path):
        try:
            qa = QARecord.model_validate(raw)
        except Exception:
            drops["invalid_schema"] += 1
            continue

        question = normalize_whitespace(qa.question)
        answer = normalize_whitespace(qa.answer)
        question_key = question.lower()

        if len(question) < min_question_chars:
            drops["short_question"] += 1
            continue
        if len(answer) < min_answer_chars or len(answer) > max_answer_chars:
            drops["answer_length"] += 1
            continue
        if question_key in seen_questions:
            drops["duplicate_question"] += 1
            continue

        chunk_text = str(qa.metadata.get("chunk_text", ""))
        if _overlap_ratio(answer, chunk_text) < 0.08:
            drops["weak_grounding"] += 1
            continue

        row = qa.model_dump(mode="json")
        row["question"] = question
        row["answer"] = answer
        rows.append(row)
        seen_questions.add(question_key)

    count = write_jsonl(output_path, rows)
    logger.info("Wrote %s cleaned QA rows to %s", count, output_path)
    if drops:
        logger.info("Dropped rows by reason: %s", dict(drops))


if __name__ == "__main__":
    main()
