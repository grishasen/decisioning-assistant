from __future__ import annotations

import argparse
import re
from statistics import mean
from typing import Any

from tqdm import tqdm

from common.io_utils import iter_jsonl, read_yaml
from common.logging_utils import get_logger
from common.mlx_utils import MLXLoadedGenerator
from common.text_utils import normalize_whitespace

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick lexical evaluation on QA test split.")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--test-path", default="data/qa/test.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--adapter-path", default="")
    return parser.parse_args()


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    common = 0
    for tok in pred_tokens:
        if gold_counts.get(tok, 0) > 0:
            common += 1
            gold_counts[tok] -= 1

    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    args = parse_args()
    models_cfg = read_yaml(args.models_config)
    answer_cfg: dict[str, Any] = models_cfg.get("answer_model", {})

    model = str(answer_cfg.get("model", "mlx-community/gemma-2-2b-it-4bit"))
    max_tokens = int(answer_cfg.get("max_tokens", 400))
    temperature = float(answer_cfg.get("temperature", 0.2))
    trust_remote_code = bool(answer_cfg.get("trust_remote_code", True))

    adapter_path = args.adapter_path.strip() or answer_cfg.get("adapter_path") or None

    rows = list(iter_jsonl(args.test_path))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        logger.warning("No rows found at %s", args.test_path)
        return

    generator = MLXLoadedGenerator(
        model=model,
        adapter_path=str(adapter_path) if adapter_path else None,
        trust_remote_code=trust_remote_code,
    )

    scores: list[float] = []
    for item in tqdm(rows, desc="Evaluating"):
        question = item.get("question", "")
        gold_answer = normalize_whitespace(item.get("answer", ""))
        if not question or not gold_answer:
            continue

        prompt = f"Question: {question}\nAnswer in concise technical style."
        try:
            prediction = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generation failed for question: %s (%s)", question, exc)
            continue

        score = token_f1(normalize_whitespace(prediction), gold_answer)
        scores.append(score)

    if not scores:
        logger.warning("No scores calculated.")
        return

    logger.info("Samples: %s", len(scores))
    logger.info("Token-F1 mean: %.4f", mean(scores))


if __name__ == "__main__":
    main()
