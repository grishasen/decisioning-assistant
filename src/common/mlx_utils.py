from __future__ import annotations

import json
import subprocess
from typing import Any, Callable


def run_mlx_generate(
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    adapter_path: str | None = None,
) -> str:
    cmd = [
        "mlx_lm.generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temperature),
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("mlx_lm.generate returned empty output")
    return output


class MLXLoadedGenerator:
    """In-process MLX generator that loads model/tokenizer once per process."""

    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "mlx_lm is required for in-process generation. Install mlx-lm first."
            ) from exc

        tokenizer_config: dict[str, Any] = {}
        if trust_remote_code:
            tokenizer_config["trust_remote_code"] = True

        self._generate_fn: Callable[..., str] = generate
        self._make_sampler: Callable[..., Any] = make_sampler
        self._model, self._tokenizer = load(
            model,
            adapter_path=adapter_path,
            tokenizer_config=tokenizer_config,
        )

    def _prepare_prompt(self, prompt: str) -> str:
        if not getattr(self._tokenizer, "has_chat_template", False):
            return prompt

        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        prepared_prompt = self._prepare_prompt(prompt)
        sampler = self._make_sampler(temp=temperature)
        output = self._generate_fn(
            self._model,
            self._tokenizer,
            prepared_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        output = output.strip()
        if not output:
            raise RuntimeError("mlx_lm.generate returned empty output")
        return output


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
