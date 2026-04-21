from __future__ import annotations

import json
import subprocess
from typing import Any, Callable


def normalize_mlx_provider(provider: str | None) -> str:
    """Return the normalized local MLX generation backend name."""
    cleaned = str(provider or "mlx_lm").strip().lower().replace("-", "_")
    if cleaned in {"mlx", "mlx_lm", "lm", "text"}:
        return "mlx_lm"
    if cleaned in {"mlx_vlm", "vlm", "vision", "multimodal"}:
        return "mlx_vlm"
    raise ValueError(
        "provider must be one of: mlx, mlx_lm, mlx_vlm, vlm"
    )


def _as_media_list(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Return image/audio inputs as a list for mlx-vlm."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else None

    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def run_mlx_generate(
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    adapter_path: str | None = None,
) -> str:
    """Signature: def run_mlx_generate(model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.2, adapter_path: str | None = None) -> str.

    Run mlx_lm.generate in a subprocess and return its text output.
    """
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
        provider: str | None = "mlx_lm",
    ) -> None:
        """Signature: def __init__(self, model: str, adapter_path: str | None = None, trust_remote_code: bool = True) -> None.

        Load the MLX model, tokenizer, and sampler once for repeated generation.
        """
        self._provider = normalize_mlx_provider(provider)

        if self._provider == "mlx_vlm":
            self._init_vlm(
                model=model,
                adapter_path=adapter_path,
                trust_remote_code=trust_remote_code,
            )
            return

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

    def _init_vlm(
        self,
        *,
        model: str,
        adapter_path: str | None,
        trust_remote_code: bool,
    ) -> None:
        """Load an MLX-VLM model/processor for multimodal or VLM text generation."""
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "mlx_vlm is required for provider=mlx_vlm. Install with "
                "`pip install -e .[vlm]` or `pip install mlx-vlm`."
            ) from exc

        load_kwargs: dict[str, Any] = {}
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        self._vlm_generate_fn: Callable[..., Any] = generate
        self._vlm_apply_chat_template: Callable[..., str] = apply_chat_template
        self._model, self._processor = load(
            model,
            adapter_path=adapter_path,
            **load_kwargs,
        )
        self._vlm_config = getattr(self._model, "config", None)

    def _prepare_prompt(self, prompt: str) -> str:
        """Signature: def _prepare_prompt(self, prompt: str) -> str.

        Apply the tokenizer chat template when the loaded model expects one.
        """
        has_chat_template = bool(getattr(self._tokenizer, "has_chat_template", False))
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if not has_chat_template and chat_template is None:
            return prompt

        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_vlm_prompt(
        self,
        prompt: str,
        *,
        images: list[str] | None,
        audio: list[str] | None,
    ) -> str:
        """Apply the MLX-VLM chat template with media counts."""
        return self._vlm_apply_chat_template(
            self._processor,
            self._vlm_config,
            prompt,
            num_images=len(images or []),
            num_audios=len(audio or []),
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        images: str | list[str] | tuple[str, ...] | None = None,
        audio: str | list[str] | tuple[str, ...] | None = None,
    ) -> str:
        """Signature: def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str.

        Generate text from the loaded MLX model.
        """
        if self._provider == "mlx_vlm":
            image_list = _as_media_list(images)
            audio_list = _as_media_list(audio)
            prepared_prompt = self._prepare_vlm_prompt(
                prompt,
                images=image_list,
                audio=audio_list,
            )
            result = self._vlm_generate_fn(
                self._model,
                self._processor,
                prepared_prompt,
                image=image_list,
                audio=audio_list,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            output = getattr(result, "text", result)
            output = str(output).strip()
            if not output:
                raise RuntimeError("mlx_vlm.generate returned empty output")
            return output

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
    """Signature: def extract_first_json_object(text: str) -> dict[str, Any] | None.

    Extract the first top-level JSON object found in generated text.
    """
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
