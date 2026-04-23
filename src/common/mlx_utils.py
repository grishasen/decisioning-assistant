from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Mapping


def _optional_positive_int(value: Any) -> int | None:
    """Return a positive integer from config-like input, otherwise None."""
    if value in (None, "", False):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _non_negative_int(value: Any, default: int = 0) -> int:
    """Return a non-negative integer from config-like input."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _bool_from_config(value: Any, default: bool = False) -> bool:
    """Return a boolean from YAML/env-style config input."""
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "f", "no", "off"}
    return bool(value)


def mlx_generation_options_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return optional generation settings understood by MLXLoadedGenerator."""
    return {
        "turboquant_fast": _bool_from_config(cfg.get("turboquant_fast", False)),
        "turboquant_kv_bits": _optional_positive_int(
            cfg.get("turboquant_kv_bits", cfg.get("kv_bits"))
        ),
        "turboquant_kv_group_size": _non_negative_int(
            cfg.get("turboquant_kv_group_size", cfg.get("kv_group_size", 64)),
            default=64,
        )
        or 64,
        "turboquant_min_tokens": _non_negative_int(
            cfg.get("turboquant_min_tokens", cfg.get("min_tokens", 0))
        ),
        "turboquant_prefill_step_size": _non_negative_int(
            cfg.get(
                "turboquant_prefill_step_size",
                cfg.get("prefill_step_size", 2048),
            ),
            default=2048,
        )
        or 2048,
    }


def normalize_mlx_provider(provider: str | None) -> str:
    """Return the normalized local MLX generation backend name."""
    cleaned = str(provider or "mlx_lm").strip().lower().replace("-", "_")
    if cleaned in {"mlx", "mlx_lm", "lm", "text"}:
        return "mlx_lm"
    if cleaned in {"mlx_vlm", "vlm", "vision", "multimodal"}:
        return "mlx_vlm"
    if cleaned in {"turboquant", "turboquant_mlx", "tq", "tq_mlx"}:
        return "turboquant_mlx"
    raise ValueError(
        "provider must be one of: mlx, mlx_lm, mlx_vlm, vlm, turboquant_mlx"
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


def _eos_token_ids(tokenizer: Any) -> set[int]:
    """Return all EOS token ids recognized by the tokenizer."""
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return {int(token_id) for token_id in ids}
    primary = getattr(tokenizer, "eos_token_id", None)
    return {int(primary)} if primary is not None else set()


def _make_min_tokens_logits_processor(
    min_tokens: int,
    eos_ids: set[int],
) -> Callable[[Any, Any], Any] | None:
    """Build a small EOS-masking logits processor without importing MLX eagerly."""
    if min_tokens <= 0 or not eos_ids:
        return None

    def processor(tokens: Any, logits: Any) -> Any:
        if tokens.size < min_tokens:
            import mlx.core as mx

            logits[..., mx.array(sorted(eos_ids))] = -float("inf")
        return logits

    return processor


class MLXLoadedGenerator:
    """In-process MLX generator that loads model/tokenizer once per process."""

    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        trust_remote_code: bool = True,
        provider: str | None = "mlx_lm",
        turboquant_fast: bool = False,
        turboquant_kv_bits: int | None = None,
        turboquant_kv_group_size: int = 64,
        turboquant_min_tokens: int = 0,
        turboquant_prefill_step_size: int = 2048,
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

        if self._provider == "turboquant_mlx":
            self._init_turboquant(
                model=model,
                adapter_path=adapter_path,
                fast=turboquant_fast,
                kv_bits=turboquant_kv_bits,
                kv_group_size=turboquant_kv_group_size,
                min_tokens=turboquant_min_tokens,
                prefill_step_size=turboquant_prefill_step_size,
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

    def _init_turboquant(
        self,
        *,
        model: str,
        adapter_path: str | None,
        fast: bool,
        kv_bits: int | None,
        kv_group_size: int,
        min_tokens: int,
        prefill_step_size: int,
    ) -> None:
        """Load a TurboQuant-compressed MLX model for repeated generation."""
        if adapter_path:
            raise RuntimeError(
                "TurboQuant generation does not support adapter_path. Fuse the "
                "adapter into the base model before TurboQuant conversion."
            )

        try:
            import turboquant_mlx.compat  # noqa: F401
            from turboquant_mlx.generate import load_turboquant
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "turboquant_mlx is required for provider=turboquant_mlx. Install "
                "with `pip install -e .[turboquant]` or "
                "`pip install turboquant-mlx-full`."
            ) from exc

        self._generate_fn: Callable[..., str] = generate
        self._make_sampler: Callable[..., Any] = make_sampler
        self._model, self._tokenizer = load_turboquant(model, fast=fast)
        self._turboquant_kv_bits = _optional_positive_int(kv_bits)
        self._turboquant_kv_group_size = max(1, int(kv_group_size or 64))
        self._turboquant_min_tokens = max(0, int(min_tokens or 0))
        self._turboquant_prefill_step_size = max(1, int(prefill_step_size or 2048))

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

    def _turboquant_logits_processors(self) -> list[Callable[[Any, Any], Any]] | None:
        processor = _make_min_tokens_logits_processor(
            self._turboquant_min_tokens,
            _eos_token_ids(self._tokenizer),
        )
        return [processor] if processor is not None else None

    def _generate_turboquant_with_kv_cache(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate with prompt-first TurboQuant KV cache compression."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from turboquant_mlx.layers import convert_cache_to_turboquant

        token_ids = self._tokenizer.encode(prompt)
        if not token_ids:
            raise RuntimeError("Tokenizer returned no prompt tokens")

        prompt_tokens = mx.array(token_ids)
        cache = make_prompt_cache(self._model)

        offset = 0
        while prompt_tokens.size - offset > self._turboquant_prefill_step_size:
            chunk = prompt_tokens[
                offset : offset + self._turboquant_prefill_step_size
            ]
            logits = self._model(chunk[None], cache=cache)
            mx.eval(logits, [c.state for c in cache])
            offset += self._turboquant_prefill_step_size
            mx.clear_cache()

        final_chunk = prompt_tokens[offset:]
        logits = self._model(final_chunk[None], cache=cache)
        mx.eval(logits)

        cache = convert_cache_to_turboquant(
            cache,
            tq_bits=self._turboquant_kv_bits,
            group_size=self._turboquant_kv_group_size,
        )
        for item in cache:
            if hasattr(item, "_tq_keys") and item._tq_keys is not None:
                mx.eval(*item._tq_keys, *item._tq_values)

        sampler = self._make_sampler(temp=temperature)
        logits_processors = self._turboquant_logits_processors() or []
        eos_ids = _eos_token_ids(self._tokenizer)
        generated: list[int] = []

        for _ in range(max_tokens):
            step_logits = logits[:, -1, :]
            generated_array = mx.array(generated)
            for processor in logits_processors:
                step_logits = processor(generated_array, step_logits)

            logprobs = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)
            token = sampler(logprobs)
            mx.eval(token)
            token_id = int(token.item())
            if token_id in eos_ids:
                break

            generated.append(token_id)
            logits = self._model(token.reshape(1, 1), cache=cache)
            mx.eval(logits)

        output = self._tokenizer.decode(generated).strip()
        if not output:
            raise RuntimeError("turboquant_mlx generation returned empty output")
        return output

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

        if self._provider == "turboquant_mlx":
            prepared_prompt = self._prepare_prompt(prompt)
            if self._turboquant_kv_bits is not None:
                return self._generate_turboquant_with_kv_cache(
                    prepared_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            sampler = self._make_sampler(temp=temperature)
            output = self._generate_fn(
                self._model,
                self._tokenizer,
                prepared_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=self._turboquant_logits_processors(),
                verbose=False,
            )
            output = output.strip()
            if not output:
                raise RuntimeError("turboquant_mlx.generate returned empty output")
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
