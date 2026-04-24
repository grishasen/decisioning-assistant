from __future__ import annotations

import threading

import pytest

from common.mlx_utils import (
    MLXThreadedGenerator,
    mlx_generation_options_from_config,
    normalize_mlx_provider,
)


def test_normalize_mlx_provider_accepts_turboquant_aliases() -> None:
    assert normalize_mlx_provider("turboquant") == "turboquant_mlx"
    assert normalize_mlx_provider("turboquant-mlx") == "turboquant_mlx"
    assert normalize_mlx_provider("tq") == "turboquant_mlx"


def test_normalize_mlx_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="provider must be one of"):
        normalize_mlx_provider("not-a-provider")


def test_mlx_generation_options_from_config_parses_turboquant_keys() -> None:
    options = mlx_generation_options_from_config(
        {
            "turboquant_fast": True,
            "turboquant_kv_bits": "3",
            "turboquant_kv_group_size": "32",
            "turboquant_min_tokens": "8",
            "turboquant_prefill_step_size": "1024",
        }
    )

    assert options == {
        "turboquant_fast": True,
        "turboquant_kv_bits": 3,
        "turboquant_kv_group_size": 32,
        "turboquant_min_tokens": 8,
        "turboquant_prefill_step_size": 1024,
    }


def test_mlx_generation_options_from_config_accepts_generic_kv_aliases() -> None:
    options = mlx_generation_options_from_config(
        {
            "kv_bits": 4,
            "kv_group_size": 128,
            "min_tokens": 5,
            "prefill_step_size": 512,
        }
    )

    assert options["turboquant_kv_bits"] == 4
    assert options["turboquant_kv_group_size"] == 128
    assert options["turboquant_min_tokens"] == 5
    assert options["turboquant_prefill_step_size"] == 512


def test_mlx_generation_options_from_config_parses_false_string() -> None:
    options = mlx_generation_options_from_config({"turboquant_fast": "false"})

    assert options["turboquant_fast"] is False


def test_threaded_generator_runs_generation_on_loader_thread() -> None:
    main_thread_id = threading.get_ident()
    init_thread_ids: list[int] = []
    generate_thread_ids: list[int] = []

    class FakeGenerator:
        def generate(self, prompt: str, **kwargs: object) -> str:
            generate_thread_ids.append(threading.get_ident())
            return f"{prompt}:{kwargs['max_tokens']}"

    def factory() -> FakeGenerator:
        init_thread_ids.append(threading.get_ident())
        return FakeGenerator()

    generator = MLXThreadedGenerator(factory)
    try:
        assert generator.generate("hello", max_tokens=7) == "hello:7"
    finally:
        generator.close()

    assert init_thread_ids
    assert generate_thread_ids == init_thread_ids
    assert init_thread_ids[0] != main_thread_id


def test_threaded_generator_propagates_generation_errors() -> None:
    class FakeGenerator:
        def generate(self, prompt: str) -> str:
            raise ValueError(f"bad prompt: {prompt}")

    generator = MLXThreadedGenerator(FakeGenerator)
    try:
        with pytest.raises(ValueError, match="bad prompt: hello"):
            generator.generate("hello")
    finally:
        generator.close()
