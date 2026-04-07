import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from torque_mlx.cli import build_parser
from torque_mlx.qwen_mlx import (
    QwenMLXGenerationResult,
    QwenMLXRuntimeProfile,
    _estimate_qwen_full_attention_kv_bytes,
    _cast_qwen_weights_to_model_dtypes,
    _normalize_qwen3_5_text_config,
    _resolve_qwen_weight_target,
    _sanitize_qwen3_5_weights,
)


def test_normalize_qwen3_5_text_config_sets_qwen3_next_defaults() -> None:
    config = _normalize_qwen3_5_text_config(
        {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 3584,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "head_dim": 256,
            "max_position_embeddings": 32768,
            "full_attention_interval": 4,
            "rope_parameters": {
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )

    assert config["model_type"] == "qwen3_next"
    assert config["rope_theta"] == 10_000_000.0
    assert config["partial_rotary_factor"] == 0.25
    assert config["num_experts"] == 0
    assert config["decoder_sparse_step"] == 1
    assert config["tie_word_embeddings"] is True


def test_sanitize_qwen3_5_weights_combines_linear_attention_inputs() -> None:
    import mlx.core as mx

    weights = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": mx.array(np.ones((6, 4), dtype=np.float32)),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": mx.array(np.full((2, 4), 3.0, dtype=np.float32)),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": mx.array(np.full((1, 4), 5.0, dtype=np.float32)),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": mx.array(np.full((1, 4), 7.0, dtype=np.float32)),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.array(np.ones((8, 1, 4), dtype=np.float32)),
        "model.language_model.layers.0.input_layernorm.weight": mx.array(np.ones((4,), dtype=np.float32)),
        "model.visual.patch_embed.proj.weight": mx.array(np.ones((2, 2), dtype=np.float32)),
        "mtp.fc.weight": mx.array(np.ones((2, 2), dtype=np.float32)),
    }

    sanitized = _sanitize_qwen3_5_weights(
        weights,
        mx=mx,
        tie_word_embeddings=True,
    )

    assert "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight" in sanitized
    assert "model.language_model.layers.0.linear_attn.in_proj_ba.weight" in sanitized
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" not in sanitized
    assert "model.language_model.layers.0.linear_attn.in_proj_z.weight" not in sanitized
    assert "model.visual.patch_embed.proj.weight" not in sanitized
    assert "mtp.fc.weight" not in sanitized

    qkvz = np.array(sanitized["model.language_model.layers.0.linear_attn.in_proj_qkvz.weight"])
    ba = np.array(sanitized["model.language_model.layers.0.linear_attn.in_proj_ba.weight"])
    conv = np.array(sanitized["model.language_model.layers.0.linear_attn.conv1d.weight"])
    norm = np.array(sanitized["model.language_model.layers.0.input_layernorm.weight"])

    assert qkvz.shape == (8, 4)
    assert ba.shape == (2, 4)
    assert conv.shape == (8, 4, 1)
    np.testing.assert_allclose(norm, np.full((4,), 2.0, dtype=np.float32))


def test_cli_parser_accepts_qwen_generate_benchmark() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "qwen-generate",
            "--model-dir",
            "./artifact",
            "--prompt",
            "hello",
        ],
    )

    assert args.command == "benchmark"
    assert args.benchmark_command == "qwen-generate"
    assert args.max_tokens == 64
    assert args.prefill_step_size == 512
    assert args.decode_tail_capacity is None
    assert args.ignore_eos is False
    assert args.profile_runtime is False


def test_cli_parser_accepts_qwen_generate_profile_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "qwen-generate",
            "--model-dir",
            "./artifact",
            "--prompt",
            "hello",
            "--profile-runtime",
        ],
    )

    assert args.profile_runtime is True


def test_cli_parser_accepts_qwen_generate_decode_tail_capacity() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "qwen-generate",
            "--model-dir",
            "./artifact",
            "--prompt",
            "hello",
            "--decode-tail-capacity",
            "96",
        ],
    )

    assert args.decode_tail_capacity == 96


def test_cli_parser_accepts_qwen_generate_ignore_eos_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "qwen-generate",
            "--model-dir",
            "./artifact",
            "--prompt",
            "hello",
            "--ignore-eos",
        ],
    )

    assert args.ignore_eos is True


def test_qwen_mlx_generation_result_serializes_runtime_profile() -> None:
    result = QwenMLXGenerationResult(
        model_dir="/tmp/model",
        prompt="hello",
        max_tokens=8,
        prefill_step_size=128,
        decode_tail_capacity=16,
        ignore_eos=True,
        generated_text=" world",
        prompt_tokens=5,
        prompt_tokens_per_second=10.0,
        generation_tokens=8,
        generation_tokens_per_second=4.0,
        peak_memory_gb=1.25,
        kv_cache_tokens=13,
        full_attention_layer_count=2,
        full_attention_kv_fp16_bytes_estimate=13312,
        full_attention_kv_packed_bytes_estimate=3328,
        full_attention_kv_bytes_saved_estimate=9984,
        is_torque_converted=True,
        artifact_layout="delta_npz",
        converted_layer_indices=(3, 7),
        prompt_seconds_estimate=0.5,
        generation_seconds_estimate=2.0,
        runtime_profile=QwenMLXRuntimeProfile(
            dense_prefill_seconds=0.1,
            dense_prefill_calls=2,
            dense_prefill_tokens=32,
            prompt_append_seconds=0.12,
            prompt_append_calls=3,
            prompt_append_tokens=24,
            decode_append_seconds=0.08,
            decode_append_calls=1,
            decode_append_tokens=16,
            torque_append_seconds=0.2,
            torque_append_calls=4,
            torque_append_tokens=40,
            torque_decode_seconds=0.3,
            torque_decode_calls=8,
            torque_decode_tokens=8,
            torque_score_seconds=0.11,
            torque_softmax_seconds=0.07,
            torque_value_seconds=0.09,
            torque_tail_seconds=0.03,
        ),
    )

    payload = result.to_dict()

    assert payload["prompt_seconds_estimate"] == 0.5
    assert payload["generation_seconds_estimate"] == 2.0
    assert payload["decode_tail_capacity"] == 16
    assert payload["ignore_eos"] is True
    assert payload["kv_cache_tokens"] == 13
    assert payload["full_attention_layer_count"] == 2
    assert payload["full_attention_kv_fp16_bytes_estimate"] == 13312
    assert payload["full_attention_kv_packed_bytes_estimate"] == 3328
    assert payload["full_attention_kv_bytes_saved_estimate"] == 9984
    assert payload["runtime_profile"] == {
        "dense_prefill_seconds": 0.1,
        "dense_prefill_calls": 2,
        "dense_prefill_tokens": 32,
        "prompt_append_seconds": 0.12,
        "prompt_append_calls": 3,
        "prompt_append_tokens": 24,
        "decode_append_seconds": 0.08,
        "decode_append_calls": 1,
        "decode_append_tokens": 16,
        "torque_append_seconds": 0.2,
        "torque_append_calls": 4,
        "torque_append_tokens": 40,
        "torque_decode_seconds": 0.3,
        "torque_decode_calls": 8,
        "torque_decode_tokens": 8,
        "torque_score_seconds": 0.11,
        "torque_softmax_seconds": 0.07,
        "torque_value_seconds": 0.09,
        "torque_tail_seconds": 0.03,
    }


def test_estimate_qwen_full_attention_kv_bytes_for_source_and_torque() -> None:
    config = {
        "text_config": {
            "head_dim": 256,
            "num_key_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        },
    }

    source_fp16, source_packed, source_saved, source_layers = _estimate_qwen_full_attention_kv_bytes(
        config=config,
        manifest=None,
        kv_cache_tokens=100,
    )
    assert source_layers == 2
    assert source_fp16 == 100 * 2 * 2 * 256 * 2 * 2
    assert source_packed is None
    assert source_saved is None

    manifest = SimpleNamespace(
        converted_layer_indices=(1, 3),
        runtime_config=SimpleNamespace(bit_width=4, head_dim=256, kv_heads=2),
    )
    packed_fp16, packed_bytes, packed_saved, packed_layers = _estimate_qwen_full_attention_kv_bytes(
        config=config,
        manifest=manifest,
        kv_cache_tokens=100,
    )
    assert packed_layers == 2
    assert packed_fp16 == source_fp16
    assert packed_bytes == 100 * 2 * 2 * 32 * 4 * 2
    assert packed_saved == packed_fp16 - packed_bytes


def test_cast_qwen_weights_to_model_dtypes_uses_existing_parameter_dtype() -> None:
    import mlx.core as mx

    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            v_proj=SimpleNamespace(
                                weight=mx.zeros((2, 2), dtype=mx.bfloat16),
                            ),
                        ),
                    ),
                ],
            ),
        ),
    )
    target = _resolve_qwen_weight_target(
        model,
        "model.language_model.layers.0.self_attn.v_proj.weight",
    )

    assert target.dtype == mx.bfloat16

    weights = _cast_qwen_weights_to_model_dtypes(
        model,
        {
            "model.language_model.layers.0.self_attn.v_proj.weight": mx.array(
                np.ones((2, 2), dtype=np.float32),
            ),
        },
        mx=mx,
    )

    assert weights[0][0] == "model.language_model.layers.0.self_attn.v_proj.weight"
    assert weights[0][1].dtype == mx.bfloat16
