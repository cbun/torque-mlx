import json
from pathlib import Path

import numpy as np
import pytest

from torque_mlx.cli import main
from torque_mlx.config import TorqueConfig
from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, QwenModelArtifactManifest
from torque_mlx.qwen_benchmark import (
    QwenDecodeBenchmarkProfile,
    QwenDecodeRuntimeBenchmarkResult,
    load_qwen_decode_benchmark_profile,
    run_qwen_decode_runtime_benchmark,
)
from torque_mlx.quantization import Codebook
from torque_mlx.rotation import RotationMode


def _write_qwen_config(model_dir: Path) -> None:
    payload = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "head_dim": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "attn_output_gate": True,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_manifest(model_dir: Path, source_model_dir: Path) -> None:
    manifest = QwenModelArtifactManifest(
        model_name="qwen-benchmark-test",
        source_model_dir=str(source_model_dir),
        runtime_config=TorqueConfig(
            bit_width=4,
            head_dim=256,
            num_layers=4,
            kv_heads=2,
            fused_weights=True,
            rotation_mode=RotationMode.HADAMARD,
            rotation_seed=7,
        ),
        full_attention_indices=[3],
        converted_layer_indices=[3],
        passthrough_layer_indices=[0, 1, 2],
        converted_tensor_names={"3": ["model.layers.3.self_attn.v_proj.weight", "model.layers.3.self_attn.o_proj.weight"]},
        layer_fusion_modes={"3": "vo_only_runtime_qk_rotation"},
        key_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        value_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        source_model_type="qwen3_5",
        source_text_model_type="qwen3_5",
        source_vision_model_type=None,
        has_vision_config=False,
        source_architectures=["Qwen3_5ForCausalLM"],
    )
    (model_dir / QWEN_MODEL_MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_load_qwen_decode_benchmark_profile_from_source_snapshot(tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    _write_qwen_config(model_dir)

    profile = load_qwen_decode_benchmark_profile(model_dir, bit_width=3, rotation_seed=11)

    assert isinstance(profile, QwenDecodeBenchmarkProfile)
    assert profile.profile_source == "hf_snapshot"
    assert profile.head_dim == 256
    assert profile.num_attention_heads == 8
    assert profile.num_key_value_heads == 2
    assert profile.target_layer_indices == (3,)
    assert profile.bit_width == 3
    assert profile.rotation_seed == 11
    assert profile.kv_group_ratio == 4.0


def test_load_qwen_decode_benchmark_profile_from_manifest_recovers_source_geometry(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_qwen_config(source_dir)

    artifact_dir = tmp_path / "torque"
    artifact_dir.mkdir()
    _write_manifest(artifact_dir, source_dir)

    profile = load_qwen_decode_benchmark_profile(artifact_dir)

    assert profile.profile_source == "torque_manifest"
    assert profile.num_attention_heads == 8
    assert profile.num_key_value_heads == 2
    assert profile.bit_width == 4
    assert profile.rotation_seed == 7
    assert profile.target_layer_indices == (3,)


def test_run_qwen_decode_runtime_benchmark_uses_injected_runner(tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    _write_qwen_config(model_dir)

    def _fake_runner(**kwargs):
        profile = kwargs["profile"]
        assert profile.target_layer_count == 1
        assert kwargs["prefill_tokens"] == 128
        assert kwargs["decode_steps"] == 16
        return QwenDecodeRuntimeBenchmarkResult(
            profile=profile,
            prefill_tokens=128,
            decode_steps=16,
            seed=3,
            torque_decode_strategy=kwargs["decode_strategy"],
            fp16_update_seconds=0.2,
            fp16_attention_seconds=0.3,
            fp16_decode_seconds=1.0,
            mlx_lm_quantized_update_seconds=0.4,
            mlx_lm_quantized_attention_seconds=0.5,
            mlx_lm_quantized_decode_seconds=2.0,
            torque_append_seconds=0.1,
            torque_kernel_seconds=0.2,
            torque_decode_seconds=0.5,
            max_abs_error_quantized_vs_fp16=0.1,
            max_abs_error_torque_vs_fp16=0.2,
        )

    result = run_qwen_decode_runtime_benchmark(
        model_dir=model_dir,
        prefill_tokens=128,
        decode_steps=16,
        seed=3,
        bit_width=4,
        runner=_fake_runner,
    )

    payload = result.to_dict()
    assert payload["benchmark"] == "qwen_decode_runtime"
    assert payload["profile"]["target_layer_count"] == 1
    assert payload["torque_decode_strategy"] == "split_batched"
    assert payload["torque_mlx_tokens_per_sec"] == 32.0
    assert payload["torque_mlx_append_ms"] == 100.0


def test_cli_benchmark_qwen_decode_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    model_dir.mkdir()

    def _fake_benchmark(**kwargs):
        assert kwargs["model_dir"] == str(model_dir)
        assert kwargs["prefill_tokens"] == 256
        assert kwargs["decode_steps"] == 32
        profile = QwenDecodeBenchmarkProfile(
            model_dir=str(model_dir),
            profile_source="hf_snapshot",
            model_type="qwen3_5",
            text_model_type="qwen3_5",
            head_dim=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            target_layer_indices=(3,),
            bit_width=4,
            rotation_seed=0,
            fused_weights=False,
            has_vision_config=False,
            attn_output_gate=True,
        )
        return QwenDecodeRuntimeBenchmarkResult(
            profile=profile,
            prefill_tokens=256,
            decode_steps=32,
            seed=0,
            torque_decode_strategy=kwargs["decode_strategy"],
            fp16_update_seconds=0.2,
            fp16_attention_seconds=0.3,
            fp16_decode_seconds=1.0,
            mlx_lm_quantized_update_seconds=0.4,
            mlx_lm_quantized_attention_seconds=0.5,
            mlx_lm_quantized_decode_seconds=1.5,
            torque_append_seconds=0.1,
            torque_kernel_seconds=0.2,
            torque_decode_seconds=2.0,
            max_abs_error_quantized_vs_fp16=0.1,
            max_abs_error_torque_vs_fp16=0.2,
        )

    monkeypatch.setattr("torque_mlx.cli.run_qwen_decode_runtime_benchmark", _fake_benchmark)

    exit_code = main(
        [
            "benchmark",
            "qwen-decode",
            "--model-dir",
            str(model_dir),
            "--prefill-tokens",
            "256",
            "--decode-steps",
            "32",
            "--decode-strategy",
            "split_batched",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "qwen_decode_runtime"
    assert payload["profile"]["target_layer_count"] == 1
    assert payload["torque_decode_strategy"] == "split_batched"
