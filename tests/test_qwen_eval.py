import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torque_mlx.cli import main
from torque_mlx.config import TorqueConfig
from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, QwenModelArtifactManifest
from torque_mlx.qwen_eval import (
    QwenTextBenchmarkComparison,
    QwenTextPerplexityResult,
    _apply_qwen_delta_overrides,
    _load_qwen_delta_overrides,
    _resolve_qwen_override_state_dict_key,
    _restore_stacked_input_projection,
    _runtime_unrotate_attention_output,
    benchmark_qwen_text_models,
    evaluate_qwen_text_perplexity,
)
from torque_mlx.quantization import Codebook
from torque_mlx.rotation import RotationMode, RotationSpec


class _FakeTokenizer:
    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        token_ids = [int(item) for item in text.split()]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}


class _PerfectNextTokenModel:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.dtype = torch.float32

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -100.0, dtype=torch.float32, device=input_ids.device)
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len - 1):
                next_token = int(input_ids[batch_idx, token_idx + 1].item())
                logits[batch_idx, token_idx, next_token] = 0.0
            logits[batch_idx, seq_len - 1, 0] = 0.0
        return SimpleNamespace(logits=logits)


def _fake_loader(*, model_dir, device, dtype):
    assert Path(model_dir).exists()
    return _PerfectNextTokenModel(), _FakeTokenizer(), torch.device("cpu"), torch.float32


def _write_manifest(model_dir: Path) -> None:
    manifest = QwenModelArtifactManifest(
        model_name="qwen-eval-test",
        source_model_dir=str(model_dir),
        runtime_config=TorqueConfig(
            bit_width=4,
            head_dim=256,
            num_layers=4,
            kv_heads=4,
            fused_weights=True,
            rotation_mode=RotationMode.HADAMARD,
            rotation_seed=0,
        ),
        full_attention_indices=[3],
        converted_layer_indices=[3],
        passthrough_layer_indices=[0, 1, 2],
        converted_tensor_names={"3": ["model.layers.3.self_attn.q_proj.weight"]},
        layer_fusion_modes={"3": "full_qkvo"},
        key_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        value_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        source_model_type="qwen3_5",
        source_text_model_type="qwen3_5_text",
        source_vision_model_type="qwen2_vl",
        has_vision_config=True,
        source_architectures=["Qwen3_5ForConditionalGeneration"],
    )
    (model_dir / QWEN_MODEL_MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_delta_manifest(model_dir: Path, source_model_dir: Path) -> None:
    manifest = QwenModelArtifactManifest(
        model_name="qwen-delta-eval-test",
        source_model_dir=str(source_model_dir),
        runtime_config=TorqueConfig(
            bit_width=4,
            head_dim=256,
            num_layers=4,
            kv_heads=4,
            fused_weights=True,
            rotation_mode=RotationMode.HADAMARD,
            rotation_seed=0,
        ),
        full_attention_indices=[3],
        converted_layer_indices=[3],
        passthrough_layer_indices=[0, 1, 2],
        converted_tensor_names={"3": ["linear.weight"]},
        layer_fusion_modes={"3": "full_qkvo"},
        key_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        value_codebook=Codebook(bit_width=4, centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32), name="gaussian"),
        source_model_type="qwen3_5",
        source_text_model_type="qwen3_5_text",
        source_vision_model_type=None,
        has_vision_config=False,
        source_architectures=["Qwen3_5ForConditionalGeneration"],
        artifact_layout="delta_npz",
        delta_weights_file="torque_qwen_delta_weights.npz",
        stored_asset_files=["config.json", "tokenizer.json"],
    )
    (model_dir / QWEN_MODEL_MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_qwen_text_perplexity_reports_manifest_and_size(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights_path = model_dir / "model-00001-of-00001.safetensors"
    weights_path.write_bytes(b"torque-weights")
    _write_manifest(model_dir)

    text_file = tmp_path / "wiki.test.raw"
    text_file.write_text("1 2 3 4 5 6", encoding="utf-8")

    result = evaluate_qwen_text_perplexity(
        model_dir=model_dir,
        text_file=text_file,
        context_length=4,
        stride=2,
        loader=_fake_loader,
    )

    assert result.token_count == 6
    assert result.evaluated_token_count == 5
    assert result.window_count == 3
    assert result.perplexity == pytest.approx(1.0, abs=1e-4)
    assert result.model_safetensor_bytes == len(b"torque-weights")
    assert result.is_torque_converted is True
    assert result.torque_manifest is not None
    assert result.torque_manifest["model_name"] == "qwen-eval-test"
    assert result.loader_seconds >= 0.0
    assert result.evaluation_seconds >= 0.0
    assert result.total_seconds >= result.loader_seconds
    assert result.evaluated_tokens_per_second >= 0.0


def test_qwen_delta_override_loader_applies_state_dict_patch(tmp_path: Path) -> None:
    source_model_dir = tmp_path / "source"
    source_model_dir.mkdir()
    artifact_dir = tmp_path / "delta"
    artifact_dir.mkdir()
    _write_delta_manifest(artifact_dir, source_model_dir)

    np.savez_compressed(
        artifact_dir / "torque_qwen_delta_weights.npz",
        **{"linear.weight": np.full((2, 3), 7.0, dtype=np.float32)},
    )

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 2, bias=False)

    model = _DummyModel()
    manifest = QwenModelArtifactManifest.from_dict(
        json.loads((artifact_dir / QWEN_MODEL_MANIFEST_FILE).read_text(encoding="utf-8"))
    )

    overrides = _load_qwen_delta_overrides(artifact_dir, manifest)
    assert set(overrides) == {"linear.weight"}

    _apply_qwen_delta_overrides(model, artifact_dir, manifest)
    torch.testing.assert_close(
        model.linear.weight.detach(),
        torch.full((2, 3), 7.0, dtype=model.linear.weight.dtype),
    )


def test_qwen_delta_override_name_resolution_handles_language_model_prefix() -> None:
    state_dict = {
        "model.layers.3.self_attn.v_proj.weight": torch.zeros((2, 2)),
    }
    resolved = _resolve_qwen_override_state_dict_key(
        "model.language_model.layers.3.self_attn.v_proj.weight",
        state_dict,
    )
    assert resolved == "model.layers.3.self_attn.v_proj.weight"


def test_qwen_text_perplexity_rejects_too_short_input(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x")
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("1", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two tokens"):
        evaluate_qwen_text_perplexity(
            model_dir=model_dir,
            text_file=text_file,
            loader=_fake_loader,
        )


def test_cli_eval_qwen_text_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text_file = tmp_path / "wiki.test.raw"
    text_file.write_text("1 2 3", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def _fake_eval(**kwargs):
        assert kwargs["model_dir"] == str(model_dir)
        assert kwargs["text_file"] == str(text_file)
        return QwenTextPerplexityResult(
            model_dir=str(model_dir),
            text_file=str(text_file),
            context_length=2048,
            stride=2048,
            device="cpu",
            torch_dtype="float32",
            token_count=3,
            evaluated_token_count=2,
            window_count=1,
            average_negative_log_likelihood=0.0,
            perplexity=1.0,
            model_safetensor_bytes=123,
            model_safetensor_gib=123 / float(1 << 30),
            is_torque_converted=False,
            torque_manifest=None,
        )

    monkeypatch.setattr("torque_mlx.cli.evaluate_qwen_text_perplexity", _fake_eval)

    exit_code = main(
        [
            "eval-qwen-text",
            "--model-dir",
            str(model_dir),
            "--text-file",
            str(text_file),
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "qwen"
    assert payload["perplexity"] == 1.0


def test_qwen_text_benchmark_compares_source_and_torque(tmp_path: Path) -> None:
    source_model_dir = tmp_path / "source"
    source_model_dir.mkdir()
    (source_model_dir / "model-00001-of-00001.safetensors").write_bytes(b"source-weights")

    torque_model_dir = tmp_path / "torque"
    torque_model_dir.mkdir()
    (torque_model_dir / "model-00001-of-00001.safetensors").write_bytes(b"torque-weights")
    _write_manifest(torque_model_dir)

    text_file = tmp_path / "wiki.test.raw"
    text_file.write_text("1 2 3 4 5 6 7 8", encoding="utf-8")

    benchmark = benchmark_qwen_text_models(
        source_model_dir=source_model_dir,
        torque_model_dir=torque_model_dir,
        text_file=text_file,
        context_lengths=[4, 6],
        max_tokens=6,
        loader=_fake_loader,
    )

    assert isinstance(benchmark, QwenTextBenchmarkComparison)
    assert len(benchmark.cases) == 2
    assert benchmark.cases[0].source.perplexity == pytest.approx(1.0, abs=1e-4)
    assert benchmark.cases[0].torque.is_torque_converted is True
    payload = benchmark.to_dict()
    assert payload["benchmark"] == "source_vs_torque_text"
    assert payload["cases"][0]["delta"]["perplexity"] == pytest.approx(0.0, abs=1e-4)
    assert payload["cases"][0]["delta"]["model_safetensor_bytes"] == len(b"torque-weights") - len(b"source-weights")


def test_cli_benchmark_qwen_text_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text_file = tmp_path / "wiki.test.raw"
    text_file.write_text("1 2 3", encoding="utf-8")
    source_model_dir = tmp_path / "source"
    source_model_dir.mkdir()
    torque_model_dir = tmp_path / "torque"
    torque_model_dir.mkdir()

    def _fake_benchmark(**kwargs):
        assert kwargs["source_model_dir"] == str(source_model_dir)
        assert kwargs["torque_model_dir"] == str(torque_model_dir)
        assert kwargs["text_file"] == str(text_file)
        return QwenTextBenchmarkComparison(
            source_model_dir=str(source_model_dir),
            torque_model_dir=str(torque_model_dir),
            text_file=str(text_file),
            device="cpu",
            torch_dtype="float32",
            cases=(),
        )

    monkeypatch.setattr("torque_mlx.cli.benchmark_qwen_text_models", _fake_benchmark)

    exit_code = main(
        [
            "benchmark",
            "qwen-text",
            "--source-model-dir",
            str(source_model_dir),
            "--torque-model-dir",
            str(torque_model_dir),
            "--text-file",
            str(text_file),
            "--context-length",
            "512",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "source_vs_torque_text"
    assert payload["family"] == "qwen"


def test_vo_only_runtime_patch_restores_gated_projection_semantics() -> None:
    num_heads = 2
    head_dim = 64
    out_dim = 3
    rotation = RotationSpec.from_seed(head_dim=head_dim, seed=0)
    rotation_matrix = torch.tensor(rotation.matrix(), dtype=torch.float32)

    attn_output = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float32)
    gate = torch.randn(1, 1, num_heads * head_dim, dtype=torch.float32)
    original_o = torch.randn(out_dim, num_heads * head_dim, dtype=torch.float32)

    rotated_attn_output = torch.matmul(attn_output, rotation_matrix.T)
    converted_o = original_o.reshape(out_dim, num_heads, head_dim)
    converted_o = torch.matmul(converted_o, rotation_matrix.T).reshape(out_dim, num_heads * head_dim)

    original_hidden = (attn_output.reshape(1, 1, -1) * torch.sigmoid(gate)) @ original_o.T

    restored_o = _restore_stacked_input_projection(
        converted_o,
        head_dim=head_dim,
        num_blocks=num_heads,
        rotation_matrix=rotation_matrix,
    )
    corrected_hidden = _runtime_unrotate_attention_output(
        rotated_attn_output,
        rotation_matrix=rotation_matrix,
    ).reshape(1, 1, -1)
    corrected_hidden = (corrected_hidden * torch.sigmoid(gate)) @ restored_o.T

    torch.testing.assert_close(corrected_hidden, original_hidden)
