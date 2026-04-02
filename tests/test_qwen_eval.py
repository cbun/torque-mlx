import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torque_mlx.cli import main
from torque_mlx.config import TorqueConfig
from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, QwenModelArtifactManifest
from torque_mlx.qwen_eval import QwenTextPerplexityResult, evaluate_qwen_text_perplexity
from torque_mlx.quantization import Codebook
from torque_mlx.rotation import RotationMode


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
