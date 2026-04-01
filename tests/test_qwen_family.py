import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from torque_mlx.conversion import fuse_attention_weights
from torque_mlx.families.qwen import (
    convert_qwen_attention_layer,
    convert_qwen_model,
    inspect_qwen_hf_directory,
    load_qwen_model_manifest,
)
from torque_mlx.rotation import RotationSpec


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _write_qwen_config(
    path: Path,
    *,
    head_dim: int,
    num_hidden_layers: int = 8,
) -> None:
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (num_hidden_layers // 4)
    payload = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "head_dim": head_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "layer_types": layer_types,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **dict(**__import__("os").environ),
        "PYTHONPATH": str(SRC),
    }
    return subprocess.run(
        [sys.executable, "-m", "torque_mlx", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _write_qwen_snapshot(model_dir: Path, *, head_dim: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    _write_qwen_config(model_dir / "config.json", head_dim=head_dim, num_hidden_layers=4)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    rng = np.random.default_rng(123)
    tensors = {
        "model.layers.3.self_attn.q_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.k_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.v_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.o_proj.weight": rng.normal(size=(48, head_dim)).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.embed_tokens.weight": rng.normal(size=(16, 32)).astype(np.float32),
    }
    weight_map = {
        "model.layers.3.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.3.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.3.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.3.self_attn.o_proj.weight": "model-00002-of-00002.safetensors",
        "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
    }
    shard1 = {
        key: value
        for key, value in tensors.items()
        if weight_map[key] == "model-00001-of-00002.safetensors"
    }
    shard2 = {
        key: value
        for key, value in tensors.items()
        if weight_map[key] == "model-00002-of-00002.safetensors"
    }
    save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return tensors, weight_map


def test_qwen_plan_reports_unsupported_head_dim(tmp_path: Path) -> None:
    _write_qwen_config(tmp_path / "config.json", head_dim=192)
    report = inspect_qwen_hf_directory(tmp_path)

    assert report.supported_runtime is False
    assert report.full_attention_indices == [3, 7]
    assert "Unsupported head_dim 192" in report.blocking_issues[0]


def test_qwen_layer_conversion_for_supported_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_config(model_dir / "config.json", head_dim=256)

    rng = np.random.default_rng(0)
    weights_path = tmp_path / "layer3.npz"
    np.savez_compressed(
        weights_path,
        w_q=rng.normal(size=(256, 32)).astype(np.float32),
        w_k=rng.normal(size=(256, 32)).astype(np.float32),
        w_v=rng.normal(size=(256, 32)).astype(np.float32),
        w_o=rng.normal(size=(48, 256)).astype(np.float32),
    )

    artifact_dir = tmp_path / "artifact"
    artifact = convert_qwen_attention_layer(
        model_dir=model_dir,
        layer_idx=3,
        input_weights=weights_path,
        output_dir=artifact_dir,
    )

    assert artifact.manifest.architecture == "qwen_full_attention_layer"
    assert artifact.runtime_config.head_dim == 256
    assert artifact.manifest.extra_metadata["converted_layer_idx"] == 3


def test_cli_qwen_plan_outputs_supported_report(tmp_path: Path) -> None:
    _write_qwen_config(tmp_path / "config.json", head_dim=256)
    result = _run_cli("plan", "qwen", "--model-dir", str(tmp_path))
    payload = json.loads(result.stdout)

    assert payload["family"] == "qwen"
    assert payload["supported_runtime"] is True
    assert payload["full_attention_indices"] == [3, 7]


def test_qwen_model_conversion_rewrites_only_supported_attention_layers(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors, _ = _write_qwen_snapshot(model_dir, head_dim=256)

    output_dir = tmp_path / "converted"
    manifest = convert_qwen_model(model_dir=model_dir, output_dir=output_dir, model_name="qwen-demo")

    loaded_manifest = load_qwen_model_manifest(output_dir)
    assert manifest.model_name == "qwen-demo"
    assert loaded_manifest.converted_layer_indices == [3]
    assert loaded_manifest.passthrough_layer_indices == [0, 1, 2]

    rotation = RotationSpec.from_seed(head_dim=256, seed=0)
    expected = fuse_attention_weights(
        w_q=tensors["model.layers.3.self_attn.q_proj.weight"],
        w_k=tensors["model.layers.3.self_attn.k_proj.weight"],
        w_v=tensors["model.layers.3.self_attn.v_proj.weight"],
        w_o=tensors["model.layers.3.self_attn.o_proj.weight"],
        rotation=rotation,
    )

    with safe_open(str(output_dir / "model-00001-of-00002.safetensors"), framework="np") as handle:
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.layers.3.self_attn.q_proj.weight")),
            expected.w_q,
        )
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.layers.3.self_attn.k_proj.weight")),
            expected.w_k,
        )
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.layers.0.self_attn.q_proj.weight")),
            tensors["model.layers.0.self_attn.q_proj.weight"],
        )

    with safe_open(str(output_dir / "model-00002-of-00002.safetensors"), framework="np") as handle:
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.layers.3.self_attn.v_proj.weight")),
            expected.w_v,
        )
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.layers.3.self_attn.o_proj.weight")),
            expected.w_o,
        )
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.embed_tokens.weight")),
            tensors["model.embed_tokens.weight"],
        )

    assert (output_dir / "tokenizer.json").exists()


def test_cli_qwen_model_convert_and_inspect(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_snapshot(model_dir, head_dim=256)

    output_dir = tmp_path / "converted"
    convert = _run_cli(
        "convert-qwen-model",
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--model-name",
        "qwen-cli",
    )
    convert_payload = json.loads(convert.stdout)
    assert convert_payload["model_name"] == "qwen-cli"
    assert convert_payload["converted_layer_indices"] == [3]

    inspect = _run_cli("inspect-qwen-model", "--artifact", str(output_dir))
    inspect_payload = json.loads(inspect.stdout)
    assert inspect_payload["model_name"] == "qwen-cli"
    assert inspect_payload["converted_tensor_count"] == 4
