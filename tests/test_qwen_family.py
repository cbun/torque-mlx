import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file
import torch

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
    include_vision_config: bool = False,
    num_attention_heads: int = 1,
    num_key_value_heads: int = 1,
    attn_output_gate: bool = False,
) -> None:
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (num_hidden_layers // 4)
    payload = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "head_dim": head_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "attn_output_gate": attn_output_gate,
            "layer_types": layer_types,
        },
    }
    if include_vision_config:
        payload["vision_config"] = {
            "model_type": "qwen2_vl",
            "hidden_size": 1280,
            "image_size": 384,
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


def _write_qwen_snapshot(
    model_dir: Path,
    *,
    head_dim: int,
    include_vision_config: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    _write_qwen_config(
        model_dir / "config.json",
        head_dim=head_dim,
        num_hidden_layers=4,
        include_vision_config=include_vision_config,
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    if include_vision_config:
        (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    rng = np.random.default_rng(123)
    tensors = {
        "model.layers.3.self_attn.q_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.k_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.v_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.layers.3.self_attn.o_proj.weight": rng.normal(size=(48, head_dim)).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(head_dim, 32)).astype(np.float32),
        "model.embed_tokens.weight": rng.normal(size=(16, 32)).astype(np.float32),
    }
    if include_vision_config:
        tensors["visual.patch_embed.proj.weight"] = rng.normal(size=(32, 3, 14, 14)).astype(np.float32)
    weight_map = {
        "model.layers.3.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.3.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.3.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.3.self_attn.o_proj.weight": "model-00002-of-00002.safetensors",
        "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
    }
    if include_vision_config:
        weight_map["visual.patch_embed.proj.weight"] = "model-00002-of-00002.safetensors"
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
    _write_qwen_config(tmp_path / "config.json", head_dim=256, include_vision_config=True)
    result = _run_cli("plan", "qwen", "--model-dir", str(tmp_path))
    payload = json.loads(result.stdout)

    assert payload["family"] == "qwen"
    assert payload["supported_runtime"] is True
    assert payload["full_attention_indices"] == [3, 7]
    assert payload["has_vision_config"] is True
    assert payload["vision_model_type"] == "qwen2_vl"
    assert payload["attn_output_gate"] is False


def test_qwen_model_conversion_rewrites_only_supported_attention_layers(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors, _ = _write_qwen_snapshot(model_dir, head_dim=256, include_vision_config=True)

    output_dir = tmp_path / "converted"
    manifest = convert_qwen_model(model_dir=model_dir, output_dir=output_dir, model_name="qwen-demo")

    loaded_manifest = load_qwen_model_manifest(output_dir)
    assert manifest.model_name == "qwen-demo"
    assert loaded_manifest.converted_layer_indices == [3]
    assert loaded_manifest.passthrough_layer_indices == [0, 1, 2]
    assert loaded_manifest.has_vision_config is True
    assert loaded_manifest.source_vision_model_type == "qwen2_vl"
    assert loaded_manifest.layer_fusion_modes == {"3": "full_qkvo"}

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
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("visual.patch_embed.proj.weight")),
            tensors["visual.patch_embed.proj.weight"],
        )

    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "preprocessor_config.json").exists()


def test_cli_qwen_model_convert_and_inspect(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_snapshot(model_dir, head_dim=256, include_vision_config=True)

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
    assert inspect_payload["has_vision_config"] is True
    assert inspect_payload["layer_fusion_modes"] == {"3": "full_qkvo"}


def test_qwen_model_conversion_supports_delta_artifact_layout(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_snapshot(model_dir, head_dim=256, include_vision_config=True)

    output_dir = tmp_path / "delta"
    manifest = convert_qwen_model(
        model_dir=model_dir,
        output_dir=output_dir,
        model_name="qwen-delta",
        artifact_layout="delta_npz",
    )

    loaded_manifest = load_qwen_model_manifest(output_dir)
    assert manifest.artifact_layout == "delta_npz"
    assert loaded_manifest.delta_weights_file == "torque_qwen_delta_weights.npz"
    assert loaded_manifest.stored_asset_files
    assert not any(output_dir.glob("*.safetensors"))
    assert not (output_dir / "model.safetensors.index.json").exists()
    assert (output_dir / "torque_qwen_delta_weights.npz").exists()
    assert (output_dir / "tokenizer.json").exists()

    delta = np.load(output_dir / "torque_qwen_delta_weights.npz")
    assert set(delta.files) == {
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
    }


def test_qwen_model_conversion_supports_bfloat16_source_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_config(model_dir / "config.json", head_dim=256, num_hidden_layers=4)

    rng = np.random.default_rng(321)
    tensors = {
        "model.language_model.layers.3.self_attn.q_proj.weight": torch.tensor(
            rng.normal(size=(256, 32)).astype(np.float32),
            dtype=torch.bfloat16,
        ),
        "model.language_model.layers.3.self_attn.k_proj.weight": torch.tensor(
            rng.normal(size=(256, 32)).astype(np.float32),
            dtype=torch.bfloat16,
        ),
        "model.language_model.layers.3.self_attn.v_proj.weight": torch.tensor(
            rng.normal(size=(256, 32)).astype(np.float32),
            dtype=torch.bfloat16,
        ),
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.tensor(
            rng.normal(size=(48, 256)).astype(np.float32),
            dtype=torch.bfloat16,
        ),
        "model.language_model.embed_tokens.weight": torch.tensor(
            rng.normal(size=(16, 32)).astype(np.float32),
            dtype=torch.bfloat16,
        ),
    }
    save_torch_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "converted"
    manifest = convert_qwen_model(model_dir=model_dir, output_dir=output_dir, model_name="qwen-bf16")

    assert manifest.converted_layer_indices == [3]
    assert manifest.layer_fusion_modes == {"3": "full_qkvo"}

    with safe_open(str(output_dir / "model-00001-of-00001.safetensors"), framework="pt") as handle:
        q_proj = handle.get_tensor("model.language_model.layers.3.self_attn.q_proj.weight")
        embed = handle.get_tensor("model.language_model.embed_tokens.weight")
        assert q_proj.dtype == torch.bfloat16
        assert embed.dtype == torch.bfloat16


def test_qwen3_5_gated_attention_uses_vo_only_fusion_when_qk_norm_present(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_qwen_config(
        model_dir / "config.json",
        head_dim=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        attn_output_gate=True,
    )

    rng = np.random.default_rng(7)
    tensors = {
        "model.language_model.layers.3.self_attn.q_proj.weight": rng.normal(size=(4096, 32)).astype(np.float32),
        "model.language_model.layers.3.self_attn.k_proj.weight": rng.normal(size=(512, 32)).astype(np.float32),
        "model.language_model.layers.3.self_attn.v_proj.weight": rng.normal(size=(512, 32)).astype(np.float32),
        "model.language_model.layers.3.self_attn.o_proj.weight": rng.normal(size=(32, 2048)).astype(np.float32),
        "model.language_model.layers.3.self_attn.q_norm.weight": np.ones((256,), dtype=np.float32),
        "model.language_model.layers.3.self_attn.k_norm.weight": np.ones((256,), dtype=np.float32),
        "model.language_model.embed_tokens.weight": rng.normal(size=(16, 32)).astype(np.float32),
    }
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "converted"
    manifest = convert_qwen_model(model_dir=model_dir, output_dir=output_dir, model_name="qwen-gated")

    assert manifest.layer_fusion_modes == {"3": "vo_only_runtime_qk_rotation"}
    assert manifest.converted_tensor_names == {
        "3": [
            "model.language_model.layers.3.self_attn.v_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
        ],
    }

    with safe_open(str(output_dir / "model-00001-of-00001.safetensors"), framework="np") as handle:
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.language_model.layers.3.self_attn.q_proj.weight")),
            tensors["model.language_model.layers.3.self_attn.q_proj.weight"],
        )
        np.testing.assert_allclose(
            np.asarray(handle.get_tensor("model.language_model.layers.3.self_attn.k_proj.weight")),
            tensors["model.language_model.layers.3.self_attn.k_proj.weight"],
        )
        assert not np.allclose(
            np.asarray(handle.get_tensor("model.language_model.layers.3.self_attn.v_proj.weight")),
            tensors["model.language_model.layers.3.self_attn.v_proj.weight"],
        )
        assert not np.allclose(
            np.asarray(handle.get_tensor("model.language_model.layers.3.self_attn.o_proj.weight")),
            tensors["model.language_model.layers.3.self_attn.o_proj.weight"],
        )
