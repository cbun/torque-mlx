from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from torque_mlx.artifact import TorqueArtifact, build_torque_artifact
from torque_mlx.config import TorqueConfig
from torque_mlx.hf_safetensors import (
    build_weight_map,
    copy_non_weight_assets,
    find_tensor_key_by_suffix,
    load_tensor,
    materialize_merged_snapshot,
)
from torque_mlx.quantization import Codebook, build_gaussian_codebook
from torque_mlx.rotation import RotationMode, RotationSpec


SUPPORTED_QWEN_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
SUPPORTED_QWEN_ATTENTION_LAYER_TYPE = "full_attention"
QWEN_MODEL_MANIFEST_FILE = "torque_qwen_manifest.json"
QWEN_DELTA_WEIGHTS_FILE = "torque_qwen_delta_weights.npz"
SUPPORTED_QWEN_ARTIFACT_LAYOUTS = {"merged_snapshot", "delta_npz"}


@dataclass(frozen=True, slots=True)
class QwenInspectionReport:
    model_dir: str
    architectures: list[str]
    model_type: str
    text_model_type: str | None
    vision_model_type: str | None
    has_vision_config: bool
    hidden_size: int
    head_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_output_gate: bool
    layer_types: list[str]
    full_attention_indices: list[int]
    linear_attention_indices: list[int]
    supported_runtime: bool
    blocking_issues: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "qwen",
            "model_dir": self.model_dir,
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "text_model_type": self.text_model_type,
            "vision_model_type": self.vision_model_type,
            "has_vision_config": self.has_vision_config,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "attn_output_gate": self.attn_output_gate,
            "layer_types": list(self.layer_types),
            "full_attention_indices": list(self.full_attention_indices),
            "linear_attention_indices": list(self.linear_attention_indices),
            "supported_runtime": self.supported_runtime,
            "blocking_issues": list(self.blocking_issues),
            "notes": list(self.notes),
            "recommended_workflow": [
                "Inspect the local Hugging Face snapshot and record the full_attention layer indices.",
                "Extract q/k/v/o weights only for supported full_attention layers.",
                "Convert those layers with torque-mlx and publish the resulting validated artifact.",
            ],
        }


@dataclass(frozen=True, slots=True)
class QwenModelArtifactManifest:
    model_name: str
    source_model_dir: str
    runtime_config: TorqueConfig
    full_attention_indices: list[int]
    converted_layer_indices: list[int]
    passthrough_layer_indices: list[int]
    converted_tensor_names: dict[str, list[str]]
    layer_fusion_modes: dict[str, str]
    key_codebook: Codebook
    value_codebook: Codebook
    source_model_type: str
    source_text_model_type: str | None
    source_vision_model_type: str | None
    has_vision_config: bool
    source_architectures: list[str]
    artifact_layout: str = "merged_snapshot"
    delta_weights_file: str | None = None
    stored_asset_files: list[str] = None
    format_name: str = "torque-qwen-model"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.artifact_layout not in SUPPORTED_QWEN_ARTIFACT_LAYOUTS:
            raise ValueError(
                "artifact_layout must be one of "
                + ", ".join(sorted(SUPPORTED_QWEN_ARTIFACT_LAYOUTS)),
            )
        if self.artifact_layout == "delta_npz" and not self.delta_weights_file:
            raise ValueError("delta_weights_file is required for delta_npz artifacts")
        if self.artifact_layout != "delta_npz" and self.delta_weights_file is not None:
            raise ValueError("delta_weights_file is only valid for delta_npz artifacts")
        if self.stored_asset_files is None:
            object.__setattr__(self, "stored_asset_files", [])

    def to_dict(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "model_name": self.model_name,
            "source_model_dir": self.source_model_dir,
            "source_model_type": self.source_model_type,
            "source_text_model_type": self.source_text_model_type,
            "source_vision_model_type": self.source_vision_model_type,
            "has_vision_config": self.has_vision_config,
            "source_architectures": list(self.source_architectures),
            "artifact_layout": self.artifact_layout,
            "delta_weights_file": self.delta_weights_file,
            "stored_asset_files": list(self.stored_asset_files),
            "runtime_config": {
                "bit_width": self.runtime_config.bit_width,
                "head_dim": self.runtime_config.head_dim,
                "num_layers": self.runtime_config.num_layers,
                "kv_heads": self.runtime_config.kv_heads,
                "fused_weights": self.runtime_config.fused_weights,
                "rotation_mode": self.runtime_config.rotation_mode.value,
                "rotation_seed": self.runtime_config.rotation_seed,
            },
            "full_attention_indices": list(self.full_attention_indices),
            "converted_layer_indices": list(self.converted_layer_indices),
            "passthrough_layer_indices": list(self.passthrough_layer_indices),
            "converted_tensor_names": {str(key): list(value) for key, value in self.converted_tensor_names.items()},
            "layer_fusion_modes": dict(self.layer_fusion_modes),
            "key_codebook": self.key_codebook.to_dict(),
            "value_codebook": self.value_codebook.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QwenModelArtifactManifest":
        runtime_payload = dict(payload["runtime_config"])
        return cls(
            model_name=str(payload["model_name"]),
            source_model_dir=str(payload["source_model_dir"]),
            runtime_config=TorqueConfig(
                bit_width=int(runtime_payload["bit_width"]),
                head_dim=int(runtime_payload["head_dim"]),
                num_layers=int(runtime_payload["num_layers"]),
                kv_heads=int(runtime_payload["kv_heads"]),
                fused_weights=bool(runtime_payload["fused_weights"]),
                rotation_mode=RotationMode(str(runtime_payload["rotation_mode"])),
                rotation_seed=int(runtime_payload["rotation_seed"]),
            ),
            full_attention_indices=[int(item) for item in payload["full_attention_indices"]],
            converted_layer_indices=[int(item) for item in payload["converted_layer_indices"]],
            passthrough_layer_indices=[int(item) for item in payload["passthrough_layer_indices"]],
            converted_tensor_names={
                str(key): [str(item) for item in value]
                for key, value in dict(payload["converted_tensor_names"]).items()
            },
            layer_fusion_modes={
                str(key): str(value)
                for key, value in dict(payload.get("layer_fusion_modes", {})).items()
            },
            key_codebook=Codebook.from_dict(dict(payload["key_codebook"])),
            value_codebook=Codebook.from_dict(dict(payload["value_codebook"])),
            source_model_type=str(payload["source_model_type"]),
            source_text_model_type=(
                str(payload["source_text_model_type"])
                if payload.get("source_text_model_type") is not None
                else None
            ),
            source_vision_model_type=(
                str(payload["source_vision_model_type"])
                if payload.get("source_vision_model_type") is not None
                else None
            ),
            has_vision_config=bool(payload.get("has_vision_config", False)),
            source_architectures=[str(item) for item in payload["source_architectures"]],
            artifact_layout=str(payload.get("artifact_layout", "merged_snapshot")),
            delta_weights_file=(
                str(payload["delta_weights_file"])
                if payload.get("delta_weights_file") is not None
                else None
            ),
            stored_asset_files=[str(item) for item in payload.get("stored_asset_files", [])],
            format_name=str(payload.get("format_name", "torque-qwen-model")),
            format_version=int(payload.get("format_version", 1)),
        )

    def summary(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "model_name": self.model_name,
            "source_model_dir": self.source_model_dir,
            "variant_id": self.runtime_config.variant_id,
            "source_model_type": self.source_model_type,
            "source_text_model_type": self.source_text_model_type,
            "source_vision_model_type": self.source_vision_model_type,
            "has_vision_config": self.has_vision_config,
            "artifact_layout": self.artifact_layout,
            "delta_weights_file": self.delta_weights_file,
            "stored_asset_file_count": len(self.stored_asset_files),
            "converted_layer_indices": list(self.converted_layer_indices),
            "passthrough_layer_indices": list(self.passthrough_layer_indices),
            "full_attention_indices": list(self.full_attention_indices),
            "converted_tensor_count": int(sum(len(items) for items in self.converted_tensor_names.values())),
            "layer_fusion_modes": dict(self.layer_fusion_modes),
            "key_codebook": {
                "name": self.key_codebook.name,
                "bit_width": self.key_codebook.bit_width,
            },
            "value_codebook": {
                "name": self.value_codebook.name,
                "bit_width": self.value_codebook.bit_width,
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _materialize_delta_artifact(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    overrides: Mapping[str, np.ndarray],
    force: bool,
) -> tuple[Path, list[str]]:
    source_root = Path(source_dir)
    target_root = Path(output_dir)

    if target_root.exists() and any(target_root.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory {target_root}; pass force=True to replace it",
        )
    if target_root.exists() and force:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    stored_asset_files = copy_non_weight_assets(source_dir=source_root, output_dir=target_root)
    np.savez_compressed(
        target_root / QWEN_DELTA_WEIGHTS_FILE,
        **{name: np.asarray(value, dtype=np.float32) for name, value in overrides.items()},
    )
    return target_root, stored_asset_files


def inspect_qwen_hf_directory(model_dir: str | Path) -> QwenInspectionReport:
    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {root}")

    payload = _load_json(config_path)
    text_config = payload.get("text_config", payload)
    if not isinstance(text_config, dict):
        raise ValueError("Expected text_config to be a JSON object")
    vision_config = payload.get("vision_config")
    if vision_config is not None and not isinstance(vision_config, dict):
        raise ValueError("Expected vision_config to be a JSON object when present")

    architectures = [str(item) for item in payload.get("architectures", [])]
    model_type = str(payload.get("model_type", ""))
    text_model_type = (
        str(text_config.get("model_type"))
        if text_config.get("model_type") is not None
        else None
    )
    vision_model_type = (
        str(vision_config.get("model_type"))
        if isinstance(vision_config, dict) and vision_config.get("model_type") is not None
        else None
    )
    layer_types = [str(item) for item in text_config.get("layer_types", [])]
    full_attention_indices = [
        index for index, layer_type in enumerate(layer_types) if layer_type == SUPPORTED_QWEN_ATTENTION_LAYER_TYPE
    ]
    linear_attention_indices = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    ]
    head_dim = int(text_config["head_dim"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_attention_heads = int(text_config["num_attention_heads"])
    hidden_size = int(text_config.get("hidden_size", head_dim * num_attention_heads))
    num_key_value_heads = int(text_config["num_key_value_heads"])
    attn_output_gate = bool(text_config.get("attn_output_gate", False))

    blocking_issues: list[str] = []
    if model_type not in SUPPORTED_QWEN_MODEL_TYPES:
        blocking_issues.append(f"Unsupported top-level model_type: {model_type}")
    if text_model_type is not None and text_model_type not in SUPPORTED_QWEN_MODEL_TYPES:
        blocking_issues.append(f"Unsupported text_config.model_type: {text_model_type}")
    if head_dim not in {64, 128, 256}:
        blocking_issues.append(
            f"Unsupported head_dim {head_dim}; torque-mlx currently supports only 64, 128, and 256",
        )
    if len(layer_types) != num_hidden_layers:
        blocking_issues.append(
            f"layer_types length {len(layer_types)} does not match num_hidden_layers {num_hidden_layers}",
        )
    if not full_attention_indices:
        blocking_issues.append("No full_attention layers found; there is nothing to convert to TorqueKVCache")

    notes = [
        "Only full_attention layers are candidates for torque-mlx conversion; linear attention layers are copied through.",
        "Curated family support is explicit. Unsupported Qwen variants should fail during planning rather than convert speculatively.",
    ]
    if vision_config is not None:
        notes.append(
            "Multimodal Qwen snapshots are supported at the conversion layer: text full_attention weights are rewritten, while vision and other non-converted components are copied through unchanged.",
        )
    if full_attention_indices:
        notes.append(
            f"Detected {len(full_attention_indices)} full_attention layers out of {num_hidden_layers} total layers.",
        )

    return QwenInspectionReport(
        model_dir=str(root.resolve()),
        architectures=architectures,
        model_type=model_type,
        text_model_type=text_model_type,
        vision_model_type=vision_model_type,
        has_vision_config=vision_config is not None,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attn_output_gate=attn_output_gate,
        layer_types=layer_types,
        full_attention_indices=full_attention_indices,
        linear_attention_indices=linear_attention_indices,
        supported_runtime=not blocking_issues,
        blocking_issues=blocking_issues,
        notes=notes,
    )


def convert_qwen_attention_layer(
    *,
    model_dir: str | Path,
    layer_idx: int,
    input_weights: str | Path,
    output_dir: str | Path,
    bit_width: int = 4,
    rotation_seed: int = 0,
    force: bool = False,
) -> TorqueArtifact:
    report = inspect_qwen_hf_directory(model_dir)
    if not report.supported_runtime:
        raise ValueError(
            "Qwen model is not currently supported by torque-mlx: "
            + "; ".join(report.blocking_issues),
        )
    if layer_idx not in report.full_attention_indices:
        raise ValueError(
            f"Layer {layer_idx} is not a full_attention layer; convertible layers are {report.full_attention_indices}",
        )

    checkpoint = np.load(input_weights)
    for required in ("w_q", "w_k", "w_v", "w_o"):
        if required not in checkpoint:
            raise ValueError("Expected NPZ checkpoint to contain w_q, w_k, w_v, and w_o arrays")

    artifact = build_torque_artifact(
        model_name=f"{Path(report.model_dir).name}-layer{layer_idx}",
        architecture="qwen_full_attention_layer",
        source_format="qwen_numpy_npz",
        config=TorqueConfig(
            bit_width=bit_width,
            head_dim=report.head_dim,
            num_layers=1,
            kv_heads=report.num_key_value_heads,
            fused_weights=True,
            rotation_mode=RotationMode.HADAMARD,
            rotation_seed=rotation_seed,
        ),
        w_q=checkpoint["w_q"],
        w_k=checkpoint["w_k"],
        w_v=checkpoint["w_v"],
        w_o=checkpoint["w_o"],
        extra_metadata={
            "family": "qwen",
            "source_model_dir": report.model_dir,
            "source_model_type": report.model_type,
            "source_text_model_type": report.text_model_type,
            "source_num_hidden_layers": report.num_hidden_layers,
            "source_full_attention_indices": list(report.full_attention_indices),
            "converted_layer_idx": int(layer_idx),
            "layer_type": SUPPORTED_QWEN_ATTENTION_LAYER_TYPE,
        },
    )
    artifact.save(output_dir, force=force)
    return artifact


def load_qwen_model_manifest(directory: str | Path) -> QwenModelArtifactManifest:
    target = Path(directory) / QWEN_MODEL_MANIFEST_FILE
    if not target.exists():
        raise FileNotFoundError(f"Qwen model manifest not found: {target}")
    return QwenModelArtifactManifest.from_dict(_load_json(target))


def _layer_projection_suffixes(layer_idx: int) -> dict[str, str]:
    return {
        "w_q": f".layers.{layer_idx}.self_attn.q_proj.weight",
        "w_k": f".layers.{layer_idx}.self_attn.k_proj.weight",
        "w_v": f".layers.{layer_idx}.self_attn.v_proj.weight",
        "w_o": f".layers.{layer_idx}.self_attn.o_proj.weight",
    }


def _optional_tensor_key_by_suffix(weight_map: Mapping[str, str], suffix: str) -> str | None:
    matches = [key for key in weight_map if key.endswith(suffix)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected at most one tensor ending with {suffix}, found {matches}")
    return matches[0]


def _rotate_stacked_output_projection(
    weight: np.ndarray,
    *,
    head_dim: int,
    num_blocks: int,
    rotation: RotationSpec,
) -> np.ndarray:
    array = np.asarray(weight, dtype=np.float32)
    if array.shape[0] != num_blocks * head_dim:
        raise ValueError(
            f"Expected projection output dim {num_blocks * head_dim}, got {array.shape[0]}",
        )
    reshaped = array.reshape(num_blocks, head_dim, array.shape[1])
    rotated = np.matmul(rotation.matrix()[None, :, :], reshaped)
    return rotated.reshape(array.shape)


def _inverse_rotate_stacked_input_projection(
    weight: np.ndarray,
    *,
    head_dim: int,
    num_blocks: int,
    rotation: RotationSpec,
) -> np.ndarray:
    array = np.asarray(weight, dtype=np.float32)
    if array.shape[1] != num_blocks * head_dim:
        raise ValueError(
            f"Expected projection input dim {num_blocks * head_dim}, got {array.shape[1]}",
        )
    reshaped = array.reshape(array.shape[0], num_blocks, head_dim)
    rotated = np.matmul(reshaped, rotation.matrix().T)
    return rotated.reshape(array.shape)


def _fuse_qwen_full_attention_tensors(
    *,
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    o_proj: np.ndarray,
    rotation: RotationSpec,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    attn_output_gate: bool,
    has_q_norm: bool,
    has_k_norm: bool,
) -> tuple[dict[str, np.ndarray], str]:
    expected_q_rows = num_attention_heads * head_dim
    expected_q_proj_rows = expected_q_rows * (2 if attn_output_gate else 1)
    expected_kv_rows = num_key_value_heads * head_dim

    if q_proj.shape[0] != expected_q_proj_rows:
        raise ValueError(
            f"Expected q_proj output dim {expected_q_proj_rows}, got {q_proj.shape[0]}",
        )
    if k_proj.shape[0] != expected_kv_rows:
        raise ValueError(
            f"Expected k_proj output dim {expected_kv_rows}, got {k_proj.shape[0]}",
        )
    if v_proj.shape[0] != expected_kv_rows:
        raise ValueError(
            f"Expected v_proj output dim {expected_kv_rows}, got {v_proj.shape[0]}",
        )
    if o_proj.shape[1] != expected_q_rows:
        raise ValueError(
            f"Expected o_proj input dim {expected_q_rows}, got {o_proj.shape[1]}",
        )

    fused_v = _rotate_stacked_output_projection(
        v_proj,
        head_dim=head_dim,
        num_blocks=num_key_value_heads,
        rotation=rotation,
    )
    fused_o = _inverse_rotate_stacked_input_projection(
        o_proj,
        head_dim=head_dim,
        num_blocks=num_attention_heads,
        rotation=rotation,
    )

    if has_q_norm or has_k_norm:
        return (
            {
                "w_v": fused_v,
                "w_o": fused_o,
            },
            "vo_only_runtime_qk_rotation",
        )

    q_rows = expected_q_rows
    q_base = np.asarray(q_proj[:q_rows], dtype=np.float32)
    q_gate = np.asarray(q_proj[q_rows:], dtype=np.float32) if attn_output_gate else None
    fused_q = _rotate_stacked_output_projection(
        q_base,
        head_dim=head_dim,
        num_blocks=num_attention_heads,
        rotation=rotation,
    )
    fused_k = _rotate_stacked_output_projection(
        k_proj,
        head_dim=head_dim,
        num_blocks=num_key_value_heads,
        rotation=rotation,
    )

    if q_gate is not None:
        fused_q = np.concatenate([fused_q, q_gate], axis=0)

    return (
        {
            "w_q": fused_q,
            "w_k": fused_k,
            "w_v": fused_v,
            "w_o": fused_o,
        },
        "full_qkvo",
    )


def convert_qwen_model(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    bit_width: int = 4,
    rotation_seed: int = 0,
    model_name: str | None = None,
    artifact_layout: str = "merged_snapshot",
    force: bool = False,
) -> QwenModelArtifactManifest:
    report = inspect_qwen_hf_directory(model_dir)
    if not report.supported_runtime:
        raise ValueError(
            "Qwen model is not currently supported by torque-mlx: "
            + "; ".join(report.blocking_issues),
        )
    if artifact_layout not in SUPPORTED_QWEN_ARTIFACT_LAYOUTS:
        raise ValueError(
            "artifact_layout must be one of "
            + ", ".join(sorted(SUPPORTED_QWEN_ARTIFACT_LAYOUTS)),
        )

    runtime_config = TorqueConfig(
        bit_width=bit_width,
        head_dim=report.head_dim,
        num_layers=report.num_hidden_layers,
        kv_heads=report.num_key_value_heads,
        fused_weights=True,
        rotation_mode=RotationMode.HADAMARD,
        rotation_seed=rotation_seed,
    )
    rotation = RotationSpec.from_seed(head_dim=report.head_dim, seed=rotation_seed)
    key_codebook = build_gaussian_codebook(bit_width, seed=rotation_seed)
    value_codebook = build_gaussian_codebook(bit_width, seed=rotation_seed + 1)

    weight_map = build_weight_map(model_dir)
    overrides: dict[str, np.ndarray] = {}
    converted_tensor_names: dict[str, list[str]] = {}
    layer_fusion_modes: dict[str, str] = {}

    for layer_idx in report.full_attention_indices:
        suffixes = _layer_projection_suffixes(layer_idx)
        keys = {
            name: find_tensor_key_by_suffix(weight_map, suffix)
            for name, suffix in suffixes.items()
        }
        weights = {
            name: load_tensor(model_dir, tensor_name, weight_map=weight_map)
            for name, tensor_name in keys.items()
        }
        q_norm_name = _optional_tensor_key_by_suffix(
            weight_map,
            f".layers.{layer_idx}.self_attn.q_norm.weight",
        )
        k_norm_name = _optional_tensor_key_by_suffix(
            weight_map,
            f".layers.{layer_idx}.self_attn.k_norm.weight",
        )
        fused_weights, fusion_mode = _fuse_qwen_full_attention_tensors(
            q_proj=weights["w_q"],
            k_proj=weights["w_k"],
            v_proj=weights["w_v"],
            o_proj=weights["w_o"],
            rotation=rotation,
            head_dim=report.head_dim,
            num_attention_heads=report.num_attention_heads,
            num_key_value_heads=report.num_key_value_heads,
            attn_output_gate=report.attn_output_gate,
            has_q_norm=q_norm_name is not None,
            has_k_norm=k_norm_name is not None,
        )
        if "w_q" in fused_weights:
            overrides[keys["w_q"]] = fused_weights["w_q"]
        if "w_k" in fused_weights:
            overrides[keys["w_k"]] = fused_weights["w_k"]
        if "w_v" in fused_weights:
            overrides[keys["w_v"]] = fused_weights["w_v"]
        if "w_o" in fused_weights:
            overrides[keys["w_o"]] = fused_weights["w_o"]
        converted = []
        for name in ("w_q", "w_k", "w_v", "w_o"):
            if name in fused_weights:
                converted.append(keys[name])
        converted_tensor_names[str(layer_idx)] = converted
        layer_fusion_modes[str(layer_idx)] = fusion_mode

    stored_asset_files: list[str] = []
    if artifact_layout == "merged_snapshot":
        materialize_merged_snapshot(
            source_dir=model_dir,
            output_dir=output_dir,
            overrides=overrides,
            weight_map=weight_map,
            force=force,
        )
    else:
        _, stored_asset_files = _materialize_delta_artifact(
            source_dir=model_dir,
            output_dir=output_dir,
            overrides=overrides,
            force=force,
        )

    manifest = QwenModelArtifactManifest(
        model_name=model_name or Path(report.model_dir).name,
        source_model_dir=report.model_dir,
        runtime_config=runtime_config,
        full_attention_indices=list(report.full_attention_indices),
        converted_layer_indices=list(report.full_attention_indices),
        passthrough_layer_indices=[
            layer_idx for layer_idx in range(report.num_hidden_layers) if layer_idx not in report.full_attention_indices
        ],
        converted_tensor_names=converted_tensor_names,
        layer_fusion_modes=layer_fusion_modes,
        key_codebook=key_codebook,
        value_codebook=value_codebook,
        source_model_type=report.model_type,
        source_text_model_type=report.text_model_type,
        source_vision_model_type=report.vision_model_type,
        has_vision_config=report.has_vision_config,
        source_architectures=list(report.architectures),
        artifact_layout=artifact_layout,
        delta_weights_file=QWEN_DELTA_WEIGHTS_FILE if artifact_layout == "delta_npz" else None,
        stored_asset_files=stored_asset_files,
    )
    _save_json(Path(output_dir) / QWEN_MODEL_MANIFEST_FILE, manifest.to_dict())
    return manifest
