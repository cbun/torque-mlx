from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.conversion import FusedAttentionWeights, fuse_attention_weights
from torque_mlx.quantization import Codebook, build_gaussian_codebook
from torque_mlx.rotation import RotationMode, RotationSpec


ARTIFACT_FORMAT = "torque-mlx-artifact"
ARTIFACT_VERSION = 1
DEFAULT_WEIGHTS_FILE = "weights.npz"
DEFAULT_MANIFEST_FILE = "manifest.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_fields(payload: Mapping[str, object], fields: tuple[str, ...]) -> None:
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class TorqueArtifactManifest:
    model_name: str
    architecture: str
    source_format: str
    runtime_config: TorqueConfig
    key_codebook: Codebook
    value_codebook: Codebook
    weights_file: str = DEFAULT_WEIGHTS_FILE
    format_name: str = ARTIFACT_FORMAT
    format_version: int = ARTIFACT_VERSION
    weight_names: dict[str, str] = field(
        default_factory=lambda: {
            "w_q": "w_q",
            "w_k": "w_k",
            "w_v": "w_v",
            "w_o": "w_o",
        },
    )
    extra_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.runtime_config.validate()
        if not self.runtime_config.fused_weights:
            raise ValueError("Artifact runtime_config must enable fused_weights")
        if self.format_name != ARTIFACT_FORMAT:
            raise ValueError(f"Unsupported artifact format: {self.format_name}")
        if self.format_version != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version: {self.format_version}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TorqueArtifactManifest":
        _require_fields(
            payload,
            (
                "model_name",
                "architecture",
                "source_format",
                "runtime_config",
                "key_codebook",
                "value_codebook",
            ),
        )
        runtime_payload = payload["runtime_config"]
        if not isinstance(runtime_payload, Mapping):
            raise ValueError("runtime_config must be an object")
        config = TorqueConfig(
            bit_width=int(runtime_payload["bit_width"]),
            head_dim=int(runtime_payload["head_dim"]),
            num_layers=int(runtime_payload["num_layers"]),
            kv_heads=int(runtime_payload["kv_heads"]),
            fused_weights=bool(runtime_payload["fused_weights"]),
            rotation_mode=RotationMode(str(runtime_payload["rotation_mode"])),
            rotation_seed=int(runtime_payload["rotation_seed"]),
        )
        weight_names = dict(payload.get("weight_names", {})) or {
            "w_q": "w_q",
            "w_k": "w_k",
            "w_v": "w_v",
            "w_o": "w_o",
        }
        extra_metadata = dict(payload.get("extra_metadata", {}))
        return cls(
            model_name=str(payload["model_name"]),
            architecture=str(payload["architecture"]),
            source_format=str(payload["source_format"]),
            runtime_config=config,
            key_codebook=Codebook.from_dict(dict(payload["key_codebook"])),
            value_codebook=Codebook.from_dict(dict(payload["value_codebook"])),
            weights_file=str(payload.get("weights_file", DEFAULT_WEIGHTS_FILE)),
            format_name=str(payload.get("format_name", ARTIFACT_FORMAT)),
            format_version=int(payload.get("format_version", ARTIFACT_VERSION)),
            weight_names={str(key): str(value) for key, value in weight_names.items()},
            extra_metadata={str(key): _json_ready(value) for key, value in extra_metadata.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "source_format": self.source_format,
            "variant_id": self.runtime_config.variant_id,
            "runtime_config": {
                "bit_width": self.runtime_config.bit_width,
                "head_dim": self.runtime_config.head_dim,
                "num_layers": self.runtime_config.num_layers,
                "kv_heads": self.runtime_config.kv_heads,
                "fused_weights": self.runtime_config.fused_weights,
                "rotation_mode": self.runtime_config.rotation_mode.value,
                "rotation_seed": self.runtime_config.rotation_seed,
            },
            "weights_file": self.weights_file,
            "weight_names": dict(self.weight_names),
            "key_codebook": self.key_codebook.to_dict(),
            "value_codebook": self.value_codebook.to_dict(),
            "extra_metadata": _json_ready(self.extra_metadata),
        }


@dataclass(frozen=True, slots=True)
class TorqueArtifact:
    manifest: TorqueArtifactManifest
    fused_weights: FusedAttentionWeights

    @property
    def runtime_config(self) -> TorqueConfig:
        return self.manifest.runtime_config

    @property
    def key_codebook(self) -> Codebook:
        return self.manifest.key_codebook

    @property
    def value_codebook(self) -> Codebook:
        return self.manifest.value_codebook

    def build_cache(self) -> TorqueKVCache:
        return TorqueKVCache(
            config=self.runtime_config,
            key_codebook=self.key_codebook,
            value_codebook=self.value_codebook,
        )

    def save(self, directory: str | Path, *, force: bool = False) -> Path:
        target = Path(directory)
        manifest_path = target / DEFAULT_MANIFEST_FILE
        weights_path = target / self.manifest.weights_file
        if target.exists() and any(target.iterdir()) and not force:
            raise FileExistsError(
                f"Refusing to overwrite non-empty artifact directory {target}; pass force=True to replace it",
            )
        target.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        np.savez_compressed(
            weights_path,
            w_q=self.fused_weights.w_q,
            w_k=self.fused_weights.w_k,
            w_v=self.fused_weights.w_v,
            w_o=self.fused_weights.w_o,
        )
        return target

    def summary(self) -> dict[str, object]:
        return {
            "format_name": self.manifest.format_name,
            "format_version": self.manifest.format_version,
            "model_name": self.manifest.model_name,
            "architecture": self.manifest.architecture,
            "source_format": self.manifest.source_format,
            "variant_id": self.runtime_config.variant_id,
            "weights_file": self.manifest.weights_file,
            "runtime_config": {
                "bit_width": self.runtime_config.bit_width,
                "head_dim": self.runtime_config.head_dim,
                "num_layers": self.runtime_config.num_layers,
                "kv_heads": self.runtime_config.kv_heads,
                "fused_weights": self.runtime_config.fused_weights,
                "rotation_mode": self.runtime_config.rotation_mode.value,
                "rotation_seed": self.runtime_config.rotation_seed,
            },
            "weight_shapes": {
                "w_q": list(self.fused_weights.w_q.shape),
                "w_k": list(self.fused_weights.w_k.shape),
                "w_v": list(self.fused_weights.w_v.shape),
                "w_o": list(self.fused_weights.w_o.shape),
            },
            "key_codebook": {
                "name": self.key_codebook.name,
                "bit_width": self.key_codebook.bit_width,
                "centroid_count": int(self.key_codebook.centroids.size),
            },
            "value_codebook": {
                "name": self.value_codebook.name,
                "bit_width": self.value_codebook.bit_width,
                "centroid_count": int(self.value_codebook.centroids.size),
            },
            "extra_metadata": _json_ready(self.manifest.extra_metadata),
        }

    @classmethod
    def load(cls, directory: str | Path) -> "TorqueArtifact":
        target = Path(directory)
        manifest_path = target / DEFAULT_MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        manifest = TorqueArtifactManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        weights_path = target / manifest.weights_file
        if not weights_path.exists():
            raise FileNotFoundError(f"Artifact weights not found: {weights_path}")
        data = np.load(weights_path)
        return cls(
            manifest=manifest,
            fused_weights=FusedAttentionWeights(
                w_q=np.asarray(data[manifest.weight_names["w_q"]], dtype=np.float32),
                w_k=np.asarray(data[manifest.weight_names["w_k"]], dtype=np.float32),
                w_v=np.asarray(data[manifest.weight_names["w_v"]], dtype=np.float32),
                w_o=np.asarray(data[manifest.weight_names["w_o"]], dtype=np.float32),
                rotation_seed=manifest.runtime_config.rotation_seed,
            ),
        )


def build_torque_artifact(
    *,
    model_name: str,
    architecture: str,
    source_format: str,
    config: TorqueConfig,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    rotation: RotationSpec | None = None,
    key_codebook: Codebook | None = None,
    value_codebook: Codebook | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> TorqueArtifact:
    runtime_config = replace(config, fused_weights=True)
    rotation = rotation or RotationSpec.from_seed(
        head_dim=runtime_config.head_dim,
        seed=runtime_config.rotation_seed,
    )
    key_codebook = key_codebook or build_gaussian_codebook(
        runtime_config.bit_width,
        seed=runtime_config.rotation_seed,
    )
    value_codebook = value_codebook or build_gaussian_codebook(
        runtime_config.bit_width,
        seed=runtime_config.rotation_seed + 1,
    )
    fused = fuse_attention_weights(
        w_q=np.asarray(w_q, dtype=np.float32),
        w_k=np.asarray(w_k, dtype=np.float32),
        w_v=np.asarray(w_v, dtype=np.float32),
        w_o=np.asarray(w_o, dtype=np.float32),
        rotation=rotation,
    )
    metadata = {
        "input_dim": int(np.asarray(w_q).shape[1]),
        "output_dim": int(np.asarray(w_o).shape[0]),
    }
    if extra_metadata:
        metadata.update({str(key): _json_ready(value) for key, value in extra_metadata.items()})
    manifest = TorqueArtifactManifest(
        model_name=model_name,
        architecture=architecture,
        source_format=source_format,
        runtime_config=runtime_config,
        key_codebook=key_codebook,
        value_codebook=value_codebook,
        extra_metadata=metadata,
    )
    return TorqueArtifact(manifest=manifest, fused_weights=fused)


def convert_npz_checkpoint(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str | None,
    architecture: str,
    source_format: str,
    bit_width: int,
    num_layers: int,
    kv_heads: int,
    rotation_seed: int,
    head_dim: int | None = None,
    key_codebook: Codebook | None = None,
    value_codebook: Codebook | None = None,
    force: bool = False,
) -> TorqueArtifact:
    checkpoint = np.load(input_path)
    for required in ("w_q", "w_k", "w_v", "w_o"):
        if required not in checkpoint:
            raise ValueError(
                "Expected NPZ checkpoint to contain w_q, w_k, w_v, and w_o arrays",
            )
    inferred_head_dim = int(np.asarray(checkpoint["w_q"]).shape[0])
    config = TorqueConfig(
        bit_width=bit_width,
        head_dim=inferred_head_dim if head_dim is None else head_dim,
        num_layers=num_layers,
        kv_heads=kv_heads,
        fused_weights=True,
        rotation_seed=rotation_seed,
    )
    artifact = build_torque_artifact(
        model_name=model_name or Path(input_path).stem,
        architecture=architecture,
        source_format=source_format,
        config=config,
        w_q=checkpoint["w_q"],
        w_k=checkpoint["w_k"],
        w_v=checkpoint["w_v"],
        w_o=checkpoint["w_o"],
        key_codebook=key_codebook,
        value_codebook=value_codebook,
    )
    artifact.save(output_dir, force=force)
    return artifact


def load_torque_artifact(directory: str | Path) -> TorqueArtifact:
    return TorqueArtifact.load(directory)
