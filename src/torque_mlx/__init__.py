"""torque-mlx package."""

from torque_mlx.artifact import (
    TorqueArtifact,
    TorqueArtifactManifest,
    build_torque_artifact,
    convert_npz_checkpoint,
    load_torque_artifact,
)
from torque_mlx.benchmarking import evaluate_artifact, run_synthetic_decode_benchmark
from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.conversion import FusedAttentionWeights, fuse_attention_weights
from torque_mlx.families.qwen import (
    QwenInspectionReport,
    QwenModelArtifactManifest,
    convert_qwen_attention_layer,
    convert_qwen_model,
    inspect_qwen_hf_directory,
    load_qwen_model_manifest,
)
from torque_mlx.mlx_ops import decode_packed_attention, metal_available

__all__ = [
    "FusedAttentionWeights",
    "QwenInspectionReport",
    "QwenModelArtifactManifest",
    "TorqueArtifact",
    "TorqueArtifactManifest",
    "TorqueConfig",
    "TorqueKVCache",
    "build_torque_artifact",
    "convert_npz_checkpoint",
    "convert_qwen_attention_layer",
    "convert_qwen_model",
    "decode_packed_attention",
    "evaluate_artifact",
    "fuse_attention_weights",
    "inspect_qwen_hf_directory",
    "load_qwen_model_manifest",
    "load_torque_artifact",
    "metal_available",
    "run_synthetic_decode_benchmark",
]
