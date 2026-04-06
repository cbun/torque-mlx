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
from torque_mlx.cache_mlx import TorqueKVCacheMLX
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
from torque_mlx.qwen_eval import (
    QwenTextBenchmarkComparison,
    QwenTextBenchmarkCase,
    QwenTextPerplexityResult,
    benchmark_qwen_text_models,
    evaluate_qwen_text_perplexity,
)
from torque_mlx.qwen_benchmark import (
    QwenDecodeBenchmarkProfile,
    QwenDecodeRuntimeBenchmarkResult,
    load_qwen_decode_benchmark_profile,
    run_qwen_decode_runtime_benchmark,
)

__all__ = [
    "FusedAttentionWeights",
    "QwenInspectionReport",
    "QwenModelArtifactManifest",
    "QwenDecodeBenchmarkProfile",
    "QwenDecodeRuntimeBenchmarkResult",
    "QwenTextBenchmarkCase",
    "QwenTextBenchmarkComparison",
    "QwenTextPerplexityResult",
    "TorqueArtifact",
    "TorqueArtifactManifest",
    "TorqueConfig",
    "TorqueKVCache",
    "TorqueKVCacheMLX",
    "build_torque_artifact",
    "convert_npz_checkpoint",
    "convert_qwen_attention_layer",
    "convert_qwen_model",
    "decode_packed_attention",
    "evaluate_artifact",
    "benchmark_qwen_text_models",
    "fuse_attention_weights",
    "inspect_qwen_hf_directory",
    "load_qwen_model_manifest",
    "load_qwen_decode_benchmark_profile",
    "load_torque_artifact",
    "metal_available",
    "evaluate_qwen_text_perplexity",
    "run_qwen_decode_runtime_benchmark",
    "run_synthetic_decode_benchmark",
]
