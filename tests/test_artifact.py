from pathlib import Path

import numpy as np

from torque_mlx.artifact import build_torque_artifact, load_torque_artifact
from torque_mlx.config import TorqueConfig
from torque_mlx.quantization import Codebook


def _uniform_codebook(bit_width: int) -> Codebook:
    levels = 1 << bit_width
    return Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, levels, dtype=np.float32),
        name="uniform",
    )


def test_artifact_roundtrip_and_cache_build(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    artifact = build_torque_artifact(
        model_name="tiny-attention",
        architecture="generic_decoder_attention",
        source_format="numpy_npz",
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=2, kv_heads=3),
        w_q=rng.normal(size=(64, 16)).astype(np.float32),
        w_k=rng.normal(size=(64, 16)).astype(np.float32),
        w_v=rng.normal(size=(64, 16)).astype(np.float32),
        w_o=rng.normal(size=(24, 64)).astype(np.float32),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )

    artifact_dir = tmp_path / "artifact"
    artifact.save(artifact_dir)
    loaded = load_torque_artifact(artifact_dir)

    assert loaded.manifest.model_name == "tiny-attention"
    assert loaded.runtime_config.variant_id == "b4-h64-hadamard-fused"
    np.testing.assert_allclose(loaded.fused_weights.w_q, artifact.fused_weights.w_q)
    np.testing.assert_allclose(loaded.fused_weights.w_o, artifact.fused_weights.w_o)

    cache = loaded.build_cache()
    assert cache.metadata.variant_id == "b4-h64-hadamard-fused"
    assert cache.metadata.kv_heads == 3


def test_artifact_summary_reports_shapes() -> None:
    rng = np.random.default_rng(1)
    artifact = build_torque_artifact(
        model_name="summary-test",
        architecture="generic_decoder_attention",
        source_format="numpy_npz",
        config=TorqueConfig(bit_width=3, head_dim=128),
        w_q=rng.normal(size=(128, 32)).astype(np.float32),
        w_k=rng.normal(size=(128, 32)).astype(np.float32),
        w_v=rng.normal(size=(128, 32)).astype(np.float32),
        w_o=rng.normal(size=(48, 128)).astype(np.float32),
    )

    summary = artifact.summary()
    assert summary["variant_id"] == "b3-h128-hadamard-fused"
    assert summary["weight_shapes"]["w_q"] == [128, 32]
    assert summary["extra_metadata"]["output_dim"] == 48
