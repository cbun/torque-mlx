import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from torque_mlx.mlx_ops import (
    decode_packed_attention,
    decode_packed_attention_split,
    metal_available,
)
from torque_mlx.quantization import Codebook, pack_indices
from torque_mlx.reference import streaming_attention_decode


@pytest.mark.skipif(not metal_available(), reason="Metal toolchain unavailable")
@pytest.mark.parametrize("bit_width", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_decode_packed_attention_matches_reference(bit_width: int, head_dim: int) -> None:
    codebook = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )
    rng = np.random.default_rng(0)
    seq_len = 7
    query = rng.uniform(-1.0, 1.0, size=(head_dim,)).astype(np.float32)
    keys = rng.uniform(-1.0, 1.0, size=(seq_len, head_dim)).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=(seq_len, head_dim)).astype(np.float32)

    key_indices = np.stack(
        [np.argmin(np.abs(row[:, None] - codebook.centroids[None, :]), axis=1) for row in keys],
        axis=0,
    ).astype(np.uint8)
    value_indices = np.stack(
        [np.argmin(np.abs(row[:, None] - codebook.centroids[None, :]), axis=1) for row in values],
        axis=0,
    ).astype(np.uint8)

    packed_k = np.stack([pack_indices(row, bit_width) for row in key_indices], axis=0)
    packed_v = np.stack([pack_indices(row, bit_width) for row in value_indices], axis=0)

    out = decode_packed_attention(
        mlx.array(query),
        mlx.array(packed_k.astype(np.uint32)),
        mlx.array(packed_v.astype(np.uint32)),
        mlx.array(codebook.centroids),
        mlx.array(codebook.centroids),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    mlx.eval(out)

    reference = streaming_attention_decode(
        query,
        codebook.centroids[key_indices],
        codebook.centroids[value_indices],
    )
    np.testing.assert_allclose(np.array(out), reference, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not metal_available(), reason="Metal toolchain unavailable")
def test_fused_and_split_decode_match() -> None:
    bit_width = 4
    head_dim = 64
    codebook = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )
    rng = np.random.default_rng(1)
    seq_len = 11
    query = rng.uniform(-1.0, 1.0, size=(head_dim,)).astype(np.float32)
    keys = rng.uniform(-1.0, 1.0, size=(seq_len, head_dim)).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=(seq_len, head_dim)).astype(np.float32)

    key_indices = np.stack(
        [np.argmin(np.abs(row[:, None] - codebook.centroids[None, :]), axis=1) for row in keys],
        axis=0,
    ).astype(np.uint8)
    value_indices = np.stack(
        [np.argmin(np.abs(row[:, None] - codebook.centroids[None, :]), axis=1) for row in values],
        axis=0,
    ).astype(np.uint8)
    packed_k = np.stack([pack_indices(row, bit_width) for row in key_indices], axis=0)
    packed_v = np.stack([pack_indices(row, bit_width) for row in value_indices], axis=0)

    fused = decode_packed_attention(
        mlx.array(query),
        mlx.array(packed_k.astype(np.uint32)),
        mlx.array(packed_v.astype(np.uint32)),
        mlx.array(codebook.centroids),
        mlx.array(codebook.centroids),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    split = decode_packed_attention_split(
        mlx.array(query),
        mlx.array(packed_k.astype(np.uint32)),
        mlx.array(packed_v.astype(np.uint32)),
        mlx.array(codebook.centroids),
        mlx.array(codebook.centroids),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    mlx.eval(fused, split)
    np.testing.assert_allclose(np.array(fused), np.array(split), atol=1e-5, rtol=1e-5)
