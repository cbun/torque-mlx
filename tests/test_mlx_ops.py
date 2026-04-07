import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

from torque_mlx.mlx_ops import (
    decode_packed_attention,
    decode_packed_attention_split,
    decode_packed_attention_split_batched,
    metal_available,
    quantize_and_pack_rows_metal,
    quantize_and_pack_rows_dual_metal,
)
from torque_mlx.quantization import Codebook, codebook_boundaries, pack_indices, pack_indices_batched, quantize
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
@pytest.mark.parametrize("bit_width", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_quantize_and_pack_rows_metal_matches_reference(bit_width: int, head_dim: int) -> None:
    codebook = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )
    rng = np.random.default_rng(12)
    values = rng.uniform(-1.0, 1.0, size=(5, head_dim)).astype(np.float32)

    actual = quantize_and_pack_rows_metal(
        mlx.array(values),
        mlx.array(codebook_boundaries(codebook)),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    mlx.eval(actual)

    expected = pack_indices_batched(quantize(values, codebook), bit_width)
    np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.skipif(not metal_available(), reason="Metal toolchain unavailable")
@pytest.mark.parametrize("bit_width", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_quantize_and_pack_rows_dual_metal_matches_reference(bit_width: int, head_dim: int) -> None:
    codebook_a = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform_a",
    )
    codebook_b = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-0.75, 0.75, 1 << bit_width, dtype=np.float32),
        name="uniform_b",
    )
    rng = np.random.default_rng(13)
    values_a = rng.uniform(-1.0, 1.0, size=(5, head_dim)).astype(np.float32)
    values_b = rng.uniform(-0.75, 0.75, size=(5, head_dim)).astype(np.float32)

    actual_a, actual_b = quantize_and_pack_rows_dual_metal(
        mlx.array(values_a),
        mlx.array(values_b),
        mlx.array(codebook_boundaries(codebook_a)),
        mlx.array(codebook_boundaries(codebook_b)),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    mlx.eval(actual_a, actual_b)

    expected_a = pack_indices_batched(quantize(values_a, codebook_a), bit_width)
    expected_b = pack_indices_batched(quantize(values_b, codebook_b), bit_width)
    np.testing.assert_array_equal(np.array(actual_a), expected_a)
    np.testing.assert_array_equal(np.array(actual_b), expected_b)


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


@pytest.mark.skipif(not metal_available(), reason="Metal toolchain unavailable")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batched_split_decode_matches_reference(head_dim: int) -> None:
    bit_width = 4
    batch_size = 5
    seq_len = 9
    codebook = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )
    rng = np.random.default_rng(2)
    query = rng.uniform(-1.0, 1.0, size=(batch_size, head_dim)).astype(np.float32)
    keys = rng.uniform(-1.0, 1.0, size=(batch_size, seq_len, head_dim)).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=(batch_size, seq_len, head_dim)).astype(np.float32)

    centroids = codebook.centroids.reshape(1, 1, 1, -1)
    key_indices = np.argmin(np.abs(keys[..., None] - centroids), axis=-1).astype(np.uint8)
    value_indices = np.argmin(np.abs(values[..., None] - centroids), axis=-1).astype(np.uint8)
    packed_k = np.stack(
        [np.stack([pack_indices(row, bit_width) for row in batch_rows], axis=0) for batch_rows in key_indices],
        axis=0,
    ).astype(np.uint32)
    packed_v = np.stack(
        [np.stack([pack_indices(row, bit_width) for row in batch_rows], axis=0) for batch_rows in value_indices],
        axis=0,
    ).astype(np.uint32)

    actual = decode_packed_attention_split_batched(
        mlx.array(query),
        mlx.array(packed_k),
        mlx.array(packed_v),
        mlx.array(codebook.centroids),
        mlx.array(codebook.centroids),
        bit_width=bit_width,
        head_dim=head_dim,
    )
    mlx.eval(actual)

    expected = np.stack(
        [
            streaming_attention_decode(
                query[batch_idx],
                codebook.centroids[key_indices[batch_idx]],
                codebook.centroids[value_indices[batch_idx]],
            )
            for batch_idx in range(batch_size)
        ],
        axis=0,
    )
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-5, rtol=1e-5)
