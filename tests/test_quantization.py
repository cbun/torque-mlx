from pathlib import Path

import numpy as np
import pytest

from torque_mlx.quantization import (
    Codebook,
    build_gaussian_codebook,
    codebook_boundaries,
    dequantize,
    load_codebook,
    pack_indices,
    pack_indices_batched,
    pack_indices_batched_mlx,
    packed_words_for_head_dim,
    quantize,
    quantize_mlx,
    save_codebook,
    unpack_indices,
)


@pytest.mark.parametrize(
    ("head_dim", "bit_width", "expected_words"),
    [
        (64, 2, 4),
        (64, 3, 6),
        (64, 4, 8),
        (128, 2, 8),
        (128, 3, 12),
        (128, 4, 16),
        (256, 2, 16),
        (256, 3, 24),
        (256, 4, 32),
    ],
)
def test_packed_words_for_head_dim(
    head_dim: int,
    bit_width: int,
    expected_words: int,
) -> None:
    assert packed_words_for_head_dim(head_dim, bit_width) == expected_words


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_pack_unpack_roundtrip(bit_width: int) -> None:
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 1 << bit_width, size=128, dtype=np.uint8)
    packed = pack_indices(indices, bit_width)
    unpacked = unpack_indices(packed, bit_width, indices.size)
    np.testing.assert_array_equal(unpacked, indices)


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_pack_indices_batched_matches_rowwise_pack(bit_width: int) -> None:
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 1 << bit_width, size=(3, 5, 128), dtype=np.uint8)
    packed = pack_indices_batched(indices, bit_width)
    expected = np.stack(
        [
            np.stack([pack_indices(row, bit_width) for row in batch], axis=0)
            for batch in indices
        ],
        axis=0,
    )
    np.testing.assert_array_equal(packed, expected)


def test_quantize_and_dequantize_roundtrip_nearest_centroid() -> None:
    codebook = Codebook(bit_width=2, centroids=np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32))
    values = np.array([-0.9, -0.2, 0.3, 0.8], dtype=np.float32)
    indices = quantize(values, codebook)
    decoded = dequantize(indices, codebook)
    np.testing.assert_allclose(decoded, np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32))


def test_quantize_mlx_matches_numpy_reference() -> None:
    pytest.importorskip("mlx.core")

    codebook = Codebook(bit_width=2, centroids=np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32))
    values = np.array([[-0.9, -0.2, 0.3, 0.8]], dtype=np.float32)

    actual = np.array(quantize_mlx(values, codebook))
    expected = quantize(values, codebook)

    np.testing.assert_array_equal(actual, expected)


def test_quantize_mlx_matches_boundary_tie_break_reference() -> None:
    pytest.importorskip("mlx.core")

    codebook = Codebook(bit_width=2, centroids=np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32))
    values = np.array([[-0.625, 0.0, 0.625]], dtype=np.float32)

    actual = np.array(quantize_mlx(values, codebook, boundaries=codebook_boundaries(codebook)))
    expected = quantize(values, codebook)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_pack_indices_batched_mlx_matches_numpy_reference(bit_width: int) -> None:
    pytest.importorskip("mlx.core")

    rng = np.random.default_rng(4)
    indices = rng.integers(0, 1 << bit_width, size=(2, 3, 64), dtype=np.uint8)

    actual = np.array(pack_indices_batched_mlx(indices, bit_width))
    expected = pack_indices_batched(indices, bit_width)

    np.testing.assert_array_equal(actual, expected)


def test_codebook_json_roundtrip(tmp_path: Path) -> None:
    codebook = build_gaussian_codebook(4, sample_size=20_000, iterations=12, seed=7)
    path = tmp_path / "codebook.json"
    save_codebook(codebook, path)
    loaded = load_codebook(path)
    assert loaded.bit_width == codebook.bit_width
    assert loaded.name == codebook.name
    np.testing.assert_allclose(loaded.centroids, codebook.centroids)
