from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from math import ceil
from pathlib import Path

import numpy as np


SUPPORTED_BIT_WIDTHS = {2, 3, 4}
WORD_BITS = 32


def _validate_bit_width(bit_width: int) -> None:
    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"Unsupported bit width: {bit_width}")


@dataclass(frozen=True, slots=True)
class Codebook:
    """Scalar quantization codebook."""

    bit_width: int
    centroids: np.ndarray
    name: str = "gaussian_lloyd_max"

    def __post_init__(self) -> None:
        _validate_bit_width(self.bit_width)
        object.__setattr__(
            self,
            "centroids",
            np.asarray(self.centroids, dtype=np.float32),
        )
        expected = 1 << self.bit_width
        if self.centroids.ndim != 1 or self.centroids.size != expected:
            raise ValueError(
                f"Expected {expected} centroids for {self.bit_width}-bit codebook, "
                f"got shape {self.centroids.shape}",
            )
        if not np.all(np.diff(self.centroids) >= 0):
            raise ValueError("Centroids must be sorted in ascending order")

    def to_dict(self) -> dict[str, object]:
        return {
            "bit_width": self.bit_width,
            "centroids": self.centroids.tolist(),
            "name": self.name,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Codebook":
        return cls(
            bit_width=int(payload["bit_width"]),
            centroids=np.asarray(payload["centroids"], dtype=np.float32),
            name=str(payload.get("name", "gaussian_lloyd_max")),
        )

    @classmethod
    def from_json(cls, payload: str) -> "Codebook":
        return cls.from_dict(json.loads(payload))


def save_codebook(codebook: Codebook, path: str | Path) -> None:
    Path(path).write_text(codebook.to_json(), encoding="utf-8")


def load_codebook(path: str | Path) -> Codebook:
    return Codebook.from_json(Path(path).read_text(encoding="utf-8"))


def packed_words_for_head_dim(head_dim: int, bit_width: int) -> int:
    """Return the number of 32-bit words required for one packed vector."""
    _validate_bit_width(bit_width)
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return ceil((head_dim * bit_width) / WORD_BITS)


def kv_bytes_per_token(head_dim: int, bit_width: int, kv_heads: int = 1) -> int:
    words = packed_words_for_head_dim(head_dim=head_dim, bit_width=bit_width)
    return words * 4 * kv_heads * 2


def build_gaussian_codebook(
    bit_width: int,
    *,
    iterations: int = 32,
    sample_size: int = 200_000,
    seed: int = 0,
) -> Codebook:
    """Build a Lloyd-Max codebook for a standard normal distribution."""
    _validate_bit_width(bit_width)
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=0.0, scale=1.0, size=sample_size).astype(np.float32)
    levels = 1 << bit_width
    quantiles = np.linspace(0.0, 1.0, levels + 2, dtype=np.float32)[1:-1]
    centroids = np.quantile(samples, quantiles).astype(np.float32)

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        assignments = np.searchsorted(boundaries, samples, side="right")
        new_centroids = centroids.copy()
        for idx in range(levels):
            cluster = samples[assignments == idx]
            if cluster.size:
                new_centroids[idx] = cluster.mean(dtype=np.float64)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return Codebook(bit_width=bit_width, centroids=np.sort(centroids))


def quantize(values: np.ndarray, codebook: Codebook) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    flat = array.reshape(-1, 1)
    distances = np.abs(flat - codebook.centroids.reshape(1, -1))
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    return indices.reshape(array.shape)


def codebook_boundaries(codebook: Codebook | np.ndarray | object) -> np.ndarray:
    centroids = (
        np.asarray(codebook.centroids, dtype=np.float32)
        if isinstance(codebook, Codebook)
        else np.asarray(codebook, dtype=np.float32)
    )
    if centroids.ndim != 1 or centroids.size < 2:
        raise ValueError("Expected at least two sorted centroids to build quantization boundaries")
    return ((centroids[:-1] + centroids[1:]) / 2.0).astype(np.float32)


def quantize_mlx(values, codebook: Codebook | object, *, boundaries=None):
    import mlx.core as mx

    array = mx.array(values, dtype=mx.float32)
    resolved_boundaries = (
        mx.array(codebook_boundaries(codebook), dtype=mx.float32)
        if boundaries is None
        else mx.array(boundaries, dtype=mx.float32)
    )
    return mx.sum(mx.expand_dims(array, axis=-1) > resolved_boundaries, axis=-1).astype(mx.uint32)


def dequantize(indices: np.ndarray, codebook: Codebook) -> np.ndarray:
    codes = np.asarray(indices)
    return codebook.centroids[codes].astype(np.float32)


def pack_indices(indices: np.ndarray, bit_width: int) -> np.ndarray:
    _validate_bit_width(bit_width)
    values = np.asarray(indices, dtype=np.uint32).reshape(-1)
    max_value = 1 << bit_width
    if np.any(values >= max_value):
        raise ValueError(f"Index out of range for {bit_width}-bit packing")

    words = np.zeros(packed_words_for_head_dim(values.size, bit_width), dtype=np.uint32)
    bit_offset = 0
    for value in values:
        word_index = bit_offset // WORD_BITS
        shift = bit_offset % WORD_BITS
        words[word_index] |= np.uint32(value << shift)
        spill = shift + bit_width - WORD_BITS
        if spill > 0:
            words[word_index + 1] |= np.uint32(value >> (bit_width - spill))
        bit_offset += bit_width
    return words


def pack_indices_batched(indices: np.ndarray, bit_width: int) -> np.ndarray:
    _validate_bit_width(bit_width)
    values = np.asarray(indices, dtype=np.uint32)
    if values.ndim == 1:
        return pack_indices(values, bit_width)

    count = values.shape[-1]
    flat = values.reshape(-1, count)
    max_value = 1 << bit_width
    if np.any(flat >= max_value):
        raise ValueError(f"Index out of range for {bit_width}-bit packing")

    packed = np.zeros(
        (flat.shape[0], packed_words_for_head_dim(count, bit_width)),
        dtype=np.uint32,
    )
    for idx in range(count):
        bit_offset = idx * bit_width
        word_index = bit_offset // WORD_BITS
        shift = bit_offset % WORD_BITS
        packed[:, word_index] |= flat[:, idx] << shift
        spill = shift + bit_width - WORD_BITS
        if spill > 0:
            packed[:, word_index + 1] |= flat[:, idx] >> (bit_width - spill)

    return packed.reshape(*values.shape[:-1], packed.shape[-1])


@lru_cache(maxsize=None)
def _pack_word_plan(count: int, bit_width: int):
    packed_words = packed_words_for_head_dim(count, bit_width)
    low_indices: list[list[int]] = [[] for _ in range(packed_words)]
    low_shifts: list[list[int]] = [[] for _ in range(packed_words)]
    spill_indices: list[list[int]] = [[] for _ in range(packed_words)]
    spill_shifts: list[list[int]] = [[] for _ in range(packed_words)]

    for idx in range(count):
        bit_offset = idx * bit_width
        word_index = bit_offset // WORD_BITS
        shift = bit_offset % WORD_BITS
        low_indices[word_index].append(idx)
        low_shifts[word_index].append(shift)
        spill = shift + bit_width - WORD_BITS
        if spill > 0:
            spill_indices[word_index + 1].append(idx)
            spill_shifts[word_index + 1].append(bit_width - spill)

    return tuple(
        (
            tuple(low_indices[word_index]),
            tuple(low_shifts[word_index]),
            tuple(spill_indices[word_index]),
            tuple(spill_shifts[word_index]),
        )
        for word_index in range(packed_words)
    )


def pack_indices_batched_mlx(indices, bit_width: int):
    import mlx.core as mx

    _validate_bit_width(bit_width)
    values = mx.array(indices, dtype=mx.uint32)
    if len(values.shape) == 1:
        values = mx.reshape(values, (1, int(values.shape[0])))
        squeeze = True
    else:
        squeeze = False

    count = int(values.shape[-1])
    flat = mx.reshape(values, (-1, count))
    max_value = 1 << bit_width
    if bool(mx.any(flat >= max_value).item()):
        raise ValueError(f"Index out of range for {bit_width}-bit packing")

    packed_words = packed_words_for_head_dim(count, bit_width)
    packed_columns = []
    for low_idx, low_shift, spill_idx, spill_shift in _pack_word_plan(count, bit_width):
        word_values = mx.zeros((int(flat.shape[0]),), dtype=mx.uint32)
        if low_idx:
            low_values = flat[:, list(low_idx)]
            low_shift_values = mx.array(low_shift, dtype=mx.uint32).reshape(1, -1)
            word_values = word_values + mx.sum(low_values << low_shift_values, axis=1).astype(mx.uint32)
        if spill_idx:
            spill_values = flat[:, list(spill_idx)]
            spill_shift_values = mx.array(spill_shift, dtype=mx.uint32).reshape(1, -1)
            word_values = word_values + mx.sum(spill_values >> spill_shift_values, axis=1).astype(mx.uint32)
        packed_columns.append(word_values)

    packed = mx.stack(packed_columns, axis=1) if packed_columns else mx.zeros((int(flat.shape[0]), 0), dtype=mx.uint32)
    result = mx.reshape(packed, (*values.shape[:-1], packed_words))
    if squeeze:
        return result[0]
    return result


def unpack_indices(words: np.ndarray, bit_width: int, count: int) -> np.ndarray:
    _validate_bit_width(bit_width)
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    if count < 0:
        raise ValueError("count must be non-negative")

    mask = (1 << bit_width) - 1
    values = np.zeros(count, dtype=np.uint8)
    bit_offset = 0
    for idx in range(count):
        word_index = bit_offset // WORD_BITS
        shift = bit_offset % WORD_BITS
        value = int(packed[word_index] >> shift)
        spill = shift + bit_width - WORD_BITS
        if spill > 0:
            value |= int(packed[word_index + 1] & ((1 << spill) - 1)) << (bit_width - spill)
        values[idx] = value & mask
        bit_offset += bit_width
    return values
