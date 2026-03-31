from __future__ import annotations

from dataclasses import dataclass
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
