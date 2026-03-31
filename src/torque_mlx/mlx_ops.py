from __future__ import annotations

from functools import lru_cache
from math import sqrt
import os
from pathlib import Path
import subprocess

import mlx.core as mx

from torque_mlx.layout import PackedKVLayout


def ensure_metal_toolchain() -> None:
    """Prefer a local Xcode install when xcode-select still points at CLT."""
    if os.environ.get("DEVELOPER_DIR"):
        return

    xcode_dir = Path("/Applications/Xcode.app/Contents/Developer")
    if not xcode_dir.exists():
        return

    try:
        subprocess.run(
            ["xcrun", "--find", "metal"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        os.environ["DEVELOPER_DIR"] = str(xcode_dir)


def metal_available() -> bool:
    ensure_metal_toolchain()
    try:
        subprocess.run(
            ["xcrun", "--find", "metal"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _score_kernel(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint token = thread_position_in_grid.x;
    float acc = 0.0f;
    constexpr uint MASK = (1u << BITW) - 1u;
    uint bit_offset = 0u;
    uint row_base = token * PACKED_WORDS;

    for (uint d = 0; d < HEAD_DIM; ++d) {
      uint word_index = bit_offset >> 5;
      uint shift = bit_offset & 31u;
      uint value = packed[row_base + word_index] >> shift;
      uint spill = shift + BITW;
      if (spill > 32u) {
        value |= packed[row_base + word_index + 1u] << (32u - shift);
      }
      uint idx = value & MASK;
      acc += query[d] * centroids[idx];
      bit_offset += BITW;
    }

    scores[token] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"torque_score_b{bit_width}_h{head_dim}",
        input_names=["query", "packed", "centroids"],
        output_names=["scores"],
        source=source,
    ), layout


@lru_cache(maxsize=None)
def _value_kernel(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint dim = thread_position_in_grid.x;
    float acc = 0.0f;
    constexpr uint MASK = (1u << BITW) - 1u;
    uint bit_offset = dim * BITW;
    uint word_index = bit_offset >> 5;
    uint shift = bit_offset & 31u;

    for (uint token = 0; token < SEQ_LEN; ++token) {
      uint row_base = token * PACKED_WORDS;
      uint value = packed[row_base + word_index] >> shift;
      uint spill = shift + BITW;
      if (spill > 32u) {
        value |= packed[row_base + word_index + 1u] << (32u - shift);
      }
      uint idx = value & MASK;
      acc += weights[token] * centroids[idx];
    }

    out[dim] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"torque_values_b{bit_width}_h{head_dim}",
        input_names=["weights", "packed", "centroids"],
        output_names=["out"],
        source=source,
    ), layout


def score_packed_query(
    query: mx.array,
    packed: mx.array,
    centroids: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    ensure_metal_toolchain()
    kernel, layout = _score_kernel(bit_width, head_dim)
    if packed.shape[-1] != layout.packed_words:
        raise ValueError(
            f"packed width mismatch: expected {layout.packed_words}, got {packed.shape[-1]}",
        )
    seq_len = packed.shape[0]
    return kernel(
        inputs=[query, packed, centroids],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
        ],
        grid=(seq_len, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
    )[0]


def accumulate_packed_values(
    weights: mx.array,
    packed: mx.array,
    centroids: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    ensure_metal_toolchain()
    kernel, layout = _value_kernel(bit_width, head_dim)
    if packed.shape[-1] != layout.packed_words:
        raise ValueError(
            f"packed width mismatch: expected {layout.packed_words}, got {packed.shape[-1]}",
        )
    seq_len = packed.shape[0]
    return kernel(
        inputs=[weights, packed, centroids],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("SEQ_LEN", seq_len),
        ],
        grid=(head_dim, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(head_dim,)],
        output_dtypes=[mx.float32],
    )[0]


def decode_packed_attention(
    query: mx.array,
    k_codes: mx.array,
    v_codes: mx.array,
    centroids_k: mx.array,
    centroids_v: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    scores = score_packed_query(
        query,
        k_codes,
        centroids_k,
        bit_width=bit_width,
        head_dim=head_dim,
    )
    weights = mx.softmax(scores * (1.0 / sqrt(head_dim)), axis=0)
    return accumulate_packed_values(
        weights,
        v_codes,
        centroids_v,
        bit_width=bit_width,
        head_dim=head_dim,
    )
