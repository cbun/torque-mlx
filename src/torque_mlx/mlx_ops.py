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


@lru_cache(maxsize=None)
def _score_kernel_batched(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint token = thread_position_in_grid.x;
    uint batch = thread_position_in_grid.y;
    float acc = 0.0f;
    constexpr uint MASK = (1u << BITW) - 1u;
    uint row_base = ((batch * SEQ_LEN) + token) * PACKED_WORDS;
    uint query_base = batch * HEAD_DIM;
    uint bit_offset = 0u;

    for (uint d = 0; d < HEAD_DIM; ++d) {
      uint word_index = bit_offset >> 5;
      uint shift = bit_offset & 31u;
      uint value = packed[row_base + word_index] >> shift;
      uint spill = shift + BITW;
      if (spill > 32u) {
        value |= packed[row_base + word_index + 1u] << (32u - shift);
      }
      uint idx = value & MASK;
      acc += query[query_base + d] * centroids[idx];
      bit_offset += BITW;
    }

    scores[batch * SEQ_LEN + token] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"torque_score_batched_b{bit_width}_h{head_dim}",
        input_names=["query", "packed", "centroids"],
        output_names=["scores"],
        source=source,
    ), layout


@lru_cache(maxsize=None)
def _value_kernel_batched(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint dim = thread_position_in_grid.x;
    uint batch = thread_position_in_grid.y;
    float acc = 0.0f;
    constexpr uint MASK = (1u << BITW) - 1u;
    uint bit_offset = dim * BITW;
    uint word_index = bit_offset >> 5;
    uint shift = bit_offset & 31u;

    for (uint token = 0; token < SEQ_LEN; ++token) {
      uint row_base = ((batch * SEQ_LEN) + token) * PACKED_WORDS;
      uint value = packed[row_base + word_index] >> shift;
      uint spill = shift + BITW;
      if (spill > 32u) {
        value |= packed[row_base + word_index + 1u] << (32u - shift);
      }
      uint idx = value & MASK;
      acc += weights[batch * SEQ_LEN + token] * centroids[idx];
    }

    out[batch * HEAD_DIM + dim] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"torque_values_batched_b{bit_width}_h{head_dim}",
        input_names=["weights", "packed", "centroids"],
        output_names=["out"],
        source=source,
    ), layout


def _fused_decode_kernel(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    if (thread_position_in_grid.x != 0) {
      return;
    }

    constexpr uint MASK = (1u << BITW) - 1u;
    float out_local[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {
      out_local[d] = 0.0f;
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float scale = rsqrt((float)HEAD_DIM);

    for (uint token = 0; token < SEQ_LEN; ++token) {
      uint row_base = token * PACKED_WORDS;
      float score = 0.0f;
      uint bit_offset = 0u;

      for (uint d = 0; d < HEAD_DIM; ++d) {
        uint word_index = bit_offset >> 5;
        uint shift = bit_offset & 31u;
        uint value = k_codes[row_base + word_index] >> shift;
        uint spill = shift + BITW;
        if (spill > 32u) {
          value |= k_codes[row_base + word_index + 1u] << (32u - shift);
        }
        uint idx = value & MASK;
        score += query[d] * centroids_k[idx];
        bit_offset += BITW;
      }

      score *= scale;
      float m_new = metal::max(m_prev, score);
      float l_scale = metal::isinf(m_prev) ? 0.0f : metal::exp(m_prev - m_new);
      float p = metal::exp(score - m_new);

      for (uint d = 0; d < HEAD_DIM; ++d) {
        out_local[d] *= l_scale;
      }

      bit_offset = 0u;
      for (uint d = 0; d < HEAD_DIM; ++d) {
        uint word_index = bit_offset >> 5;
        uint shift = bit_offset & 31u;
        uint value = v_codes[row_base + word_index] >> shift;
        uint spill = shift + BITW;
        if (spill > 32u) {
          value |= v_codes[row_base + word_index + 1u] << (32u - shift);
        }
        uint idx = value & MASK;
        out_local[d] += p * centroids_v[idx];
        bit_offset += BITW;
      }

      l_prev = l_prev * l_scale + p;
      m_prev = m_new;
    }

    float inv_l = l_prev > 0.0f ? 1.0f / l_prev : 0.0f;
    for (uint d = 0; d < HEAD_DIM; ++d) {
      out[d] = out_local[d] * inv_l;
    }
    """
    return mx.fast.metal_kernel(
        name=f"torque_fused_decode_b{bit_width}_h{head_dim}",
        input_names=["query", "k_codes", "v_codes", "centroids_k", "centroids_v"],
        output_names=["out"],
        source=source,
    ), layout


@lru_cache(maxsize=None)
def _append_pack_kernel(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint word = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint packed_word = 0u;

    for (uint d = 0; d < HEAD_DIM; ++d) {
      uint bit_offset = d * BITW;
      uint base_word = bit_offset >> 5;
      uint shift = bit_offset & 31u;
      float value = values[row * HEAD_DIM + d];
      uint idx = 0u;

      for (uint b = 0; b < LEVELS_MINUS_ONE; ++b) {
        idx += value > boundaries[b] ? 1u : 0u;
      }

      if (base_word == word) {
        packed_word |= idx << shift;
      }

      uint spill = shift + BITW;
      if (spill > 32u && (base_word + 1u) == word) {
        packed_word |= idx >> (BITW - (spill - 32u));
      }
    }

    packed[row * PACKED_WORDS + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name=f"torque_append_pack_b{bit_width}_h{head_dim}",
        input_names=["values", "boundaries"],
        output_names=["packed"],
        source=source,
    ), layout


@lru_cache(maxsize=None)
def _append_pack_dual_kernel(bit_width: int, head_dim: int) -> object:
    layout = PackedKVLayout(bit_width=bit_width, head_dim=head_dim)
    source = """
    uint word = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint packed_a_word = 0u;
    uint packed_b_word = 0u;

    for (uint d = 0; d < HEAD_DIM; ++d) {
      uint bit_offset = d * BITW;
      uint base_word = bit_offset >> 5;
      uint shift = bit_offset & 31u;
      float value_a = values_a[row * HEAD_DIM + d];
      float value_b = values_b[row * HEAD_DIM + d];
      uint idx_a = 0u;
      uint idx_b = 0u;

      for (uint b = 0; b < LEVELS_MINUS_ONE; ++b) {
        idx_a += value_a > boundaries_a[b] ? 1u : 0u;
        idx_b += value_b > boundaries_b[b] ? 1u : 0u;
      }

      if (base_word == word) {
        packed_a_word |= idx_a << shift;
        packed_b_word |= idx_b << shift;
      }

      uint spill = shift + BITW;
      if (spill > 32u && (base_word + 1u) == word) {
        uint spill_shift = BITW - (spill - 32u);
        packed_a_word |= idx_a >> spill_shift;
        packed_b_word |= idx_b >> spill_shift;
      }
    }

    packed_a[row * PACKED_WORDS + word] = packed_a_word;
    packed_b[row * PACKED_WORDS + word] = packed_b_word;
    """
    return mx.fast.metal_kernel(
        name=f"torque_append_pack_dual_b{bit_width}_h{head_dim}",
        input_names=["values_a", "values_b", "boundaries_a", "boundaries_b"],
        output_names=["packed_a", "packed_b"],
        source=source,
    ), layout


def quantize_and_pack_rows_metal(
    values: mx.array,
    boundaries: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    ensure_metal_toolchain()
    kernel, layout = _append_pack_kernel(bit_width, head_dim)
    if len(values.shape) != 2 or values.shape[1] != head_dim:
        raise ValueError(f"values must have shape (rows, {head_dim}), got {values.shape}")
    expected_boundaries = (1 << bit_width) - 1
    if len(boundaries.shape) != 1 or boundaries.shape[0] != expected_boundaries:
        raise ValueError(
            f"boundaries must have shape ({expected_boundaries},), got {boundaries.shape}",
        )
    rows = int(values.shape[0])
    return kernel(
        inputs=[values, boundaries],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("LEVELS_MINUS_ONE", expected_boundaries),
        ],
        grid=(layout.packed_words, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, layout.packed_words)],
        output_dtypes=[mx.uint32],
    )[0]


def quantize_and_pack_rows_dual_metal(
    values_a: mx.array,
    values_b: mx.array,
    boundaries_a: mx.array,
    boundaries_b: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> tuple[mx.array, mx.array]:
    ensure_metal_toolchain()
    kernel, layout = _append_pack_dual_kernel(bit_width, head_dim)
    if len(values_a.shape) != 2 or values_a.shape[1] != head_dim:
        raise ValueError(f"values_a must have shape (rows, {head_dim}), got {values_a.shape}")
    if values_a.shape != values_b.shape:
        raise ValueError(f"values_a and values_b must match, got {values_a.shape} and {values_b.shape}")
    expected_boundaries = (1 << bit_width) - 1
    if len(boundaries_a.shape) != 1 or boundaries_a.shape[0] != expected_boundaries:
        raise ValueError(
            f"boundaries_a must have shape ({expected_boundaries},), got {boundaries_a.shape}",
        )
    if len(boundaries_b.shape) != 1 or boundaries_b.shape[0] != expected_boundaries:
        raise ValueError(
            f"boundaries_b must have shape ({expected_boundaries},), got {boundaries_b.shape}",
        )
    rows = int(values_a.shape[0])
    outputs = kernel(
        inputs=[values_a, values_b, boundaries_a, boundaries_b],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("LEVELS_MINUS_ONE", expected_boundaries),
        ],
        grid=(layout.packed_words, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, layout.packed_words), (rows, layout.packed_words)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return outputs[0], outputs[1]


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


def decode_packed_attention_split(
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


def score_packed_query_batched(
    query: mx.array,
    packed: mx.array,
    centroids: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    ensure_metal_toolchain()
    kernel, layout = _score_kernel_batched(bit_width, head_dim)
    if len(query.shape) != 2 or query.shape[1] != head_dim:
        raise ValueError(f"batched query must have shape (batch, {head_dim}), got {query.shape}")
    if len(packed.shape) != 3 or packed.shape[2] != layout.packed_words:
        raise ValueError(
            f"batched packed codes must have shape (batch, seq, {layout.packed_words}), got {packed.shape}",
        )
    batch_size = packed.shape[0]
    seq_len = packed.shape[1]
    return kernel(
        inputs=[query, packed, centroids],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("SEQ_LEN", seq_len),
        ],
        grid=(seq_len, batch_size, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch_size, seq_len)],
        output_dtypes=[mx.float32],
    )[0]


def accumulate_packed_values_batched(
    weights: mx.array,
    packed: mx.array,
    centroids: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    ensure_metal_toolchain()
    kernel, layout = _value_kernel_batched(bit_width, head_dim)
    if len(weights.shape) != 2:
        raise ValueError(f"batched weights must have shape (batch, seq), got {weights.shape}")
    if len(packed.shape) != 3 or packed.shape[2] != layout.packed_words:
        raise ValueError(
            f"batched packed codes must have shape (batch, seq, {layout.packed_words}), got {packed.shape}",
        )
    if packed.shape[0] != weights.shape[0] or packed.shape[1] != weights.shape[1]:
        raise ValueError(
            "batched packed codes and weights must agree on batch and sequence dimensions",
        )
    batch_size = packed.shape[0]
    seq_len = packed.shape[1]
    return kernel(
        inputs=[weights, packed, centroids],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("SEQ_LEN", seq_len),
        ],
        grid=(head_dim, batch_size, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch_size, head_dim)],
        output_dtypes=[mx.float32],
    )[0]


def decode_packed_attention_split_batched(
    query: mx.array,
    k_codes: mx.array,
    v_codes: mx.array,
    centroids_k: mx.array,
    centroids_v: mx.array,
    *,
    bit_width: int,
    head_dim: int,
) -> mx.array:
    scores = score_packed_query_batched(
        query,
        k_codes,
        centroids_k,
        bit_width=bit_width,
        head_dim=head_dim,
    )
    weights = mx.softmax(scores * (1.0 / sqrt(head_dim)), axis=1)
    return accumulate_packed_values_batched(
        weights,
        v_codes,
        centroids_v,
        bit_width=bit_width,
        head_dim=head_dim,
    )


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
    ensure_metal_toolchain()
    kernel, layout = _fused_decode_kernel(bit_width, head_dim)
    if k_codes.shape[-1] != layout.packed_words or v_codes.shape[-1] != layout.packed_words:
        raise ValueError(
            "packed width mismatch for fused decode kernel",
        )
    seq_len = k_codes.shape[0]
    return kernel(
        inputs=[query, k_codes, v_codes, centroids_k, centroids_v],
        template=[
            ("BITW", bit_width),
            ("HEAD_DIM", head_dim),
            ("PACKED_WORDS", layout.packed_words),
            ("SEQ_LEN", seq_len),
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(head_dim,)],
        output_dtypes=[mx.float32],
    )[0]
