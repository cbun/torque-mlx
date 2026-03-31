from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx
import numpy as np

from torque_mlx.mlx_ops import (
    decode_packed_attention,
    decode_packed_attention_split,
    metal_available,
)
from torque_mlx.quantization import Codebook, pack_indices
from torque_mlx.reference import streaming_attention_decode


def run_benchmark(*, seq_len: int, head_dim: int, bit_width: int, seed: int) -> dict[str, float]:
    if not metal_available():
        raise RuntimeError("Metal toolchain unavailable for MLX packed decode benchmark")

    rng = np.random.default_rng(seed)
    codebook = Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )
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

    packed_k = np.stack([pack_indices(row, bit_width) for row in key_indices], axis=0).astype(np.uint32)
    packed_v = np.stack([pack_indices(row, bit_width) for row in value_indices], axis=0).astype(np.uint32)

    q = mx.array(query)
    k = mx.array(packed_k)
    v = mx.array(packed_v)
    cent = mx.array(codebook.centroids)

    warmup = decode_packed_attention(q, k, v, cent, cent, bit_width=bit_width, head_dim=head_dim)
    warmup_split = decode_packed_attention_split(q, k, v, cent, cent, bit_width=bit_width, head_dim=head_dim)
    mx.eval(warmup, warmup_split)

    started = perf_counter()
    out_fused = decode_packed_attention(q, k, v, cent, cent, bit_width=bit_width, head_dim=head_dim)
    mx.eval(out_fused)
    fused_elapsed = perf_counter() - started

    started = perf_counter()
    out_split = decode_packed_attention_split(q, k, v, cent, cent, bit_width=bit_width, head_dim=head_dim)
    mx.eval(out_split)
    split_elapsed = perf_counter() - started

    started = perf_counter()
    reference = streaming_attention_decode(
        query,
        codebook.centroids[key_indices],
        codebook.centroids[value_indices],
    )
    reference_elapsed = perf_counter() - started

    return {
        "seq_len": float(seq_len),
        "head_dim": float(head_dim),
        "bit_width": float(bit_width),
        "mlx_fused_decode_ms": fused_elapsed * 1_000.0,
        "mlx_fused_tokens_per_sec": 1.0 / fused_elapsed,
        "mlx_split_decode_ms": split_elapsed * 1_000.0,
        "mlx_split_tokens_per_sec": 1.0 / split_elapsed,
        "reference_decode_ms": reference_elapsed * 1_000.0,
        "reference_tokens_per_sec": 1.0 / reference_elapsed,
        "max_abs_error_fused": float(np.max(np.abs(np.array(out_fused) - reference))),
        "max_abs_error_split": float(np.max(np.abs(np.array(out_split) - reference))),
        "max_abs_diff_fused_vs_split": float(np.max(np.abs(np.array(out_fused) - np.array(out_split)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
