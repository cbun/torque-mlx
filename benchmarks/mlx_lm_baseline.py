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
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.mlx_ops import metal_available
from torque_mlx.quantization import Codebook


def _uniform_codebook(bit_width: int) -> Codebook:
    return Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )


def run_benchmark(*, seq_len: int, head_dim: int, bit_width: int, seed: int) -> dict[str, float]:
    if not metal_available():
        raise RuntimeError("Metal toolchain unavailable for benchmark")

    rng = np.random.default_rng(seed)
    keys_np = rng.uniform(-1.0, 1.0, size=(1, 1, seq_len, head_dim)).astype(np.float32)
    values_np = rng.uniform(-1.0, 1.0, size=(1, 1, seq_len, head_dim)).astype(np.float32)
    query_np = rng.uniform(-1.0, 1.0, size=(1, 1, 1, head_dim)).astype(np.float32)
    scale = head_dim ** -0.5

    keys = mx.array(keys_np)
    values = mx.array(values_np)
    query = mx.array(query_np)

    fp_cache = KVCache()
    fp_keys, fp_values = fp_cache.update_and_fetch(keys, values)
    mx.eval(fp_keys, fp_values)

    quant_cache = QuantizedKVCache(group_size=64, bits=bit_width)
    q_keys, q_values = quant_cache.update_and_fetch(keys, values)
    mx.eval(*q_keys, *q_values)

    torque_cache = TorqueKVCache(
        config=TorqueConfig(bit_width=bit_width, head_dim=head_dim),
        key_codebook=_uniform_codebook(bit_width),
        value_codebook=_uniform_codebook(bit_width),
    )
    for token_idx in range(seq_len):
        torque_cache.append(
            key=keys_np[0, 0, token_idx],
            value=values_np[0, 0, token_idx],
        )

    warm_fp = scaled_dot_product_attention(query, fp_keys, fp_values, fp_cache, scale=scale, mask=None)
    warm_q = scaled_dot_product_attention(query, q_keys, q_values, quant_cache, scale=scale, mask=None)
    warm_t = torque_cache.decode_mlx(query=query_np[0, 0, 0])
    mx.eval(warm_fp, warm_q)
    _ = warm_t

    started = perf_counter()
    out_fp = scaled_dot_product_attention(query, fp_keys, fp_values, fp_cache, scale=scale, mask=None)
    mx.eval(out_fp)
    fp_elapsed = perf_counter() - started

    started = perf_counter()
    out_q = scaled_dot_product_attention(query, q_keys, q_values, quant_cache, scale=scale, mask=None)
    mx.eval(out_q)
    quant_elapsed = perf_counter() - started

    started = perf_counter()
    out_t = torque_cache.decode_mlx(query=query_np[0, 0, 0])
    torque_elapsed = perf_counter() - started

    out_fp_np = np.array(out_fp)[0, 0, 0]
    out_q_np = np.array(out_q)[0, 0, 0]

    return {
        "seq_len": float(seq_len),
        "head_dim": float(head_dim),
        "bit_width": float(bit_width),
        "mlx_fp16_decode_ms": fp_elapsed * 1_000.0,
        "mlx_fp16_tokens_per_sec": 1.0 / fp_elapsed,
        "mlx_lm_quantized_decode_ms": quant_elapsed * 1_000.0,
        "mlx_lm_quantized_tokens_per_sec": 1.0 / quant_elapsed,
        "torque_mlx_decode_ms": torque_elapsed * 1_000.0,
        "torque_mlx_tokens_per_sec": 1.0 / torque_elapsed,
        "max_abs_error_quantized_vs_fp16": float(np.max(np.abs(out_q_np - out_fp_np))),
        "max_abs_error_torque_vs_fp16": float(np.max(np.abs(out_t - out_fp_np))),
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
