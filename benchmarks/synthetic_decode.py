from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.quantization import kv_bytes_per_token
from torque_mlx.reference import streaming_attention_decode
from torque_mlx.rotation import RotationSpec


def run_synthetic_decode(
    *,
    seq_len: int,
    head_dim: int,
    kv_heads: int,
    bit_width: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    config = TorqueConfig(
        bit_width=bit_width,
        head_dim=head_dim,
        num_layers=1,
        kv_heads=kv_heads,
        rotation_seed=seed,
    )
    cache = TorqueKVCache(config=config)

    keys = rng.normal(size=(seq_len, kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(size=(seq_len, kv_heads, head_dim)).astype(np.float32)
    query = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)

    for token_idx in range(seq_len):
        cache.append(key=keys[token_idx], value=values[token_idx])

    first_token_cache = TorqueKVCache(config=config)
    first_token_cache.append(key=keys[0], value=values[0])

    rotation = RotationSpec.from_seed(head_dim=head_dim, seed=seed)
    started = perf_counter()
    quantized_out = cache.decode(query=query)
    quantized_elapsed = perf_counter() - started

    started = perf_counter()
    first_token_cache.decode(query=query)
    first_decode_elapsed = perf_counter() - started

    started = perf_counter()
    baseline_out = np.zeros_like(query)
    for head_idx in range(kv_heads):
        baseline_out[head_idx] = streaming_attention_decode(
            rotation.apply(query[head_idx]),
            rotation.apply(keys[:, head_idx, :]),
            rotation.apply(values[:, head_idx, :]),
        )
        baseline_out[head_idx] = rotation.inverse(baseline_out[head_idx])
    baseline_elapsed = perf_counter() - started

    max_abs_error = float(np.max(np.abs(quantized_out - baseline_out)))
    return {
        "seq_len": float(seq_len),
        "head_dim": float(head_dim),
        "kv_heads": float(kv_heads),
        "bit_width": float(bit_width),
        "naive_quantized_decode_ms": quantized_elapsed * 1_000.0,
        "naive_quantized_tokens_per_sec": 1.0 / quantized_elapsed,
        "ttft_proxy_ms": first_decode_elapsed * 1_000.0,
        "reference_decode_ms": baseline_elapsed * 1_000.0,
        "reference_tokens_per_sec": 1.0 / baseline_elapsed,
        "kv_bytes_per_token": float(kv_bytes_per_token(head_dim, bit_width, kv_heads)),
        "max_abs_error_vs_rotated_reference": max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_synthetic_decode(
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        kv_heads=args.kv_heads,
        bit_width=args.bit_width,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
