from __future__ import annotations

from time import perf_counter

import numpy as np

from torque_mlx.artifact import TorqueArtifact
from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.mlx_ops import decode_packed_attention, decode_packed_attention_split, metal_available
from torque_mlx.quantization import Codebook, kv_bytes_per_token, pack_indices
from torque_mlx.reference import streaming_attention_decode
from torque_mlx.rotation import RotationSpec


def build_uniform_codebook(bit_width: int) -> Codebook:
    return Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, 1 << bit_width, dtype=np.float32),
        name="uniform",
    )


def run_synthetic_decode_benchmark(
    *,
    seq_len: int,
    head_dim: int,
    kv_heads: int,
    bit_width: int,
    seed: int,
    rotation_seed: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    config = TorqueConfig(
        bit_width=bit_width,
        head_dim=head_dim,
        num_layers=1,
        kv_heads=kv_heads,
        rotation_seed=seed if rotation_seed is None else rotation_seed,
    )
    cache = TorqueKVCache(config=config)

    keys = rng.normal(size=(seq_len, kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(size=(seq_len, kv_heads, head_dim)).astype(np.float32)
    query = rng.normal(size=(kv_heads, head_dim)).astype(np.float32)

    for token_idx in range(seq_len):
        cache.append(key=keys[token_idx], value=values[token_idx])

    first_token_cache = TorqueKVCache(config=config)
    first_token_cache.append(key=keys[0], value=values[0])

    rotation = RotationSpec.from_seed(head_dim=head_dim, seed=config.rotation_seed)
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


def run_mlx_packed_decode_benchmark(
    *,
    seq_len: int,
    head_dim: int,
    bit_width: int,
    seed: int,
) -> dict[str, float]:
    if not metal_available():
        raise RuntimeError("Metal toolchain unavailable for MLX packed decode benchmark")

    import mlx.core as mx

    rng = np.random.default_rng(seed)
    codebook = build_uniform_codebook(bit_width)
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


def run_mlx_lm_baseline_benchmark(
    *,
    seq_len: int,
    head_dim: int,
    bit_width: int,
    seed: int,
) -> dict[str, float]:
    if not metal_available():
        raise RuntimeError("Metal toolchain unavailable for benchmark")

    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

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
        key_codebook=build_uniform_codebook(bit_width),
        value_codebook=build_uniform_codebook(bit_width),
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


def evaluate_artifact(
    artifact: TorqueArtifact,
    *,
    seq_len: int,
    seed: int,
) -> dict[str, object]:
    report = run_synthetic_decode_benchmark(
        seq_len=seq_len,
        head_dim=artifact.runtime_config.head_dim,
        kv_heads=artifact.runtime_config.kv_heads,
        bit_width=artifact.runtime_config.bit_width,
        seed=seed,
        rotation_seed=artifact.runtime_config.rotation_seed,
    )
    report["artifact"] = {
        "model_name": artifact.manifest.model_name,
        "variant_id": artifact.runtime_config.variant_id,
        "architecture": artifact.manifest.architecture,
    }
    return report
