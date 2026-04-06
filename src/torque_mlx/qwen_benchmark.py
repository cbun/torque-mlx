from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from torque_mlx.benchmarking import build_uniform_codebook
from torque_mlx.cache_mlx import SUPPORTED_DECODE_STRATEGIES, TorqueKVCacheMLX
from torque_mlx.config import TorqueConfig
from torque_mlx.families.qwen import (
    QWEN_MODEL_MANIFEST_FILE,
    inspect_qwen_hf_directory,
    load_qwen_model_manifest,
)
from torque_mlx.mlx_ops import metal_available
from torque_mlx.quantization import kv_bytes_per_token


def _fp16_kv_bytes_per_token(*, head_dim: int, kv_heads: int) -> int:
    return head_dim * kv_heads * 2 * 2


def _flatten_for_eval(value):
    if isinstance(value, tuple):
        flattened: list[object] = []
        for item in value:
            flattened.extend(_flatten_for_eval(item))
        return flattened
    return [value]


@dataclass(frozen=True, slots=True)
class QwenDecodeBenchmarkProfile:
    model_dir: str
    profile_source: str
    model_type: str
    text_model_type: str | None
    head_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    target_layer_indices: tuple[int, ...]
    bit_width: int
    rotation_seed: int
    fused_weights: bool
    has_vision_config: bool
    attn_output_gate: bool

    @property
    def target_layer_count(self) -> int:
        return len(self.target_layer_indices)

    @property
    def kv_group_ratio(self) -> float:
        return float(self.num_attention_heads) / float(self.num_key_value_heads)

    @property
    def torque_kv_bytes_per_token_per_layer(self) -> int:
        return kv_bytes_per_token(
            head_dim=self.head_dim,
            bit_width=self.bit_width,
            kv_heads=self.num_key_value_heads,
        )

    @property
    def fp16_kv_bytes_per_token_per_layer(self) -> int:
        return _fp16_kv_bytes_per_token(
            head_dim=self.head_dim,
            kv_heads=self.num_key_value_heads,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "qwen",
            "model_dir": self.model_dir,
            "profile_source": self.profile_source,
            "model_type": self.model_type,
            "text_model_type": self.text_model_type,
            "head_dim": self.head_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "kv_group_ratio": self.kv_group_ratio,
            "target_layer_indices": list(self.target_layer_indices),
            "target_layer_count": self.target_layer_count,
            "bit_width": self.bit_width,
            "rotation_seed": self.rotation_seed,
            "fused_weights": self.fused_weights,
            "has_vision_config": self.has_vision_config,
            "attn_output_gate": self.attn_output_gate,
            "fp16_kv_bytes_per_token_per_layer": self.fp16_kv_bytes_per_token_per_layer,
            "torque_kv_bytes_per_token_per_layer": self.torque_kv_bytes_per_token_per_layer,
            "fp16_kv_bytes_per_token_total": self.fp16_kv_bytes_per_token_per_layer * self.target_layer_count,
            "torque_kv_bytes_per_token_total": self.torque_kv_bytes_per_token_per_layer * self.target_layer_count,
        }


@dataclass(frozen=True, slots=True)
class QwenDecodeRuntimeBenchmarkResult:
    profile: QwenDecodeBenchmarkProfile
    prefill_tokens: int
    decode_steps: int
    seed: int
    torque_decode_strategy: str
    fp16_update_seconds: float
    fp16_attention_seconds: float
    fp16_decode_seconds: float
    mlx_lm_quantized_update_seconds: float
    mlx_lm_quantized_attention_seconds: float
    mlx_lm_quantized_decode_seconds: float
    torque_append_seconds: float
    torque_kernel_seconds: float
    torque_decode_seconds: float
    max_abs_error_quantized_vs_fp16: float
    max_abs_error_torque_vs_fp16: float

    def to_dict(self) -> dict[str, object]:
        total_tokens = self.prefill_tokens + self.decode_steps
        fp16_total_kv_bytes = self.profile.fp16_kv_bytes_per_token_per_layer * self.profile.target_layer_count * total_tokens
        torque_total_kv_bytes = self.profile.torque_kv_bytes_per_token_per_layer * self.profile.target_layer_count * total_tokens
        fp16_host_seconds = max(0.0, self.fp16_decode_seconds - self.fp16_update_seconds - self.fp16_attention_seconds)
        quant_host_seconds = max(
            0.0,
            self.mlx_lm_quantized_decode_seconds
            - self.mlx_lm_quantized_update_seconds
            - self.mlx_lm_quantized_attention_seconds,
        )
        torque_host_seconds = max(0.0, self.torque_decode_seconds - self.torque_append_seconds - self.torque_kernel_seconds)
        return {
            "family": "qwen",
            "benchmark": "qwen_decode_runtime",
            "profile": self.profile.to_dict(),
            "prefill_tokens": self.prefill_tokens,
            "decode_steps": self.decode_steps,
            "seed": self.seed,
            "torque_decode_strategy": self.torque_decode_strategy,
            "notes": [
                "This benchmark measures the KV-growing decode hot path only, not full dense forwards or perplexity.",
                "It uses the Qwen model geometry from the local snapshot or torque manifest.",
                "The current runtime models grouped-query attention explicitly using the source model's attention-head to KV-head ratio.",
            ],
            "fp16_kv_cache_total_bytes": fp16_total_kv_bytes,
            "torque_kv_cache_total_bytes": torque_total_kv_bytes,
            "fp16_kv_cache_total_gib": fp16_total_kv_bytes / float(1 << 30),
            "torque_kv_cache_total_gib": torque_total_kv_bytes / float(1 << 30),
            "kv_cache_bytes_saved_vs_fp16": fp16_total_kv_bytes - torque_total_kv_bytes,
            "mlx_fp16_update_ms": self.fp16_update_seconds * 1_000.0,
            "mlx_fp16_attention_ms": self.fp16_attention_seconds * 1_000.0,
            "mlx_fp16_host_overhead_ms": fp16_host_seconds * 1_000.0,
            "mlx_fp16_decode_ms": self.fp16_decode_seconds * 1_000.0,
            "mlx_fp16_tokens_per_sec": self.decode_steps / self.fp16_decode_seconds if self.fp16_decode_seconds > 0 else 0.0,
            "mlx_lm_quantized_update_ms": self.mlx_lm_quantized_update_seconds * 1_000.0,
            "mlx_lm_quantized_attention_ms": self.mlx_lm_quantized_attention_seconds * 1_000.0,
            "mlx_lm_quantized_host_overhead_ms": quant_host_seconds * 1_000.0,
            "mlx_lm_quantized_decode_ms": self.mlx_lm_quantized_decode_seconds * 1_000.0,
            "mlx_lm_quantized_tokens_per_sec": self.decode_steps / self.mlx_lm_quantized_decode_seconds if self.mlx_lm_quantized_decode_seconds > 0 else 0.0,
            "torque_mlx_append_ms": self.torque_append_seconds * 1_000.0,
            "torque_mlx_kernel_ms": self.torque_kernel_seconds * 1_000.0,
            "torque_mlx_host_overhead_ms": torque_host_seconds * 1_000.0,
            "torque_mlx_decode_ms": self.torque_decode_seconds * 1_000.0,
            "torque_mlx_tokens_per_sec": self.decode_steps / self.torque_decode_seconds if self.torque_decode_seconds > 0 else 0.0,
            "max_abs_error_quantized_vs_fp16": self.max_abs_error_quantized_vs_fp16,
            "max_abs_error_torque_vs_fp16": self.max_abs_error_torque_vs_fp16,
        }


def load_qwen_decode_benchmark_profile(
    model_dir: str | Path,
    *,
    bit_width: int | None = None,
    rotation_seed: int = 0,
    fused_weights: bool = False,
) -> QwenDecodeBenchmarkProfile:
    root = Path(model_dir)
    manifest_path = root / QWEN_MODEL_MANIFEST_FILE
    if manifest_path.exists():
        manifest = load_qwen_model_manifest(root)
        source_report = None
        source_root = Path(manifest.source_model_dir)
        if source_root.exists():
            source_report = inspect_qwen_hf_directory(source_root)
        return QwenDecodeBenchmarkProfile(
            model_dir=str(root.resolve()),
            profile_source="torque_manifest",
            model_type=manifest.source_model_type,
            text_model_type=manifest.source_text_model_type,
            head_dim=manifest.runtime_config.head_dim,
            num_hidden_layers=manifest.runtime_config.num_layers,
            num_attention_heads=(
                source_report.num_attention_heads
                if source_report is not None
                else manifest.runtime_config.kv_heads
            ),
            num_key_value_heads=manifest.runtime_config.kv_heads,
            target_layer_indices=tuple(manifest.converted_layer_indices),
            bit_width=manifest.runtime_config.bit_width,
            rotation_seed=manifest.runtime_config.rotation_seed,
            fused_weights=fused_weights,
            has_vision_config=manifest.has_vision_config,
            attn_output_gate=source_report.attn_output_gate if source_report is not None else False,
        )

    report = inspect_qwen_hf_directory(root)
    if not report.supported_runtime:
        raise ValueError(
            "Qwen model is not currently supported by torque-mlx: "
            + "; ".join(report.blocking_issues),
        )
    return QwenDecodeBenchmarkProfile(
        model_dir=str(root.resolve()),
        profile_source="hf_snapshot",
        model_type=report.model_type,
        text_model_type=report.text_model_type,
        head_dim=report.head_dim,
        num_hidden_layers=report.num_hidden_layers,
        num_attention_heads=report.num_attention_heads,
        num_key_value_heads=report.num_key_value_heads,
        target_layer_indices=tuple(report.full_attention_indices),
        bit_width=4 if bit_width is None else bit_width,
        rotation_seed=rotation_seed,
        fused_weights=fused_weights,
        has_vision_config=report.has_vision_config,
        attn_output_gate=report.attn_output_gate,
    )


def _build_decode_inputs(
    *,
    profile: QwenDecodeBenchmarkProfile,
    prefill_tokens: int,
    decode_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total_tokens = prefill_tokens + decode_steps
    shape = (
        profile.target_layer_count,
        1,
        profile.num_key_value_heads,
        total_tokens,
        profile.head_dim,
    )
    keys = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    queries = rng.uniform(
        -1.0,
        1.0,
        size=(profile.target_layer_count, 1, profile.num_attention_heads, decode_steps, profile.head_dim),
    ).astype(np.float32)
    return keys, values, queries


def _run_qwen_decode_runtime_benchmark_impl(
    *,
    profile: QwenDecodeBenchmarkProfile,
    prefill_tokens: int,
    decode_steps: int,
    seed: int,
    decode_strategy: str,
) -> QwenDecodeRuntimeBenchmarkResult:
    if not metal_available():
        raise RuntimeError("Metal toolchain unavailable for benchmark")

    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    if prefill_tokens <= 0:
        raise ValueError("prefill_tokens must be positive")
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if profile.target_layer_count <= 0:
        raise ValueError("Qwen decode benchmark requires at least one target layer")

    keys_np, values_np, queries_np = _build_decode_inputs(
        profile=profile,
        prefill_tokens=prefill_tokens,
        decode_steps=decode_steps,
        seed=seed,
    )
    scale = profile.head_dim ** -0.5

    def _prefill_baseline_cache(cache_factory):
        caches = [cache_factory() for _ in range(profile.target_layer_count)]
        for token_idx in range(prefill_tokens):
            pending: list[object] = []
            for layer_idx in range(profile.target_layer_count):
                key_step = mx.array(keys_np[layer_idx, :, :, token_idx : token_idx + 1, :])
                value_step = mx.array(values_np[layer_idx, :, :, token_idx : token_idx + 1, :])
                fetched_keys, fetched_values = caches[layer_idx].update_and_fetch(key_step, value_step)
                pending.extend(_flatten_for_eval(fetched_keys))
                pending.extend(_flatten_for_eval(fetched_values))
            mx.eval(*pending)
        return caches

    fp_caches = _prefill_baseline_cache(KVCache)
    quant_caches = _prefill_baseline_cache(lambda: QuantizedKVCache(group_size=64, bits=profile.bit_width))

    torque_cache = TorqueKVCacheMLX(
        config=TorqueConfig(
            bit_width=profile.bit_width,
            head_dim=profile.head_dim,
            num_layers=profile.target_layer_count,
            kv_heads=profile.num_key_value_heads,
            fused_weights=profile.fused_weights,
            rotation_seed=profile.rotation_seed,
        ),
        key_codebook=build_uniform_codebook(profile.bit_width),
        value_codebook=build_uniform_codebook(profile.bit_width),
        initial_capacity=prefill_tokens + decode_steps,
        decode_strategy=decode_strategy,
    )
    for token_idx in range(prefill_tokens):
        torque_cache.append(
            key=keys_np[:, 0, :, token_idx, :],
            value=values_np[:, 0, :, token_idx, :],
        )

    def _time_baseline(caches) -> tuple[float, float, float, np.ndarray]:
        final_outputs = np.zeros(
            (profile.target_layer_count, profile.num_attention_heads, profile.head_dim),
            dtype=np.float32,
        )
        started = perf_counter()
        update_elapsed = 0.0
        attention_elapsed = 0.0
        for step_idx in range(decode_steps):
            update_pending: list[object] = []
            step_state = []
            update_started = perf_counter()
            for layer_idx in range(profile.target_layer_count):
                key_step = mx.array(keys_np[layer_idx, :, :, prefill_tokens + step_idx : prefill_tokens + step_idx + 1, :])
                value_step = mx.array(values_np[layer_idx, :, :, prefill_tokens + step_idx : prefill_tokens + step_idx + 1, :])
                query_step = mx.array(queries_np[layer_idx, :, :, step_idx : step_idx + 1, :])
                fetched_keys, fetched_values = caches[layer_idx].update_and_fetch(key_step, value_step)
                step_state.append((query_step, fetched_keys, fetched_values, caches[layer_idx]))
                update_pending.extend(_flatten_for_eval(fetched_keys))
                update_pending.extend(_flatten_for_eval(fetched_values))
            mx.eval(*update_pending)
            update_elapsed += perf_counter() - update_started

            attention_started = perf_counter()
            outputs = []
            for query_step, fetched_keys, fetched_values, cache in step_state:
                outputs.append(
                    scaled_dot_product_attention(
                        query_step,
                        fetched_keys,
                        fetched_values,
                        cache,
                        scale=scale,
                        mask=None,
                    )
                )
            mx.eval(*outputs)
            attention_elapsed += perf_counter() - attention_started
            for layer_idx, out in enumerate(outputs):
                final_outputs[layer_idx] = np.array(out)[0, :, 0, :]
        total_elapsed = perf_counter() - started
        return total_elapsed, update_elapsed, attention_elapsed, final_outputs

    fp_elapsed, fp_update_elapsed, fp_attention_elapsed, fp_out = _time_baseline(fp_caches)
    quant_elapsed, quant_update_elapsed, quant_attention_elapsed, quant_out = _time_baseline(quant_caches)

    started = perf_counter()
    torque_append_elapsed = 0.0
    torque_kernel_elapsed = 0.0
    torque_out = np.zeros_like(fp_out)
    for step_idx in range(decode_steps):
        append_started = perf_counter()
        torque_cache.append(
            key=keys_np[:, 0, :, prefill_tokens + step_idx, :],
            value=values_np[:, 0, :, prefill_tokens + step_idx, :],
        )
        torque_append_elapsed += perf_counter() - append_started

        kernel_started = perf_counter()
        torque_out_device = torque_cache.decode_mlx(
            query=queries_np[:, 0, :, step_idx, :],
            return_numpy=False,
        )
        mx.eval(torque_out_device)
        torque_kernel_elapsed += perf_counter() - kernel_started
        torque_out = np.array(torque_out_device)
    torque_elapsed = perf_counter() - started

    return QwenDecodeRuntimeBenchmarkResult(
        profile=profile,
        prefill_tokens=prefill_tokens,
        decode_steps=decode_steps,
        seed=seed,
        torque_decode_strategy=decode_strategy,
        fp16_update_seconds=fp_update_elapsed,
        fp16_attention_seconds=fp_attention_elapsed,
        fp16_decode_seconds=fp_elapsed,
        mlx_lm_quantized_update_seconds=quant_update_elapsed,
        mlx_lm_quantized_attention_seconds=quant_attention_elapsed,
        mlx_lm_quantized_decode_seconds=quant_elapsed,
        torque_append_seconds=torque_append_elapsed,
        torque_kernel_seconds=torque_kernel_elapsed,
        torque_decode_seconds=torque_elapsed,
        max_abs_error_quantized_vs_fp16=float(np.max(np.abs(quant_out - fp_out))),
        max_abs_error_torque_vs_fp16=float(np.max(np.abs(torque_out - fp_out))),
    )


def run_qwen_decode_runtime_benchmark(
    *,
    model_dir: str | Path,
    prefill_tokens: int,
    decode_steps: int,
    seed: int,
    bit_width: int | None = None,
    rotation_seed: int = 0,
    fused_weights: bool = False,
    decode_strategy: str = "split_batched",
    runner: Callable[..., QwenDecodeRuntimeBenchmarkResult] | None = None,
) -> QwenDecodeRuntimeBenchmarkResult:
    if decode_strategy not in SUPPORTED_DECODE_STRATEGIES:
        raise ValueError(
            "decode_strategy must be one of "
            + ", ".join(SUPPORTED_DECODE_STRATEGIES),
        )
    profile = load_qwen_decode_benchmark_profile(
        model_dir,
        bit_width=bit_width,
        rotation_seed=rotation_seed,
        fused_weights=fused_weights,
    )
    runtime_runner = runner or _run_qwen_decode_runtime_benchmark_impl
    return runtime_runner(
        profile=profile,
        prefill_tokens=prefill_tokens,
        decode_steps=decode_steps,
        seed=seed,
        decode_strategy=decode_strategy,
    )
