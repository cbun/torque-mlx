from __future__ import annotations

from dataclasses import dataclass, field
import glob
from pathlib import Path
from time import perf_counter
from types import MethodType, SimpleNamespace
from typing import Any, Mapping

import numpy as np

from torque_mlx.cache_mlx import TorqueKVCacheMLX
from torque_mlx.config import TorqueConfig
from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, load_qwen_model_manifest
from torque_mlx.layout import PackedKVLayout
from torque_mlx.rotation import RotationSpec, inverse_structured_rotation_mlx

DEFAULT_QWEN_DECODE_TAIL_CAPACITY = 8
DEFAULT_QWEN_LARGE_MODEL_DECODE_TAIL_CAPACITY = 0
QWEN_SMALL_MODEL_HIDDEN_SIZE_THRESHOLD = 2048


def resolve_qwen_decode_tail_capacity(*, hidden_size: int | None, requested: int | None) -> int:
    if requested is not None:
        return requested
    if hidden_size is None:
        return DEFAULT_QWEN_DECODE_TAIL_CAPACITY
    if hidden_size < QWEN_SMALL_MODEL_HIDDEN_SIZE_THRESHOLD:
        return DEFAULT_QWEN_DECODE_TAIL_CAPACITY
    return DEFAULT_QWEN_LARGE_MODEL_DECODE_TAIL_CAPACITY


def _qwen_hidden_size_from_config(config: Mapping[str, object]) -> int | None:
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return None
    hidden_size = text_config.get("hidden_size")
    return int(hidden_size) if hidden_size is not None else None


def _qwen_full_attention_layer_count(config: Mapping[str, object]) -> int:
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return 0
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        return 0
    return sum(1 for layer_type in layer_types if layer_type == "full_attention")


def _estimate_qwen_full_attention_kv_bytes(
    *,
    config: Mapping[str, object],
    manifest,
    kv_cache_tokens: int,
) -> tuple[int, int | None, int | None, int]:
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        return 0, None, None, 0

    head_dim = int(text_config["head_dim"])
    kv_heads = int(text_config["num_key_value_heads"])
    full_attention_layers = (
        len(manifest.converted_layer_indices)
        if manifest is not None
        else _qwen_full_attention_layer_count(config)
    )
    fp16_bytes = kv_cache_tokens * full_attention_layers * kv_heads * head_dim * 2 * 2
    if manifest is None:
        return fp16_bytes, None, None, full_attention_layers

    layout = PackedKVLayout(
        bit_width=manifest.runtime_config.bit_width,
        head_dim=manifest.runtime_config.head_dim,
    )
    packed_bytes = (
        kv_cache_tokens
        * full_attention_layers
        * manifest.runtime_config.kv_heads
        * layout.kv_bytes_per_token_per_head
    )
    return fp16_bytes, packed_bytes, fp16_bytes - packed_bytes, full_attention_layers


@dataclass(frozen=True, slots=True)
class QwenMLXRuntimeProfile:
    dense_prefill_seconds: float = 0.0
    dense_prefill_calls: int = 0
    dense_prefill_tokens: int = 0
    converted_attention_seconds: float = 0.0
    converted_attention_calls: int = 0
    converted_attention_tokens: int = 0
    passthrough_attention_seconds: float = 0.0
    passthrough_attention_calls: int = 0
    passthrough_attention_tokens: int = 0
    linear_layer_seconds: float = 0.0
    linear_layer_calls: int = 0
    linear_layer_tokens: int = 0
    prompt_append_seconds: float = 0.0
    prompt_append_calls: int = 0
    prompt_append_tokens: int = 0
    decode_append_seconds: float = 0.0
    decode_append_calls: int = 0
    decode_append_tokens: int = 0
    torque_append_seconds: float = 0.0
    torque_append_calls: int = 0
    torque_append_tokens: int = 0
    torque_decode_seconds: float = 0.0
    torque_decode_calls: int = 0
    torque_decode_tokens: int = 0
    torque_score_seconds: float = 0.0
    torque_softmax_seconds: float = 0.0
    torque_value_seconds: float = 0.0
    torque_tail_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "dense_prefill_seconds": self.dense_prefill_seconds,
            "dense_prefill_calls": self.dense_prefill_calls,
            "dense_prefill_tokens": self.dense_prefill_tokens,
            "converted_attention_seconds": self.converted_attention_seconds,
            "converted_attention_calls": self.converted_attention_calls,
            "converted_attention_tokens": self.converted_attention_tokens,
            "passthrough_attention_seconds": self.passthrough_attention_seconds,
            "passthrough_attention_calls": self.passthrough_attention_calls,
            "passthrough_attention_tokens": self.passthrough_attention_tokens,
            "linear_layer_seconds": self.linear_layer_seconds,
            "linear_layer_calls": self.linear_layer_calls,
            "linear_layer_tokens": self.linear_layer_tokens,
            "prompt_append_seconds": self.prompt_append_seconds,
            "prompt_append_calls": self.prompt_append_calls,
            "prompt_append_tokens": self.prompt_append_tokens,
            "decode_append_seconds": self.decode_append_seconds,
            "decode_append_calls": self.decode_append_calls,
            "decode_append_tokens": self.decode_append_tokens,
            "torque_append_seconds": self.torque_append_seconds,
            "torque_append_calls": self.torque_append_calls,
            "torque_append_tokens": self.torque_append_tokens,
            "torque_decode_seconds": self.torque_decode_seconds,
            "torque_decode_calls": self.torque_decode_calls,
            "torque_decode_tokens": self.torque_decode_tokens,
            "torque_score_seconds": self.torque_score_seconds,
            "torque_softmax_seconds": self.torque_softmax_seconds,
            "torque_value_seconds": self.torque_value_seconds,
            "torque_tail_seconds": self.torque_tail_seconds,
        }


@dataclass(frozen=True, slots=True)
class QwenMLXGenerationResult:
    model_dir: str
    prompt: str
    max_tokens: int
    prefill_step_size: int
    decode_tail_capacity: int
    ignore_eos: bool
    generated_text: str
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens: int
    generation_tokens_per_second: float
    peak_memory_gb: float
    kv_cache_tokens: int
    full_attention_layer_count: int
    full_attention_kv_fp16_bytes_estimate: int
    full_attention_kv_packed_bytes_estimate: int | None
    full_attention_kv_bytes_saved_estimate: int | None
    is_torque_converted: bool
    artifact_layout: str | None
    converted_layer_indices: tuple[int, ...]
    prompt_seconds_estimate: float | None = None
    generation_seconds_estimate: float | None = None
    runtime_profile: QwenMLXRuntimeProfile | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "family": "qwen",
            "benchmark": "qwen_mlx_generation",
            "model_dir": self.model_dir,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "prefill_step_size": self.prefill_step_size,
            "decode_tail_capacity": self.decode_tail_capacity,
            "ignore_eos": self.ignore_eos,
            "generated_text": self.generated_text,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "generation_tokens": self.generation_tokens,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "peak_memory_gb": self.peak_memory_gb,
            "kv_cache_tokens": self.kv_cache_tokens,
            "full_attention_layer_count": self.full_attention_layer_count,
            "full_attention_kv_fp16_bytes_estimate": self.full_attention_kv_fp16_bytes_estimate,
            "full_attention_kv_packed_bytes_estimate": self.full_attention_kv_packed_bytes_estimate,
            "full_attention_kv_bytes_saved_estimate": self.full_attention_kv_bytes_saved_estimate,
            "is_torque_converted": self.is_torque_converted,
            "artifact_layout": self.artifact_layout,
            "converted_layer_indices": list(self.converted_layer_indices),
            "prompt_seconds_estimate": self.prompt_seconds_estimate,
            "generation_seconds_estimate": self.generation_seconds_estimate,
        }
        if self.runtime_profile is not None:
            payload["runtime_profile"] = self.runtime_profile.to_dict()
        return payload


@dataclass(slots=True)
class _QwenMLXRuntimeProfiler:
    dense_prefill_seconds: float = 0.0
    dense_prefill_calls: int = 0
    dense_prefill_tokens: int = 0
    converted_attention_seconds: float = 0.0
    converted_attention_calls: int = 0
    converted_attention_tokens: int = 0
    passthrough_attention_seconds: float = 0.0
    passthrough_attention_calls: int = 0
    passthrough_attention_tokens: int = 0
    linear_layer_seconds: float = 0.0
    linear_layer_calls: int = 0
    linear_layer_tokens: int = 0
    prompt_append_seconds: float = 0.0
    prompt_append_calls: int = 0
    prompt_append_tokens: int = 0
    decode_append_seconds: float = 0.0
    decode_append_calls: int = 0
    decode_append_tokens: int = 0
    torque_append_seconds: float = 0.0
    torque_append_calls: int = 0
    torque_append_tokens: int = 0
    torque_decode_seconds: float = 0.0
    torque_decode_calls: int = 0
    torque_decode_tokens: int = 0
    torque_score_seconds: float = 0.0
    torque_softmax_seconds: float = 0.0
    torque_value_seconds: float = 0.0
    torque_tail_seconds: float = 0.0

    def record_dense_prefill(self, *, seconds: float, tokens: int) -> None:
        self.dense_prefill_seconds += seconds
        self.dense_prefill_calls += 1
        self.dense_prefill_tokens += tokens

    def record_converted_attention(self, *, seconds: float, tokens: int) -> None:
        self.converted_attention_seconds += seconds
        self.converted_attention_calls += 1
        self.converted_attention_tokens += tokens

    def record_passthrough_attention(self, *, seconds: float, tokens: int) -> None:
        self.passthrough_attention_seconds += seconds
        self.passthrough_attention_calls += 1
        self.passthrough_attention_tokens += tokens

    def record_linear_layer(self, *, seconds: float, tokens: int) -> None:
        self.linear_layer_seconds += seconds
        self.linear_layer_calls += 1
        self.linear_layer_tokens += tokens

    def record_prompt_append(self, *, seconds: float, tokens: int) -> None:
        self.prompt_append_seconds += seconds
        self.prompt_append_calls += 1
        self.prompt_append_tokens += tokens
        self.torque_append_seconds += seconds
        self.torque_append_calls += 1
        self.torque_append_tokens += tokens

    def record_decode_append(self, *, seconds: float, tokens: int) -> None:
        self.decode_append_seconds += seconds
        self.decode_append_calls += 1
        self.decode_append_tokens += tokens
        self.torque_append_seconds += seconds
        self.torque_append_calls += 1
        self.torque_append_tokens += tokens

    def record_torque_decode(self, *, seconds: float, tokens: int, components: Mapping[str, float] | None = None) -> None:
        self.torque_decode_seconds += seconds
        self.torque_decode_calls += 1
        self.torque_decode_tokens += tokens
        if components is not None:
            self.torque_score_seconds += float(components.get("packed_score_seconds", 0.0))
            self.torque_softmax_seconds += float(components.get("softmax_seconds", 0.0))
            self.torque_value_seconds += float(components.get("packed_value_seconds", 0.0))
            self.torque_tail_seconds += float(components.get("tail_seconds", 0.0))

    def freeze(self) -> QwenMLXRuntimeProfile:
        return QwenMLXRuntimeProfile(
            dense_prefill_seconds=self.dense_prefill_seconds,
            dense_prefill_calls=self.dense_prefill_calls,
            dense_prefill_tokens=self.dense_prefill_tokens,
            converted_attention_seconds=self.converted_attention_seconds,
            converted_attention_calls=self.converted_attention_calls,
            converted_attention_tokens=self.converted_attention_tokens,
            passthrough_attention_seconds=self.passthrough_attention_seconds,
            passthrough_attention_calls=self.passthrough_attention_calls,
            passthrough_attention_tokens=self.passthrough_attention_tokens,
            linear_layer_seconds=self.linear_layer_seconds,
            linear_layer_calls=self.linear_layer_calls,
            linear_layer_tokens=self.linear_layer_tokens,
            prompt_append_seconds=self.prompt_append_seconds,
            prompt_append_calls=self.prompt_append_calls,
            prompt_append_tokens=self.prompt_append_tokens,
            decode_append_seconds=self.decode_append_seconds,
            decode_append_calls=self.decode_append_calls,
            decode_append_tokens=self.decode_append_tokens,
            torque_append_seconds=self.torque_append_seconds,
            torque_append_calls=self.torque_append_calls,
            torque_append_tokens=self.torque_append_tokens,
            torque_decode_seconds=self.torque_decode_seconds,
            torque_decode_calls=self.torque_decode_calls,
            torque_decode_tokens=self.torque_decode_tokens,
            torque_score_seconds=self.torque_score_seconds,
            torque_softmax_seconds=self.torque_softmax_seconds,
            torque_value_seconds=self.torque_value_seconds,
            torque_tail_seconds=self.torque_tail_seconds,
        )


def _normalize_qwen3_5_text_config(text_config: Mapping[str, object]) -> dict[str, object]:
    payload = dict(text_config)
    rope_parameters = dict(payload.get("rope_parameters") or {})
    payload["model_type"] = "qwen3_next"
    payload["rope_theta"] = payload.get("rope_theta") or rope_parameters.get("rope_theta", 10_000_000.0)
    payload["partial_rotary_factor"] = payload.get("partial_rotary_factor") or rope_parameters.get(
        "partial_rotary_factor",
        0.25,
    )
    payload.setdefault("num_experts", 0)
    payload.setdefault("num_experts_per_tok", 0)
    payload.setdefault("decoder_sparse_step", 1)
    payload.setdefault("shared_expert_intermediate_size", 0)
    payload.setdefault("moe_intermediate_size", 0)
    payload.setdefault("norm_topk_prob", False)
    payload.setdefault("attention_bias", False)
    payload.setdefault("tie_word_embeddings", True)
    payload.setdefault("full_attention_interval", 4)
    payload.setdefault("mlp_only_layers", [])
    return payload


def _load_mlx_qwen_runtime():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import KVCache, MambaCache
        from mlx_lm.models.qwen3_next import ModelArgs as Qwen3NextArgs
        from mlx_lm.models.qwen3_next import Qwen3NextModel
        from mlx_lm.utils import load_config, load_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "MLX Qwen generation requires the optional mlx and mlx-lm runtime dependencies.",
        ) from exc

    return SimpleNamespace(
        mx=mx,
        nn=nn,
        tree_flatten=tree_flatten,
        KVCache=KVCache,
        MambaCache=MambaCache,
        Qwen3NextArgs=Qwen3NextArgs,
        Qwen3NextModel=Qwen3NextModel,
        load_config=load_config,
        load_tokenizer=load_tokenizer,
    )


def _norm_suffixes() -> tuple[str, ...]:
    return (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "model.language_model.norm.weight",
    )


def _sanitize_qwen3_5_weights(
    weights: Mapping[str, Any],
    *,
    mx,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    handled: set[str] = set()

    for key, value in weights.items():
        if key in handled:
            continue
        if key.startswith("model.visual.") or key.startswith("vision_tower.") or key.startswith("mtp."):
            continue
        if key == "lm_head.weight" and tie_word_embeddings:
            continue

        if key.endswith(".linear_attn.in_proj_qkv.weight"):
            prefix = key.removesuffix("in_proj_qkv.weight")
            z_key = prefix + "in_proj_z.weight"
            if z_key not in weights:
                raise KeyError(f"Missing paired Qwen3.5 linear attention tensor: {z_key}")
            sanitized[prefix + "in_proj_qkvz.weight"] = mx.concatenate([value, weights[z_key]], axis=0)
            handled.add(z_key)
            continue

        if key.endswith(".linear_attn.in_proj_b.weight"):
            prefix = key.removesuffix("in_proj_b.weight")
            a_key = prefix + "in_proj_a.weight"
            if a_key not in weights:
                raise KeyError(f"Missing paired Qwen3.5 linear attention tensor: {a_key}")
            sanitized[prefix + "in_proj_ba.weight"] = mx.concatenate([value, weights[a_key]], axis=0)
            handled.add(a_key)
            continue

        if key.endswith(".linear_attn.in_proj_z.weight") or key.endswith(".linear_attn.in_proj_a.weight"):
            continue

        transformed = value
        if "conv1d.weight" in key and len(value.shape) == 3 and value.shape[-1] != 1:
            transformed = value.moveaxis(2, 1)
        if any(key.endswith(suffix) for suffix in _norm_suffixes()) and len(value.shape) == 1:
            transformed = transformed + 1.0
        sanitized[key] = transformed

    return sanitized


def _resolve_qwen_weight_target(model, weight_name: str):
    target = model
    for part in weight_name.split("."):
        if part.isdigit():
            target = target[int(part)]
            continue
        target = getattr(target, part)
    return target


def _cast_qwen_weights_to_model_dtypes(model, weights: Mapping[str, Any], *, mx) -> list[tuple[str, Any]]:
    cast_weights: list[tuple[str, Any]] = []
    for name, value in weights.items():
        target = _resolve_qwen_weight_target(model, name)
        target_dtype = getattr(target, "dtype", None)
        array = value if target_dtype is None or value.dtype == target_dtype else value.astype(target_dtype)
        cast_weights.append((name, array))
    return cast_weights


def _build_qwen3_5_mlx_model(config: Mapping[str, object]):
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx
    nn = runtime.nn
    Qwen3NextArgs = runtime.Qwen3NextArgs
    Qwen3NextModel = runtime.Qwen3NextModel
    MambaCache = runtime.MambaCache
    KVCache = runtime.KVCache

    from dataclasses import dataclass

    text_config = _normalize_qwen3_5_text_config(dict(config["text_config"]))
    text_args = Qwen3NextArgs.from_dict(text_config)

    @dataclass
    class Qwen35Args:
        model_type: str
        text_config: dict

        @classmethod
        def from_dict(cls, params):
            return cls(
                model_type=str(params["model_type"]),
                text_config=dict(params["text_config"]),
            )

    class Qwen35Backbone(nn.Module):
        def __init__(self, args: Qwen35Args):
            super().__init__()
            self.args = Qwen3NextArgs.from_dict(_normalize_qwen3_5_text_config(args.text_config))
            self.language_model = Qwen3NextModel(self.args)
            self.runtime_profiler = None
            self.converted_layer_indices = frozenset()

        def __call__(self, inputs, cache=None, input_embeddings=None):
            if input_embeddings is not None:
                hidden_states = input_embeddings
            else:
                hidden_states = self.language_model.embed_tokens(inputs)

            if cache is None:
                cache = [None] * len(self.language_model.layers)

            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            fa_idx = self.args.full_attention_interval - 1
            fa_mask = create_attention_mask(hidden_states, cache[fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[0])

            for layer_idx, (layer, layer_cache) in enumerate(zip(self.language_model.layers, cache)):
                mask = ssm_mask if layer.is_linear else fa_mask
                if self.runtime_profiler is not None:
                    layer_start = perf_counter()
                    hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                    mx.eval(hidden_states)
                    elapsed = perf_counter() - layer_start
                    tokens = int(hidden_states.shape[1])
                    if layer.is_linear:
                        self.runtime_profiler.record_linear_layer(seconds=elapsed, tokens=tokens)
                    elif layer_idx in self.converted_layer_indices:
                        self.runtime_profiler.record_converted_attention(seconds=elapsed, tokens=tokens)
                    else:
                        self.runtime_profiler.record_passthrough_attention(seconds=elapsed, tokens=tokens)
                    continue
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            return self.language_model.norm(hidden_states)

    class Qwen35ForConditionalGeneration(nn.Module):
        def __init__(self, args: Qwen35Args):
            super().__init__()
            self.args = args
            self.model = Qwen35Backbone(args)
            if not self.model.args.tie_word_embeddings:
                self.lm_head = nn.Linear(
                    self.model.args.hidden_size,
                    self.model.args.vocab_size,
                    bias=False,
                )

        def __call__(self, inputs, cache=None, input_embeddings=None):
            out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
            if self.model.args.tie_word_embeddings:
                out = self.model.language_model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
            return out

        @property
        def layers(self):
            return self.model.language_model.layers

        def make_cache(self):
            return [MambaCache() if layer.is_linear else KVCache() for layer in self.layers]

        def sanitize(self, weights):
            return _sanitize_qwen3_5_weights(
                weights,
                mx=mx,
                tie_word_embeddings=self.model.args.tie_word_embeddings,
            )

    return Qwen35ForConditionalGeneration, Qwen35Args


def _load_qwen3_5_mlx_model(
    *,
    model_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
    lazy: bool = False,
):
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx
    load_config = runtime.load_config
    load_tokenizer = runtime.load_tokenizer

    model_path = Path(model_dir)
    config = load_config(model_path)
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Unsupported MLX Qwen model_type: {config.get('model_type')}")

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: dict[str, Any] = {}
    for weight_file in weight_files:
        weights.update(mx.load(weight_file))

    model_class, model_args_class = _build_qwen3_5_mlx_model(config)
    model = model_class(model_args_class.from_dict(config))
    weights = model.sanitize(weights)
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())

    tokenizer_path = Path(tokenizer_dir) if tokenizer_dir is not None else model_path
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer, config


def _load_qwen_mlx_delta_overrides(artifact_dir: str | Path, manifest) -> dict[str, np.ndarray]:
    artifact_path = Path(artifact_dir)
    if manifest.delta_weights_file is None:
        raise ValueError("delta_weights_file missing from delta_npz manifest")
    delta_path = artifact_path / manifest.delta_weights_file
    if not delta_path.exists():
        raise FileNotFoundError(f"Qwen delta weights file not found: {delta_path}")
    data = np.load(delta_path)
    return {str(name): np.asarray(data[name], dtype=np.float32) for name in data.files}


def _apply_qwen_mlx_delta_overrides(model, artifact_dir: str | Path, manifest) -> None:
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx

    overrides = _load_qwen_mlx_delta_overrides(artifact_dir, manifest)
    weights = _cast_qwen_weights_to_model_dtypes(
        model,
        {name: mx.array(values) for name, values in overrides.items()},
        mx=mx,
    )
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())


def _restore_stacked_input_projection_mlx(weight, *, head_dim: int, num_blocks: int, rotation_matrix):
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx

    reshaped = weight.astype(mx.float32).reshape(weight.shape[0], num_blocks, head_dim)
    flat = reshaped.reshape(-1, head_dim)
    restored = flat @ rotation_matrix.astype(mx.float32)
    restored = restored.reshape(weight.shape[0], num_blocks, head_dim)
    return restored.reshape(weight.shape)


def _runtime_unrotate_attention_output_mlx(attn_output, *, signs_left, signs_right):
    head_dim = int(attn_output.shape[-1])
    flat = attn_output.reshape(-1, head_dim)
    unrotated = inverse_structured_rotation_mlx(
        flat,
        signs_left=signs_left,
        signs_right=signs_right,
    )
    return unrotated.reshape(attn_output.shape)


@dataclass(slots=True)
class QwenTorqueFullAttentionCacheMLX:
    torque_cache: TorqueKVCacheMLX
    dense_cache: Any | None = field(default=None)
    profiler: _QwenMLXRuntimeProfiler | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        runtime = _load_mlx_qwen_runtime()
        self.dense_cache = runtime.KVCache()

    @property
    def offset(self) -> int:
        return self.torque_cache.offset

    @property
    def state(self):
        state = [self.torque_cache._key_codes, self.torque_cache._value_codes]
        if self.dense_cache is not None and not self.dense_cache.empty():
            state.extend(list(self.dense_cache.state))
        return state

    def make_mask(self, N: int, *, window_size: int | None = None, return_array: bool = False):
        return self.torque_cache.make_mask(N, window_size=window_size, return_array=return_array)

    def update_dense_and_fetch(self, keys, values):
        runtime = _load_mlx_qwen_runtime()
        if self.dense_cache is None:
            if self.offset > 0:
                raise RuntimeError(
                    "Torque MLX cache cannot process a multi-token chunk after switching to decode mode",
                )
            self.dense_cache = runtime.KVCache()
        return self.dense_cache.update_and_fetch(keys, values)

    def _append_tokens(self, keys, values, *, stage: str) -> None:
        runtime = _load_mlx_qwen_runtime()
        if int(keys.shape[0]) != 1:
            raise ValueError("Torque MLX Qwen cache currently supports batch size 1")
        start = perf_counter() if self.profiler is not None else None
        self.torque_cache.append_many_mlx(
            key=keys[0],
            value=values[0],
        )
        if self.profiler is not None and start is not None:
            runtime.mx.eval(self.torque_cache._key_codes, self.torque_cache._value_codes)
            elapsed = perf_counter() - start
            tokens = int(keys.shape[2])
            if stage == "prompt":
                self.profiler.record_prompt_append(seconds=elapsed, tokens=tokens)
            elif stage == "decode":
                self.profiler.record_decode_append(seconds=elapsed, tokens=tokens)
            else:
                raise ValueError(f"Unknown append profiling stage: {stage}")

    def append_prompt_tokens(self, keys, values) -> None:
        self._append_tokens(keys, values, stage="prompt")

    def append_decode_tokens(self, keys, values) -> None:
        self._append_tokens(keys, values, stage="decode")

    def decode_token(self, queries):
        runtime = _load_mlx_qwen_runtime()
        mx = runtime.mx

        if int(queries.shape[0]) != 1 or int(queries.shape[2]) != 1:
            raise ValueError("Torque MLX Qwen decode currently supports batch size 1 and q_len=1")
        self.dense_cache = None
        start = perf_counter() if self.profiler is not None else None
        output = self.torque_cache.decode_mlx(query=queries[0, :, 0, :], return_numpy=False)
        if self.profiler is not None and start is not None:
            mx.eval(output)
            self.profiler.record_torque_decode(
                seconds=perf_counter() - start,
                tokens=int(queries.shape[2]),
                components=self.torque_cache.last_decode_profile,
            )
        return mx.reshape(output, (1, int(queries.shape[1]), 1, int(queries.shape[3])))

    def decode_token_with_current(self, queries, keys, values):
        runtime = _load_mlx_qwen_runtime()
        mx = runtime.mx

        if int(queries.shape[0]) != 1 or int(queries.shape[2]) != 1:
            raise ValueError("Torque MLX Qwen decode currently supports batch size 1 and q_len=1")
        if int(keys.shape[0]) != 1 or int(keys.shape[2]) != 1 or int(values.shape[0]) != 1 or int(values.shape[2]) != 1:
            raise ValueError("Torque MLX Qwen decode currently expects single-token key/value updates")
        self.dense_cache = None
        start = perf_counter() if self.profiler is not None else None
        output = self.torque_cache.decode_mlx_with_current(
            query=queries[0, :, 0, :],
            key=keys[0],
            value=values[0],
            return_numpy=False,
        )
        if self.profiler is not None and start is not None:
            mx.eval(output)
            self.profiler.record_torque_decode(
                seconds=perf_counter() - start,
                tokens=int(queries.shape[2]),
                components=self.torque_cache.last_decode_profile,
            )
        return mx.reshape(output, (1, int(queries.shape[1]), 1, int(queries.shape[3])))


def _build_qwen_torque_cache(
    manifest,
    *,
    profiler: _QwenMLXRuntimeProfiler | None = None,
    decode_tail_capacity: int = DEFAULT_QWEN_DECODE_TAIL_CAPACITY,
) -> QwenTorqueFullAttentionCacheMLX:
    config = TorqueConfig(
        bit_width=manifest.runtime_config.bit_width,
        head_dim=manifest.runtime_config.head_dim,
        num_layers=1,
        kv_heads=manifest.runtime_config.kv_heads,
        fused_weights=True,
        rotation_mode=manifest.runtime_config.rotation_mode,
        rotation_seed=manifest.runtime_config.rotation_seed,
    )
    return QwenTorqueFullAttentionCacheMLX(
        torque_cache=TorqueKVCacheMLX(
            config=config,
            key_codebook=manifest.key_codebook,
            value_codebook=manifest.value_codebook,
            decode_tail_capacity=decode_tail_capacity,
            profile_decode_components=profiler is not None,
            rotate_keys_on_append=True,
            rotate_values_on_append=False,
            rotate_queries_on_decode=True,
        ),
        profiler=profiler,
    )


def _patch_vo_only_qwen_attention_mlx(
    layer,
    *,
    rotation_matrix,
    rotation_signs_left,
    rotation_signs_right,
    manifest,
    profiler: _QwenMLXRuntimeProfiler | None = None,
) -> None:
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx
    nn = runtime.nn
    attention = layer.self_attn

    restored_o = _restore_stacked_input_projection_mlx(
        attention.o_proj.weight,
        head_dim=int(manifest.runtime_config.head_dim),
        num_blocks=int(attention.num_attention_heads),
        rotation_matrix=rotation_matrix,
    ).astype(attention.o_proj.weight.dtype)
    attention.o_proj.update({"weight": restored_o})

    class _TorqueAttentionAdapter(nn.Module):
        def __init__(self, base_attn):
            super().__init__()
            self.base_attn = base_attn
            self.rotation_signs_left = rotation_signs_left.astype(mx.float32)
            self.rotation_signs_right = rotation_signs_right.astype(mx.float32)
            self.profiler = profiler

        def __call__(self, x, mask=None, cache=None):
            B, L, _ = x.shape
            if B != 1:
                raise ValueError("Torque MLX Qwen runtime currently supports batch size 1")

            q_proj_output = self.base_attn.q_proj(x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.base_attn.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)

            keys = self.base_attn.k_proj(x)
            values = self.base_attn.v_proj(x)

            queries = self.base_attn.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.base_attn.k_norm(
                keys.reshape(B, L, self.base_attn.num_key_value_heads, -1),
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.base_attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)

            if cache is not None:
                queries = self.base_attn.rope(queries, offset=cache.offset)
                keys = self.base_attn.rope(keys, offset=cache.offset)
            else:
                queries = self.base_attn.rope(queries)
                keys = self.base_attn.rope(keys)

            if cache is not None and isinstance(cache, QwenTorqueFullAttentionCacheMLX):
                if L == 1:
                    output = cache.decode_token_with_current(queries, keys, values)
                    cache.append_decode_tokens(keys, values)
                else:
                    dense_start = perf_counter() if self.profiler is not None else None
                    dense_keys, dense_values = cache.update_dense_and_fetch(keys, values)
                    output = mx.fast.scaled_dot_product_attention(
                        queries,
                        dense_keys,
                        dense_values,
                        scale=self.base_attn.scale,
                        mask=mask,
                    )
                    if self.profiler is not None and dense_start is not None:
                        mx.eval(output)
                        self.profiler.record_dense_prefill(
                            seconds=perf_counter() - dense_start,
                            tokens=int(L),
                        )
                    cache.append_prompt_tokens(keys, values)
            else:
                output = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=self.base_attn.scale,
                    mask=mask,
                )

            output = _runtime_unrotate_attention_output_mlx(
                output,
                signs_left=self.rotation_signs_left,
                signs_right=self.rotation_signs_right,
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            projected = (output * mx.sigmoid(gate)).astype(self.base_attn.o_proj.weight.dtype)
            return self.base_attn.o_proj(projected)

    layer.self_attn = _TorqueAttentionAdapter(attention)


def _patch_qwen_mlx_make_cache(
    model,
    manifest,
    *,
    profiler: _QwenMLXRuntimeProfiler | None = None,
    decode_tail_capacity: int = DEFAULT_QWEN_DECODE_TAIL_CAPACITY,
) -> None:
    runtime = _load_mlx_qwen_runtime()

    converted = set(int(idx) for idx in manifest.converted_layer_indices)

    def _make_cache(self):
        caches = []
        for layer_idx, layer in enumerate(self.layers):
            if getattr(layer, "is_linear", False):
                caches.append(runtime.MambaCache())
                continue
            if layer_idx in converted:
                caches.append(
                    _build_qwen_torque_cache(
                        manifest,
                        profiler=profiler,
                        decode_tail_capacity=decode_tail_capacity,
                    ),
                )
                continue
            caches.append(runtime.KVCache())
        return caches

    model.make_cache = MethodType(_make_cache, model)


def _patch_qwen_mlx_runtime(
    model,
    manifest,
    *,
    profiler: _QwenMLXRuntimeProfiler | None = None,
    decode_tail_capacity: int = DEFAULT_QWEN_DECODE_TAIL_CAPACITY,
) -> None:
    runtime = _load_mlx_qwen_runtime()
    mx = runtime.mx

    rotation = RotationSpec.from_seed(
        head_dim=manifest.runtime_config.head_dim,
        seed=manifest.runtime_config.rotation_seed,
    )
    converted = frozenset(int(idx) for idx in manifest.converted_layer_indices)
    model.model.runtime_profiler = profiler
    model.model.converted_layer_indices = converted
    rotation_matrix = mx.array(rotation.matrix().astype(np.float32))
    rotation_signs_left = mx.array(rotation.signs_left.astype(np.float32))
    rotation_signs_right = mx.array(rotation.signs_right.astype(np.float32))

    _patch_qwen_mlx_make_cache(
        model,
        manifest,
        profiler=profiler,
        decode_tail_capacity=decode_tail_capacity,
    )
    for layer_idx_str, mode in manifest.layer_fusion_modes.items():
        if mode != "vo_only_runtime_qk_rotation":
            raise ValueError(f"Unsupported MLX Qwen torque fusion mode: {mode}")
        layer = model.layers[int(layer_idx_str)]
        _patch_vo_only_qwen_attention_mlx(
            layer,
            rotation_matrix=rotation_matrix,
            rotation_signs_left=rotation_signs_left,
            rotation_signs_right=rotation_signs_right,
            manifest=manifest,
            profiler=profiler,
        )


def load_mlx_qwen_model(
    *,
    model_dir: str | Path,
    lazy: bool = False,
    profile_runtime: bool = False,
    decode_tail_capacity: int | None = None,
):
    model_path = Path(model_dir)
    manifest = load_qwen_model_manifest(model_path) if (model_path / QWEN_MODEL_MANIFEST_FILE).exists() else None
    model_load_path = (
        Path(manifest.source_model_dir)
        if manifest is not None and manifest.artifact_layout == "delta_npz"
        else model_path
    )
    tokenizer_path = model_path if (model_path / "tokenizer.json").exists() else model_load_path

    model, tokenizer, config = _load_qwen3_5_mlx_model(
        model_dir=model_load_path,
        tokenizer_dir=tokenizer_path,
        lazy=lazy,
    )
    resolved_decode_tail_capacity = resolve_qwen_decode_tail_capacity(
        hidden_size=_qwen_hidden_size_from_config(config),
        requested=decode_tail_capacity,
    )
    profiler = _QwenMLXRuntimeProfiler() if (manifest is not None and profile_runtime) else None
    if manifest is not None and manifest.artifact_layout == "delta_npz":
        _apply_qwen_mlx_delta_overrides(model, model_path, manifest)
    if manifest is not None:
        _patch_qwen_mlx_runtime(
            model,
            manifest,
            profiler=profiler,
            decode_tail_capacity=resolved_decode_tail_capacity,
        )
    return model, tokenizer, config, manifest, profiler


def benchmark_qwen_mlx_generation(
    *,
    model_dir: str | Path,
    prompt: str,
    max_tokens: int = 64,
    prefill_step_size: int = 512,
    profile_runtime: bool = False,
    decode_tail_capacity: int | None = None,
    ignore_eos: bool = False,
) -> QwenMLXGenerationResult:
    try:
        import mlx.core as mx
        from mlx_lm import stream_generate
    except ImportError as exc:
        raise RuntimeError(
            "Qwen MLX generation requires the optional mlx and mlx-lm runtime dependencies.",
        ) from exc

    model, tokenizer, config, manifest, profiler = load_mlx_qwen_model(
        model_dir=model_dir,
        profile_runtime=profile_runtime,
        decode_tail_capacity=decode_tail_capacity,
    )
    resolved_decode_tail_capacity = resolve_qwen_decode_tail_capacity(
        hidden_size=_qwen_hidden_size_from_config(config),
        requested=decode_tail_capacity,
    )

    generated = ""
    last_response = None
    original_eos_token_ids = tuple(tokenizer.eos_token_ids)
    if ignore_eos:
        tokenizer.eos_token_ids = ()
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
        ):
            generated += response.text
            last_response = response
    finally:
        if ignore_eos:
            tokenizer.eos_token_ids = original_eos_token_ids
    if last_response is None:
        raise RuntimeError("MLX generation produced no responses")

    mx.clear_cache()
    prompt_seconds_estimate = (
        float(last_response.prompt_tokens) / float(last_response.prompt_tps)
        if float(last_response.prompt_tps) > 0.0
        else None
    )
    generation_seconds_estimate = (
        float(last_response.generation_tokens) / float(last_response.generation_tps)
        if float(last_response.generation_tps) > 0.0
        else None
    )
    kv_cache_tokens = int(last_response.prompt_tokens) + int(last_response.generation_tokens)
    (
        full_attention_kv_fp16_bytes_estimate,
        full_attention_kv_packed_bytes_estimate,
        full_attention_kv_bytes_saved_estimate,
        full_attention_layer_count,
    ) = _estimate_qwen_full_attention_kv_bytes(
        config=config,
        manifest=manifest,
        kv_cache_tokens=kv_cache_tokens,
    )
    return QwenMLXGenerationResult(
        model_dir=str(Path(model_dir).resolve()),
        prompt=prompt,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        decode_tail_capacity=resolved_decode_tail_capacity,
        ignore_eos=ignore_eos,
        generated_text=generated,
        prompt_tokens=int(last_response.prompt_tokens),
        prompt_tokens_per_second=float(last_response.prompt_tps),
        generation_tokens=int(last_response.generation_tokens),
        generation_tokens_per_second=float(last_response.generation_tps),
        peak_memory_gb=float(last_response.peak_memory),
        kv_cache_tokens=kv_cache_tokens,
        full_attention_layer_count=full_attention_layer_count,
        full_attention_kv_fp16_bytes_estimate=full_attention_kv_fp16_bytes_estimate,
        full_attention_kv_packed_bytes_estimate=full_attention_kv_packed_bytes_estimate,
        full_attention_kv_bytes_saved_estimate=full_attention_kv_bytes_saved_estimate,
        is_torque_converted=manifest is not None,
        artifact_layout=manifest.artifact_layout if manifest is not None else None,
        converted_layer_indices=tuple(manifest.converted_layer_indices) if manifest is not None else (),
        prompt_seconds_estimate=prompt_seconds_estimate,
        generation_seconds_estimate=generation_seconds_estimate,
        runtime_profile=profiler.freeze() if profiler is not None else None,
    )
