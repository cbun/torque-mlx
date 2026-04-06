from __future__ import annotations

from dataclasses import dataclass
from math import exp
from time import perf_counter
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
from typing import Any, Callable

import numpy as np

from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, load_qwen_model_manifest
from torque_mlx.rotation import RotationSpec


def _resolve_torch_device(device: str):
    import torch

    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def _load_transformers_qwen_model(
    *,
    model_dir: str | Path,
    device: str,
    dtype: str,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen text evaluation requires the optional torch/transformers runtime dependencies.",
        ) from exc

    model_path = Path(model_dir)
    manifest = load_qwen_model_manifest(model_path) if (model_path / QWEN_MODEL_MANIFEST_FILE).exists() else None
    load_path = (
        Path(manifest.source_model_dir)
        if manifest is not None and manifest.artifact_layout == "delta_npz"
        else model_path
    )
    resolved_device = _resolve_torch_device(device)
    resolved_dtype = _resolve_torch_dtype(dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(load_path), use_fast=True)
        model_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if resolved_dtype is not None:
            model_kwargs["dtype"] = resolved_dtype
        model = AutoModelForCausalLM.from_pretrained(str(load_path), **model_kwargs)
    except ValueError as exc:
        if "qwen3_5" in str(exc):
            raise RuntimeError(
                "This Transformers build does not support qwen3_5. Install a newer Transformers release "
                "with native qwen3_5 support before running Qwen perplexity evaluation.",
            ) from exc
        raise

    if manifest is not None and manifest.artifact_layout == "delta_npz":
        _apply_qwen_delta_overrides(model, model_path, manifest)

    model.eval()
    model.to(resolved_device)
    if manifest is not None:
        _patch_qwen_torque_runtime(model, manifest)
    return model, tokenizer, resolved_device, getattr(model, "dtype", None)


def _load_qwen_delta_overrides(artifact_dir: str | Path, manifest) -> dict[str, np.ndarray]:
    artifact_path = Path(artifact_dir)
    if manifest.delta_weights_file is None:
        raise ValueError("delta_weights_file missing from delta_npz manifest")
    delta_path = artifact_path / manifest.delta_weights_file
    if not delta_path.exists():
        raise FileNotFoundError(f"Qwen delta weights file not found: {delta_path}")
    data = np.load(delta_path)
    return {
        str(name): np.asarray(data[name], dtype=np.float32)
        for name in data.files
    }


def _resolve_qwen_override_state_dict_key(name: str, state_dict: Mapping[str, Any]) -> str:
    if name in state_dict:
        return name

    candidates = [
        name.replace("model.language_model.", "model.", 1),
        name.replace("language_model.", "model.", 1),
        name.removeprefix("model."),
    ]
    for candidate in candidates:
        if candidate in state_dict:
            return candidate

    suffix = name.split("layers.", 1)
    if len(suffix) == 2:
        suffix_pattern = "layers." + suffix[1]
        matches = [key for key in state_dict if key.endswith(suffix_pattern)]
        if len(matches) == 1:
            return matches[0]

    raise KeyError(f"Override tensor {name} not found in loaded model state_dict")


def _apply_qwen_delta_overrides(model, artifact_dir: str | Path, manifest) -> None:
    import torch

    overrides = _load_qwen_delta_overrides(artifact_dir, manifest)
    state_dict = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    for name, values in overrides.items():
        resolved_name = _resolve_qwen_override_state_dict_key(name, state_dict)
        reference = state_dict[resolved_name]
        loadable[resolved_name] = torch.as_tensor(values, dtype=reference.dtype)

    incompatible = model.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected override tensors while loading delta artifact: {incompatible.unexpected_keys}")


def _restore_stacked_input_projection(weight, *, head_dim: int, num_blocks: int, rotation_matrix):
    import torch

    reshaped = weight.to(dtype=torch.float32).reshape(weight.shape[0], num_blocks, head_dim)
    flat = reshaped.reshape(-1, head_dim)
    restored = flat @ rotation_matrix.to(dtype=torch.float32)
    restored = restored.reshape(weight.shape[0], num_blocks, head_dim)
    return restored.reshape(weight.shape)


def _runtime_unrotate_attention_output(attn_output, *, rotation_matrix):
    head_dim = int(attn_output.shape[-1])
    flat = attn_output.reshape(-1, head_dim)
    unrotated = flat @ rotation_matrix.to(dtype=attn_output.dtype, device=attn_output.device)
    return unrotated.reshape(attn_output.shape)


def _qwen_text_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unsupported Qwen model structure for torque runtime patching")


def _patch_vo_only_qwen_attention(layer, *, rotation_matrix, num_attention_heads: int, head_dim: int) -> None:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

    with torch.no_grad():
        restored_o = _restore_stacked_input_projection(
            layer.o_proj.weight.data,
            head_dim=head_dim,
            num_blocks=num_attention_heads,
            rotation_matrix=rotation_matrix.to(device=layer.o_proj.weight.device),
        )
        layer.o_proj.weight.copy_(restored_o.to(dtype=layer.o_proj.weight.dtype))

    layer._torque_rotation_matrix = rotation_matrix.to(device=layer.o_proj.weight.device, dtype=torch.float32)

    def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.view(*input_shape, self.config.num_attention_heads, self.head_dim)
        attn_output = _runtime_unrotate_attention_output(
            attn_output,
            rotation_matrix=self._torque_rotation_matrix,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    layer.forward = MethodType(_forward, layer)


def _patch_qwen_torque_runtime(model, manifest) -> None:
    rotation = RotationSpec.from_seed(
        head_dim=manifest.runtime_config.head_dim,
        seed=manifest.runtime_config.rotation_seed,
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for Qwen torque runtime patching") from exc

    rotation_matrix = torch.tensor(rotation.matrix(), dtype=torch.float32)
    layers = _qwen_text_layers(model)
    num_attention_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.head_dim)

    for layer_idx_str, mode in manifest.layer_fusion_modes.items():
        if mode == "vo_only_runtime_qk_rotation":
            _patch_vo_only_qwen_attention(
                layers[int(layer_idx_str)].self_attn,
                rotation_matrix=rotation_matrix,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
            )
            continue
        if mode == "full_qkvo":
            continue
        raise ValueError(f"Unsupported Qwen torque fusion mode: {mode}")


def _directory_safetensor_size_bytes(directory: str | Path) -> int:
    root = Path(directory)
    return sum(path.stat().st_size for path in root.glob("model*.safetensors"))


def _chunk_token_ids(
    token_ids,
    *,
    context_length: int,
    stride: int,
):
    total_tokens = int(token_ids.shape[0])
    if total_tokens < 2:
        raise ValueError("Need at least two tokens to compute perplexity")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if stride <= 0:
        raise ValueError("stride must be positive")

    max_targets_per_window = context_length - 1
    target_start = 1
    while target_start < total_tokens:
        score_count = min(stride, max_targets_per_window, total_tokens - target_start)
        score_end = target_start + score_count
        window_end = score_end
        window_start = max(0, window_end - context_length)
        if target_start <= window_start:
            raise ValueError("Invalid score window generated for perplexity evaluation")
        yield SimpleNamespace(
            window_start=window_start,
            window_end=window_end,
            score_start=target_start,
            score_end=score_end,
        )
        target_start = score_end


@dataclass(frozen=True, slots=True)
class QwenTextPerplexityResult:
    model_dir: str
    text_file: str
    context_length: int
    stride: int
    device: str
    torch_dtype: str
    token_count: int
    evaluated_token_count: int
    window_count: int
    average_negative_log_likelihood: float
    perplexity: float
    model_safetensor_bytes: int
    model_safetensor_gib: float
    is_torque_converted: bool
    torque_manifest: dict[str, object] | None
    loader_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    total_seconds: float = 0.0
    evaluated_tokens_per_second: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "qwen",
            "model_dir": self.model_dir,
            "text_file": self.text_file,
            "context_length": self.context_length,
            "stride": self.stride,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "token_count": self.token_count,
            "evaluated_token_count": self.evaluated_token_count,
            "window_count": self.window_count,
            "average_negative_log_likelihood": self.average_negative_log_likelihood,
            "perplexity": self.perplexity,
            "model_safetensor_bytes": self.model_safetensor_bytes,
            "model_safetensor_gib": self.model_safetensor_gib,
            "is_torque_converted": self.is_torque_converted,
            "torque_manifest": self.torque_manifest,
            "loader_seconds": self.loader_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "total_seconds": self.total_seconds,
            "evaluated_tokens_per_second": self.evaluated_tokens_per_second,
        }


@dataclass(frozen=True, slots=True)
class QwenTextBenchmarkCase:
    context_length: int
    stride: int
    max_tokens: int | None
    source: QwenTextPerplexityResult
    torque: QwenTextPerplexityResult

    def to_dict(self) -> dict[str, object]:
        size_bytes_delta = self.torque.model_safetensor_bytes - self.source.model_safetensor_bytes
        size_gib_delta = self.torque.model_safetensor_gib - self.source.model_safetensor_gib
        perplexity_delta = self.torque.perplexity - self.source.perplexity
        avg_nll_delta = self.torque.average_negative_log_likelihood - self.source.average_negative_log_likelihood
        eval_seconds_delta = self.torque.evaluation_seconds - self.source.evaluation_seconds
        total_seconds_delta = self.torque.total_seconds - self.source.total_seconds
        tokens_per_second_delta = self.torque.evaluated_tokens_per_second - self.source.evaluated_tokens_per_second
        return {
            "context_length": self.context_length,
            "stride": self.stride,
            "max_tokens": self.max_tokens,
            "source": self.source.to_dict(),
            "torque": self.torque.to_dict(),
            "delta": {
                "perplexity": perplexity_delta,
                "average_negative_log_likelihood": avg_nll_delta,
                "model_safetensor_bytes": size_bytes_delta,
                "model_safetensor_gib": size_gib_delta,
                "evaluation_seconds": eval_seconds_delta,
                "total_seconds": total_seconds_delta,
                "evaluated_tokens_per_second": tokens_per_second_delta,
            },
        }


@dataclass(frozen=True, slots=True)
class QwenTextBenchmarkComparison:
    source_model_dir: str
    torque_model_dir: str
    text_file: str
    device: str
    torch_dtype: str
    cases: tuple[QwenTextBenchmarkCase, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "qwen",
            "benchmark": "source_vs_torque_text",
            "source_model_dir": self.source_model_dir,
            "torque_model_dir": self.torque_model_dir,
            "text_file": self.text_file,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "cases": [case.to_dict() for case in self.cases],
        }


def _encode_text_tokens(tokenizer, raw_text: str, *, max_tokens: int | None):
    encode_kwargs: dict[str, object] = {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if max_tokens is not None:
        encode_kwargs["truncation"] = True
        encode_kwargs["max_length"] = max_tokens
    try:
        encoded = tokenizer(raw_text, **encode_kwargs)
        return encoded["input_ids"][0]
    except TypeError:
        encoded = tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        if max_tokens is not None:
            token_ids = token_ids[:max_tokens]
        return token_ids


def evaluate_qwen_text_perplexity(
    *,
    model_dir: str | Path,
    text_file: str | Path,
    context_length: int = 2048,
    stride: int | None = None,
    max_tokens: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
    loader: Callable[..., tuple[Any, Any, Any, Any]] | None = None,
) -> QwenTextPerplexityResult:
    import torch
    import torch.nn.functional as F

    model_path = Path(model_dir)
    text_path = Path(text_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    effective_stride = context_length if stride is None else stride
    runtime_loader = loader or _load_transformers_qwen_model
    total_start = perf_counter()
    loader_start = perf_counter()
    model, tokenizer, resolved_device, resolved_dtype = runtime_loader(
        model_dir=model_path,
        device=device,
        dtype=dtype,
    )
    loader_seconds = perf_counter() - loader_start

    raw_text = text_path.read_text(encoding="utf-8")
    if max_tokens is not None and max_tokens < 2:
        raise ValueError("max_tokens must be at least 2 when provided")
    token_ids = _encode_text_tokens(tokenizer, raw_text, max_tokens=max_tokens)

    token_ids = token_ids.to(torch.long)

    total_nll = 0.0
    total_targets = 0
    window_count = 0

    evaluation_start = perf_counter()
    with torch.no_grad():
        for chunk in _chunk_token_ids(
            token_ids,
            context_length=context_length,
            stride=effective_stride,
        ):
            window_count += 1
            input_ids = token_ids[chunk.window_start : chunk.window_end].unsqueeze(0).to(resolved_device)
            logits = model(input_ids).logits[0]

            score_count = chunk.score_end - chunk.score_start
            local_prediction_start = chunk.score_start - chunk.window_start - 1
            local_prediction_end = local_prediction_start + score_count
            if local_prediction_start < 0:
                raise ValueError("Invalid score window generated for perplexity evaluation")

            logits_slice = logits[local_prediction_start:local_prediction_end]
            target_ids = token_ids[chunk.score_start : chunk.score_end].to(resolved_device)
            nll = F.cross_entropy(logits_slice, target_ids, reduction="sum")
            total_nll += float(nll.detach().cpu().item())
            total_targets += int(target_ids.shape[0])
    evaluation_seconds = perf_counter() - evaluation_start

    if total_targets <= 0:
        raise ValueError("Perplexity evaluation produced zero target tokens")

    manifest_path = model_path / QWEN_MODEL_MANIFEST_FILE
    manifest_summary = load_qwen_model_manifest(model_path).summary() if manifest_path.exists() else None
    average_nll = total_nll / total_targets

    if resolved_dtype is None:
        dtype_name = "auto"
    else:
        dtype_name = str(resolved_dtype).replace("torch.", "")

    model_bytes = _directory_safetensor_size_bytes(model_path)
    total_seconds = perf_counter() - total_start
    return QwenTextPerplexityResult(
        model_dir=str(model_path.resolve()),
        text_file=str(text_path.resolve()),
        context_length=context_length,
        stride=effective_stride,
        device=str(resolved_device),
        torch_dtype=dtype_name,
        token_count=int(token_ids.shape[0]),
        evaluated_token_count=total_targets,
        window_count=window_count,
        average_negative_log_likelihood=average_nll,
        perplexity=exp(average_nll),
        model_safetensor_bytes=model_bytes,
        model_safetensor_gib=model_bytes / float(1 << 30),
        is_torque_converted=manifest_summary is not None,
        torque_manifest=manifest_summary,
        loader_seconds=loader_seconds,
        evaluation_seconds=evaluation_seconds,
        total_seconds=total_seconds,
        evaluated_tokens_per_second=total_targets / evaluation_seconds if evaluation_seconds > 0 else 0.0,
    )


def benchmark_qwen_text_models(
    *,
    source_model_dir: str | Path,
    torque_model_dir: str | Path,
    text_file: str | Path,
    context_lengths: list[int] | tuple[int, ...],
    stride: int | None = None,
    max_tokens: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
    loader: Callable[..., tuple[Any, Any, Any, Any]] | None = None,
) -> QwenTextBenchmarkComparison:
    if not context_lengths:
        raise ValueError("At least one context length is required for Qwen text benchmarking")

    cases: list[QwenTextBenchmarkCase] = []
    resolved_dtype = dtype
    resolved_device = device

    for context_length in context_lengths:
        source_result = evaluate_qwen_text_perplexity(
            model_dir=source_model_dir,
            text_file=text_file,
            context_length=context_length,
            stride=stride,
            max_tokens=max_tokens,
            device=device,
            dtype=dtype,
            loader=loader,
        )
        torque_result = evaluate_qwen_text_perplexity(
            model_dir=torque_model_dir,
            text_file=text_file,
            context_length=context_length,
            stride=stride,
            max_tokens=max_tokens,
            device=device,
            dtype=dtype,
            loader=loader,
        )
        cases.append(
            QwenTextBenchmarkCase(
                context_length=context_length,
                stride=source_result.stride,
                max_tokens=max_tokens,
                source=source_result,
                torque=torque_result,
            )
        )
        resolved_dtype = source_result.torch_dtype
        resolved_device = source_result.device

    return QwenTextBenchmarkComparison(
        source_model_dir=str(Path(source_model_dir).resolve()),
        torque_model_dir=str(Path(torque_model_dir).resolve()),
        text_file=str(Path(text_file).resolve()),
        device=resolved_device,
        torch_dtype=resolved_dtype,
        cases=tuple(cases),
    )
