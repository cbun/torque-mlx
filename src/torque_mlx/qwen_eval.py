from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from torque_mlx.families.qwen import QWEN_MODEL_MANIFEST_FILE, load_qwen_model_manifest


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
    resolved_device = _resolve_torch_device(device)
    resolved_dtype = _resolve_torch_dtype(dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        model_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    except ValueError as exc:
        if "qwen3_5" in str(exc):
            raise RuntimeError(
                "This Transformers build does not support qwen3_5. Install a newer Transformers release "
                "with native qwen3_5 support before running Qwen perplexity evaluation.",
            ) from exc
        raise

    model.eval()
    model.to(resolved_device)
    return model, tokenizer, resolved_device, getattr(model, "dtype", None)


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

    for target_start in range(1, total_tokens, stride):
        score_end = min(target_start + stride, total_tokens)
        window_end = score_end
        window_start = max(0, window_end - context_length)
        if score_end <= target_start:
            continue
        yield SimpleNamespace(
            window_start=window_start,
            window_end=window_end,
            score_start=target_start,
            score_end=score_end,
        )


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
        }


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
    model, tokenizer, resolved_device, resolved_dtype = runtime_loader(
        model_dir=model_path,
        device=device,
        dtype=dtype,
    )

    raw_text = text_path.read_text(encoding="utf-8")
    encoded = tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    if max_tokens is not None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 when provided")
        token_ids = token_ids[:max_tokens]

    token_ids = token_ids.to(torch.long)

    total_nll = 0.0
    total_targets = 0
    window_count = 0

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
    )
