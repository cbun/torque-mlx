from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def build_weight_map(model_dir: str | Path) -> dict[str, str]:
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Malformed weight_map in {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {root}")

    weight_map: dict[str, str] = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt") as handle:
            for key in handle.keys():
                weight_map[str(key)] = file_path.name
    return weight_map


def find_tensor_key_by_suffix(weight_map: Mapping[str, str], suffix: str) -> str:
    matches = [key for key in weight_map if key.endswith(suffix)]
    if not matches:
        raise KeyError(f"Could not find tensor ending with {suffix}")
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one tensor ending with {suffix}, found {matches}")
    return matches[0]


def load_tensor(model_dir: str | Path, tensor_name: str, *, weight_map: Mapping[str, str]) -> np.ndarray:
    import torch

    root = Path(model_dir)
    filename = weight_map[tensor_name]
    path = root / filename
    with safe_open(str(path), framework="pt") as handle:
        tensor = handle.get_tensor(tensor_name)
        if not isinstance(tensor, torch.Tensor):
            return np.asarray(tensor)
        return tensor.detach().to(dtype=torch.float32).cpu().numpy()

def load_file_tensors(model_dir: str | Path, filename: str) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(model_dir)
    path = root / filename
    tensors: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            tensors[str(key)] = handle.get_tensor(key)
        meta = handle.metadata()
        if meta:
            metadata = {str(key): str(value) for key, value in meta.items()}
    return tensors, metadata


def cast_like(values: np.ndarray, reference):
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype)
    try:
        return np.asarray(values, dtype=reference.dtype)
    except TypeError:
        return np.asarray(values)


def materialize_merged_snapshot(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    overrides: Mapping[str, np.ndarray],
    weight_map: Mapping[str, str],
    force: bool = False,
) -> Path:
    source_root = Path(source_dir)
    target_root = Path(output_dir)

    if target_root.exists() and any(target_root.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory {target_root}; pass force=True to replace it",
        )
    if target_root.exists() and force:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    affected_files = {weight_map[key] for key in overrides}
    for source_path in source_root.iterdir():
        if source_path.name in affected_files:
            tensors, metadata = load_file_tensors(source_root, source_path.name)
            for tensor_name, override in overrides.items():
                if weight_map[tensor_name] == source_path.name:
                    tensors[tensor_name] = cast_like(override, tensors[tensor_name])
            first_tensor = next(iter(tensors.values()), None)
            try:
                import torch
                from safetensors.torch import save_file as save_torch_file
            except ImportError:
                torch = None
                save_torch_file = None
            if torch is not None and isinstance(first_tensor, torch.Tensor):
                save_torch_file(tensors, str(target_root / source_path.name), metadata=metadata or None)
            else:
                save_file(tensors, str(target_root / source_path.name), metadata=metadata or None)
            continue
        if source_path.is_dir():
            shutil.copytree(source_path, target_root / source_path.name, dirs_exist_ok=True)
            continue
        if source_path.is_file():
            shutil.copy2(source_path, target_root / source_path.name)

    return target_root
