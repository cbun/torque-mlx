from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

import numpy as np


class RotationMode(StrEnum):
    HADAMARD = "hadamard"


def validate_head_dim(head_dim: int) -> None:
    if head_dim not in {64, 128, 256}:
        raise ValueError(f"Unsupported head dimension for rotation: {head_dim}")


def _fwht_last_axis(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = array.copy()
    head_dim = result.shape[-1]
    step = 1
    while step < head_dim:
        reshaped = result.reshape(*result.shape[:-1], -1, 2 * step)
        left = reshaped[..., :step].copy()
        right = reshaped[..., step : 2 * step].copy()
        reshaped[..., :step] = left + right
        reshaped[..., step : 2 * step] = left - right
        result = reshaped.reshape(result.shape)
        step *= 2
    return result


@dataclass(frozen=True, slots=True)
class RotationSpec:
    """Structured orthogonal rotation Pi = D1 H D2."""

    head_dim: int
    signs_left: np.ndarray
    signs_right: np.ndarray
    seed: int | None = None

    def __post_init__(self) -> None:
        validate_head_dim(self.head_dim)
        object.__setattr__(
            self,
            "signs_left",
            np.asarray(self.signs_left, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "signs_right",
            np.asarray(self.signs_right, dtype=np.float32),
        )
        if self.signs_left.shape != (self.head_dim,) or self.signs_right.shape != (self.head_dim,):
            raise ValueError("Rotation sign vectors must match head_dim")

    @classmethod
    def from_seed(cls, head_dim: int, seed: int = 0) -> "RotationSpec":
        validate_head_dim(head_dim)
        rng = np.random.default_rng(seed)
        signs_left = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=head_dim)
        signs_right = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=head_dim)
        return cls(
            head_dim=head_dim,
            signs_left=signs_left,
            signs_right=signs_right,
            seed=seed,
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.head_dim:
            raise ValueError(f"Expected last dimension {self.head_dim}, got {array.shape[-1]}")
        rotated = array * self.signs_right
        rotated = _fwht_last_axis(rotated) / sqrt(self.head_dim)
        return rotated * self.signs_left

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.head_dim:
            raise ValueError(f"Expected last dimension {self.head_dim}, got {array.shape[-1]}")
        rotated = array * self.signs_left
        rotated = _fwht_last_axis(rotated) / sqrt(self.head_dim)
        return rotated * self.signs_right

    def matrix(self) -> np.ndarray:
        eye = np.eye(self.head_dim, dtype=np.float32)
        return self.apply(eye).T


def apply_structured_rotation(values: np.ndarray, head_dim: int, seed: int = 0) -> np.ndarray:
    return RotationSpec.from_seed(head_dim=head_dim, seed=seed).apply(values)
