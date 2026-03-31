from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torque_mlx.rotation import RotationSpec


@dataclass(frozen=True, slots=True)
class FusedAttentionWeights:
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    rotation_seed: int


def fuse_attention_weights(
    *,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    rotation: RotationSpec,
) -> FusedAttentionWeights:
    """Fuse Pi into Q/K/V and Pi^T into O.

    The weight convention is matrix-vector multiplication with shapes
    `[head_dim, input_dim]` for Q/K/V and `[output_dim, head_dim]` for O.
    """
    rotation_matrix = rotation.matrix()
    rotation_inverse = rotation_matrix.T

    q = np.asarray(w_q, dtype=np.float32)
    k = np.asarray(w_k, dtype=np.float32)
    v = np.asarray(w_v, dtype=np.float32)
    o = np.asarray(w_o, dtype=np.float32)

    if q.shape[0] != rotation.head_dim or k.shape[0] != rotation.head_dim or v.shape[0] != rotation.head_dim:
        raise ValueError("Q/K/V output dimension must equal rotation head_dim")
    if o.shape[1] != rotation.head_dim:
        raise ValueError("O input dimension must equal rotation head_dim")

    return FusedAttentionWeights(
        w_q=rotation_matrix @ q,
        w_k=rotation_matrix @ k,
        w_v=rotation_matrix @ v,
        w_o=o @ rotation_inverse,
        rotation_seed=-1 if rotation.seed is None else rotation.seed,
    )
