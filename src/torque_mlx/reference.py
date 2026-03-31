from __future__ import annotations

from math import sqrt

import numpy as np


def streaming_attention_decode(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Reference decode attention for q_len=1."""
    query_vec = np.asarray(query, dtype=np.float32)
    key_matrix = np.asarray(keys, dtype=np.float32)
    value_matrix = np.asarray(values, dtype=np.float32)

    if key_matrix.ndim != 2 or value_matrix.ndim != 2:
        raise ValueError("keys and values must be rank-2 [seq_len, head_dim]")
    if key_matrix.shape != value_matrix.shape:
        raise ValueError("keys and values must share the same shape")
    if query_vec.shape != (key_matrix.shape[1],):
        raise ValueError("query must have shape [head_dim]")

    scale = 1.0 / sqrt(query_vec.shape[0])
    m_prev = -np.inf
    l_prev = 0.0
    out = np.zeros_like(query_vec)

    for key_vec, value_vec in zip(key_matrix, value_matrix, strict=True):
        score = float(np.dot(query_vec, key_vec) * scale)
        m_new = max(m_prev, score)
        l_scale = 0.0 if np.isneginf(m_prev) else float(np.exp(m_prev - m_new))
        p = float(np.exp(score - m_new))
        out *= l_scale
        out += p * value_vec
        l_prev = l_prev * l_scale + p
        m_prev = m_new

    if l_prev == 0.0:
        return out
    return out / l_prev
