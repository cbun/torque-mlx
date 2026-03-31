from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from torque_mlx.config import TorqueConfig
from torque_mlx.layout import CacheMetadata
from torque_mlx.mlx_ops import decode_packed_attention
from torque_mlx.quantization import (
    Codebook,
    build_gaussian_codebook,
    dequantize,
    pack_indices,
    quantize,
    unpack_indices,
)
from torque_mlx.reference import streaming_attention_decode
from torque_mlx.rotation import RotationSpec


@dataclass(slots=True)
class _LayerStorage:
    key_codes: list[list[np.ndarray]]
    value_codes: list[list[np.ndarray]]


def _init_layer_storage(kv_heads: int) -> _LayerStorage:
    return _LayerStorage(
        key_codes=[[] for _ in range(kv_heads)],
        value_codes=[[] for _ in range(kv_heads)],
    )


@dataclass(slots=True)
class TorqueKVCache:
    """Reference runtime for the rotation-native KV cache."""

    config: TorqueConfig = field(default_factory=TorqueConfig)
    key_codebook: Codebook | None = None
    value_codebook: Codebook | None = None
    sequence_length: int = 0
    rotation: RotationSpec = field(init=False, repr=False)
    _layers: list[_LayerStorage] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self.key_codebook = self.key_codebook or build_gaussian_codebook(
            self.config.bit_width,
            seed=self.config.rotation_seed,
        )
        self.value_codebook = self.value_codebook or build_gaussian_codebook(
            self.config.bit_width,
            seed=self.config.rotation_seed + 1,
        )
        self.rotation = RotationSpec.from_seed(
            head_dim=self.config.head_dim,
            seed=self.config.rotation_seed,
        )
        self._layers = [
            _init_layer_storage(self.config.kv_heads)
            for _ in range(self.config.num_layers)
        ]

    @property
    def metadata(self) -> CacheMetadata:
        return CacheMetadata(
            variant_id=self.config.variant_id,
            bit_width=self.config.bit_width,
            head_dim=self.config.head_dim,
            num_layers=self.config.num_layers,
            kv_heads=self.config.kv_heads,
            rotation_mode=self.config.rotation_mode.value,
            fused_weights=self.config.fused_weights,
            codebook_name=self.key_codebook.name,
        )

    def append(self, *, key, value) -> None:
        """Append one decode step for all layers and KV heads."""
        keys = self._normalize_layer_heads(key, "key")
        values = self._normalize_layer_heads(value, "value")

        for layer_idx in range(self.config.num_layers):
            for head_idx in range(self.config.kv_heads):
                key_rot = self.rotation.apply(keys[layer_idx, head_idx])
                value_rot = self.rotation.apply(values[layer_idx, head_idx])
                key_codes = quantize(key_rot, self.key_codebook)
                value_codes = quantize(value_rot, self.value_codebook)
                self._layers[layer_idx].key_codes[head_idx].append(
                    pack_indices(key_codes, self.config.bit_width),
                )
                self._layers[layer_idx].value_codes[head_idx].append(
                    pack_indices(value_codes, self.config.bit_width),
                )

        self.sequence_length += 1

    def decode(self, *, query) -> np.ndarray:
        queries = self._normalize_layer_heads(query, "query")
        outputs = np.zeros_like(queries)

        for layer_idx in range(self.config.num_layers):
            for head_idx in range(self.config.kv_heads):
                packed_keys = self._layers[layer_idx].key_codes[head_idx]
                packed_values = self._layers[layer_idx].value_codes[head_idx]
                if not packed_keys:
                    continue

                query_rot = self.rotation.apply(queries[layer_idx, head_idx])
                keys = np.stack(
                    [
                        dequantize(
                            unpack_indices(words, self.config.bit_width, self.config.head_dim),
                            self.key_codebook,
                        )
                        for words in packed_keys
                    ],
                    axis=0,
                )
                values = np.stack(
                    [
                        dequantize(
                            unpack_indices(words, self.config.bit_width, self.config.head_dim),
                            self.value_codebook,
                        )
                        for words in packed_values
                    ],
                    axis=0,
                )
                out_rot = streaming_attention_decode(query_rot, keys, values)
                if self.config.fused_weights:
                    outputs[layer_idx, head_idx] = out_rot
                else:
                    outputs[layer_idx, head_idx] = self.rotation.inverse(out_rot)

        if self.config.num_layers == 1 and self.config.kv_heads == 1:
            return outputs[0, 0]
        if self.config.num_layers == 1:
            return outputs[0]
        return outputs

    def decode_mlx(self, *, query):
        """Run decode through the MLX JIT Metal packed-code path."""
        import mlx.core as mx

        queries = self._normalize_layer_heads(query, "query")
        outputs = np.zeros_like(queries)

        key_centroids = mx.array(self.key_codebook.centroids)
        value_centroids = mx.array(self.value_codebook.centroids)

        for layer_idx in range(self.config.num_layers):
            for head_idx in range(self.config.kv_heads):
                packed_keys = self._layers[layer_idx].key_codes[head_idx]
                packed_values = self._layers[layer_idx].value_codes[head_idx]
                if not packed_keys:
                    continue

                query_rot = self.rotation.apply(queries[layer_idx, head_idx])
                out_rot = decode_packed_attention(
                    mx.array(query_rot),
                    mx.array(np.stack(packed_keys, axis=0).astype(np.uint32)),
                    mx.array(np.stack(packed_values, axis=0).astype(np.uint32)),
                    key_centroids,
                    value_centroids,
                    bit_width=self.config.bit_width,
                    head_dim=self.config.head_dim,
                )
                mx.eval(out_rot)
                out_rot_np = np.array(out_rot)
                if self.config.fused_weights:
                    outputs[layer_idx, head_idx] = out_rot_np
                else:
                    outputs[layer_idx, head_idx] = self.rotation.inverse(out_rot_np)

        if self.config.num_layers == 1 and self.config.kv_heads == 1:
            return outputs[0, 0]
        if self.config.num_layers == 1:
            return outputs[0]
        return outputs

    def reset(self) -> None:
        self.sequence_length = 0
        self._layers = [
            _init_layer_storage(self.config.kv_heads)
            for _ in range(self.config.num_layers)
        ]

    def export_dequantized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dequantized rotated cache contents for testing and benchmarking."""
        keys = np.zeros(
            (
                self.config.num_layers,
                self.config.kv_heads,
                self.sequence_length,
                self.config.head_dim,
            ),
            dtype=np.float32,
        )
        values = np.zeros_like(keys)

        for layer_idx in range(self.config.num_layers):
            for head_idx in range(self.config.kv_heads):
                for token_idx, words in enumerate(self._layers[layer_idx].key_codes[head_idx]):
                    keys[layer_idx, head_idx, token_idx] = dequantize(
                        unpack_indices(words, self.config.bit_width, self.config.head_dim),
                        self.key_codebook,
                    )
                for token_idx, words in enumerate(self._layers[layer_idx].value_codes[head_idx]):
                    values[layer_idx, head_idx, token_idx] = dequantize(
                        unpack_indices(words, self.config.bit_width, self.config.head_dim),
                        self.value_codebook,
                    )
        return keys, values

    def _normalize_layer_heads(self, values, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        target = (self.config.num_layers, self.config.kv_heads, self.config.head_dim)
        if array.shape == (self.config.head_dim,):
            if target[:2] != (1, 1):
                raise ValueError(f"{label} shape {array.shape} only valid for 1 layer / 1 head")
            return array.reshape(target)
        if array.shape == (self.config.kv_heads, self.config.head_dim) and self.config.num_layers == 1:
            return array.reshape(target)
        if array.shape != target:
            raise ValueError(f"{label} must have shape {target}, got {array.shape}")
        return array
