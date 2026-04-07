from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np

from torque_mlx.config import TorqueConfig
from torque_mlx.layout import CacheMetadata, PackedKVLayout
from torque_mlx.mlx_ops import (
    accumulate_packed_values_batched,
    decode_packed_attention,
    decode_packed_attention_split_batched,
    metal_available,
    quantize_and_pack_rows_metal,
    quantize_and_pack_rows_dual_metal,
    score_packed_query_batched,
)
from torque_mlx.quantization import (
    Codebook,
    build_gaussian_codebook,
    codebook_boundaries,
    pack_indices_batched,
    pack_indices_batched_mlx,
    quantize,
    quantize_mlx,
)
from torque_mlx.rotation import RotationSpec, apply_structured_rotation_mlx, inverse_structured_rotation_mlx

DecodeStrategy = Literal["auto", "split_batched", "fused_per_head"]
SUPPORTED_DECODE_STRATEGIES: tuple[DecodeStrategy, ...] = (
    "auto",
    "split_batched",
    "fused_per_head",
)


@dataclass(slots=True)
class TorqueKVCacheMLX:
    """MLX-backed packed KV cache with device-resident code storage."""

    config: TorqueConfig = field(default_factory=TorqueConfig)
    key_codebook: Codebook | None = None
    value_codebook: Codebook | None = None
    initial_capacity: int = 256
    growth_factor: int = 2
    decode_tail_capacity: int = 16
    decode_strategy: DecodeStrategy = "split_batched"
    profile_decode_components: bool = False
    rotate_keys_on_append: bool = True
    rotate_values_on_append: bool = True
    rotate_queries_on_decode: bool = True
    sequence_length: int = 0
    rotation: RotationSpec = field(init=False, repr=False)
    _capacity: int = field(init=False, repr=False)
    _layout: PackedKVLayout = field(init=False, repr=False)
    _key_codes: Any = field(init=False, repr=False)
    _value_codes: Any = field(init=False, repr=False)
    _key_centroids: Any = field(init=False, repr=False)
    _value_centroids: Any = field(init=False, repr=False)
    _key_boundaries: Any = field(init=False, repr=False)
    _value_boundaries: Any = field(init=False, repr=False)
    _rotation_signs_left: Any = field(init=False, repr=False)
    _rotation_signs_right: Any = field(init=False, repr=False)
    _use_append_pack_kernel: bool = field(init=False, repr=False)
    _tail_storage_dtype: Any = field(init=False, repr=False)
    _tail_keys: Any = field(init=False, repr=False)
    _tail_values: Any = field(init=False, repr=False)
    _tail_length: int = field(init=False, repr=False)
    _query_index_cache: dict[int, tuple[Any, Any] | Any] = field(init=False, repr=False)
    last_decode_profile: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import mlx.core as mx

        self.config.validate()
        if self.initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if self.growth_factor < 2:
            raise ValueError("growth_factor must be at least 2")
        if self.decode_tail_capacity < 0:
            raise ValueError("decode_tail_capacity must be non-negative")
        if self.decode_strategy not in SUPPORTED_DECODE_STRATEGIES:
            raise ValueError(
                "decode_strategy must be one of "
                + ", ".join(SUPPORTED_DECODE_STRATEGIES),
            )

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
        self._layout = self.config.layout
        self._capacity = self.initial_capacity
        self._key_codes = mx.zeros(
            (self.config.num_layers, self.config.kv_heads, self._capacity, self._layout.packed_words),
            dtype=mx.uint32,
        )
        self._value_codes = mx.zeros_like(self._key_codes)
        self._key_centroids = mx.array(self.key_codebook.centroids)
        self._value_centroids = mx.array(self.value_codebook.centroids)
        self._key_boundaries = mx.array(codebook_boundaries(self.key_codebook))
        self._value_boundaries = mx.array(codebook_boundaries(self.value_codebook))
        self._rotation_signs_left = mx.array(self.rotation.signs_left.astype(np.float32))
        self._rotation_signs_right = mx.array(self.rotation.signs_right.astype(np.float32))
        self._use_append_pack_kernel = metal_available()
        self._tail_storage_dtype = mx.float16
        self._tail_length = 0
        self._query_index_cache = {}
        self.last_decode_profile = {}
        self._tail_keys = mx.zeros(
            (self.config.num_layers, self.config.kv_heads, max(1, self.decode_tail_capacity), self.config.head_dim),
            dtype=self._tail_storage_dtype,
        )
        self._tail_values = mx.zeros_like(self._tail_keys)

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

    @property
    def capacity(self) -> int:
        return self._capacity + self.decode_tail_capacity

    @property
    def offset(self) -> int:
        return self.sequence_length

    def append(self, *, key, value) -> None:
        self.append_many_mlx(key=key, value=value)

    def append_many_mlx(self, *, key, value) -> None:
        keys = self._normalize_layer_heads_sequence_device(key, "key")
        values = self._normalize_layer_heads_sequence_device(value, "value")
        if keys.shape[:3] != values.shape[:3]:
            raise ValueError(
                f"key/value sequence dimensions must match, got {keys.shape} and {values.shape}",
            )

        seq_len = int(keys.shape[2])
        keys_rot = self._apply_append_rotation(keys, rotate=self.rotate_keys_on_append)
        values_rot = self._apply_append_rotation(values, rotate=self.rotate_values_on_append)
        if seq_len == 1 and self.decode_tail_capacity > 0:
            self._append_dense_tail(keys_rot, values_rot)
            self.sequence_length += 1
            return

        if self._tail_length > 0:
            self._flush_dense_tail()

        self._append_packed_tokens(keys_rot, values_rot)
        self.sequence_length += seq_len

    def decode_mlx(self, *, query, return_numpy: bool = True):
        import mlx.core as mx

        self.last_decode_profile = {}
        queries = self._normalize_query_heads_device(query, "query")
        if self.sequence_length == 0:
            zeros = mx.zeros((self.config.num_layers, int(queries.shape[1]), self.config.head_dim), dtype=mx.float32)
            return self._finalize_output(zeros, query_heads=int(queries.shape[1]), return_numpy=return_numpy)

        query_heads = int(queries.shape[1])
        if query_heads % self.config.kv_heads != 0:
            raise ValueError(
                f"query head count {query_heads} must be divisible by kv_heads {self.config.kv_heads}",
            )
        kv_group_ratio = query_heads // self.config.kv_heads
        query_rot = (
            apply_structured_rotation_mlx(
                queries,
                signs_left=self._rotation_signs_left,
                signs_right=self._rotation_signs_right,
            )
            if self.rotate_queries_on_decode
            else queries
        )
        query_rot_flat = mx.reshape(query_rot, (self.config.num_layers * query_heads, self.config.head_dim))
        indices = self._grouped_query_indices(query_heads=query_heads, kv_group_ratio=kv_group_ratio)
        if self.config.num_layers == 1:
            kv_head_indices = indices
            layer_indices = None
        else:
            layer_indices, kv_head_indices = indices
        packed_length = self.sequence_length - self._tail_length

        if self._tail_length == 0:
            if self.config.num_layers == 1:
                batched_k_codes = self._key_codes[0, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[0, kv_head_indices, :packed_length, :]
            else:
                batched_k_codes = self._key_codes[layer_indices, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[layer_indices, kv_head_indices, :packed_length, :]
            out_rot = self._decode_packed_batch(
                query_rot_flat=query_rot_flat,
                batched_k_codes=batched_k_codes,
                batched_v_codes=batched_v_codes,
            )
        else:
            out_rot = self._decode_packed_and_tail_batch(
                query_rot_flat=query_rot_flat,
                layer_indices=layer_indices,
                kv_head_indices=kv_head_indices,
                packed_length=packed_length,
            )
        if self.config.fused_weights:
            outputs = out_rot
        else:
            outputs = inverse_structured_rotation_mlx(
                out_rot,
                signs_left=self._rotation_signs_left,
                signs_right=self._rotation_signs_right,
            )
        outputs = mx.reshape(outputs, (self.config.num_layers, query_heads, self.config.head_dim))
        return self._finalize_output(outputs, query_heads=query_heads, return_numpy=return_numpy)

    def decode_mlx_with_current(self, *, query, key, value, return_numpy: bool = True):
        import mlx.core as mx

        self.last_decode_profile = {}
        queries = self._normalize_query_heads_device(query, "query")
        current_keys = self._normalize_layer_heads_sequence_device(key, "key")
        current_values = self._normalize_layer_heads_sequence_device(value, "value")
        if int(current_keys.shape[2]) != 1 or int(current_values.shape[2]) != 1:
            raise ValueError("decode_mlx_with_current expects a single current token for key/value")

        current_keys = self._quantize_rows(
            self._apply_append_rotation(current_keys, rotate=self.rotate_keys_on_append),
            centroids=self._key_centroids,
            boundaries=self._key_boundaries,
        )
        current_values = self._quantize_rows(
            self._apply_append_rotation(current_values, rotate=self.rotate_values_on_append),
            centroids=self._value_centroids,
            boundaries=self._value_boundaries,
        )

        query_heads = int(queries.shape[1])
        if query_heads % self.config.kv_heads != 0:
            raise ValueError(
                f"query head count {query_heads} must be divisible by kv_heads {self.config.kv_heads}",
            )
        kv_group_ratio = query_heads // self.config.kv_heads
        query_rot = (
            apply_structured_rotation_mlx(
                queries,
                signs_left=self._rotation_signs_left,
                signs_right=self._rotation_signs_right,
            )
            if self.rotate_queries_on_decode
            else queries
        )
        query_rot_flat = mx.reshape(query_rot, (self.config.num_layers * query_heads, self.config.head_dim))
        indices = self._grouped_query_indices(query_heads=query_heads, kv_group_ratio=kv_group_ratio)
        if self.config.num_layers == 1:
            kv_head_indices = indices
            layer_indices = None
            current_tail_keys = current_keys[0, kv_head_indices, :, :]
            current_tail_values = current_values[0, kv_head_indices, :, :]
        else:
            layer_indices, kv_head_indices = indices
            current_tail_keys = current_keys[layer_indices, kv_head_indices, :, :]
            current_tail_values = current_values[layer_indices, kv_head_indices, :, :]

        packed_length = self.sequence_length - self._tail_length
        out_rot = self._decode_packed_and_tail_batch(
            query_rot_flat=query_rot_flat,
            layer_indices=layer_indices,
            kv_head_indices=kv_head_indices,
            packed_length=packed_length,
            extra_tail_keys=current_tail_keys,
            extra_tail_values=current_tail_values,
        )
        if self.config.fused_weights:
            outputs = out_rot
        else:
            outputs = inverse_structured_rotation_mlx(
                out_rot,
                signs_left=self._rotation_signs_left,
                signs_right=self._rotation_signs_right,
            )
        outputs = mx.reshape(outputs, (self.config.num_layers, query_heads, self.config.head_dim))
        return self._finalize_output(outputs, query_heads=query_heads, return_numpy=return_numpy)

    def _record_profile_timing(self, profile: dict[str, float] | None, key: str, start: float, *values) -> None:
        if profile is None:
            return
        import mlx.core as mx

        mx.eval(*values)
        profile[key] = profile.get(key, 0.0) + (perf_counter() - start)

    def reset(self) -> None:
        import mlx.core as mx

        self.sequence_length = 0
        self._tail_length = 0
        self._key_codes = mx.zeros_like(self._key_codes)
        self._value_codes = mx.zeros_like(self._value_codes)
        self._tail_keys = mx.zeros_like(self._tail_keys)
        self._tail_values = mx.zeros_like(self._tail_values)

    def make_mask(self, N: int, *, window_size: int | None = None, return_array: bool = False):
        import mlx.core as mx

        if N == 1:
            return None
        if not return_array and (window_size is None or N <= window_size):
            return "causal"

        offset = self.offset
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        if window_size is not None:
            mask = mask & (linds < rinds + window_size)
        return mask

    def _finalize_output(self, outputs, *, query_heads: int, return_numpy: bool):
        import mlx.core as mx

        if not return_numpy:
            return outputs
        mx.eval(outputs)
        output_np = np.array(outputs)
        if self.config.num_layers == 1 and query_heads == 1:
            return output_np[0, 0]
        if self.config.num_layers == 1:
            return output_np[0]
        return output_np

    def _ensure_capacity(self, min_capacity: int) -> None:
        import mlx.core as mx

        if min_capacity <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < min_capacity:
            new_capacity *= self.growth_factor

        new_key_codes = mx.zeros(
            (self.config.num_layers, self.config.kv_heads, new_capacity, self._layout.packed_words),
            dtype=self._key_codes.dtype,
        )
        new_value_codes = mx.zeros_like(new_key_codes)
        new_key_codes[:, :, : self.sequence_length, :] = self._key_codes[:, :, : self.sequence_length, :]
        new_value_codes[:, :, : self.sequence_length, :] = self._value_codes[:, :, : self.sequence_length, :]
        self._key_codes = new_key_codes
        self._value_codes = new_value_codes
        self._capacity = new_capacity

    def _append_dense_tail(self, keys_rot, values_rot) -> None:
        import mlx.core as mx

        quantized_keys = self._quantize_rows(
            keys_rot,
            centroids=self._key_centroids,
            boundaries=self._key_boundaries,
        )
        quantized_values = self._quantize_rows(
            values_rot,
            centroids=self._value_centroids,
            boundaries=self._value_boundaries,
        )
        if self._tail_length == self.decode_tail_capacity:
            self._flush_dense_tail()
        self._tail_keys[:, :, self._tail_length, :] = quantized_keys[:, :, 0, :].astype(
            self._tail_storage_dtype,
        )
        self._tail_values[:, :, self._tail_length, :] = quantized_values[:, :, 0, :].astype(
            self._tail_storage_dtype,
        )
        self._tail_length += 1

    def _append_packed_tokens(self, keys_rot, values_rot) -> None:
        packed_length = self.sequence_length - self._tail_length
        seq_len = int(keys_rot.shape[2])
        self._ensure_capacity(packed_length + seq_len)
        if self._use_append_pack_kernel:
            packed_keys, packed_values = self._quantize_and_pack_rows_dual(
                keys_rot,
                values_rot,
            )
        else:
            packed_keys = self._quantize_and_pack_rows(
                keys_rot,
                centroids=self._key_centroids,
                boundaries=self._key_boundaries,
            )
            packed_values = self._quantize_and_pack_rows(
                values_rot,
                centroids=self._value_centroids,
                boundaries=self._value_boundaries,
            )
        seq_slice = slice(packed_length, packed_length + seq_len)
        self._key_codes[:, :, seq_slice, :] = packed_keys
        self._value_codes[:, :, seq_slice, :] = packed_values

    def _flush_dense_tail(self) -> None:
        if self._tail_length == 0:
            return
        self._append_packed_tokens(
            self._tail_keys[:, :, : self._tail_length, :],
            self._tail_values[:, :, : self._tail_length, :],
        )
        self._tail_length = 0

    def _normalize_layer_heads_numpy(self, values, label: str) -> np.ndarray:
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

    def _normalize_layer_heads_sequence_device(self, values, label: str):
        import mlx.core as mx

        array = mx.array(values, dtype=mx.float32)
        target = (self.config.num_layers, self.config.kv_heads, self.config.head_dim)

        if len(array.shape) == 1 and array.shape == (self.config.head_dim,):
            if target[:2] != (1, 1):
                raise ValueError(f"{label} shape {array.shape} only valid for 1 layer / 1 head")
            return mx.reshape(array, (1, 1, 1, self.config.head_dim))
        if len(array.shape) == 2 and array.shape == (self.config.kv_heads, self.config.head_dim):
            if self.config.num_layers != 1:
                raise ValueError(f"{label} shape {array.shape} only valid for 1 layer")
            return mx.reshape(array, (1, self.config.kv_heads, 1, self.config.head_dim))
        if len(array.shape) == 3:
            if array.shape == target:
                return mx.reshape(array, (*target[:2], 1, self.config.head_dim))
            if self.config.num_layers == 1 and array.shape[0] == self.config.kv_heads and array.shape[2] == self.config.head_dim:
                return mx.reshape(array, (1, self.config.kv_heads, int(array.shape[1]), self.config.head_dim))
        if len(array.shape) == 4 and array.shape[0] == self.config.num_layers and array.shape[1] == self.config.kv_heads and array.shape[3] == self.config.head_dim:
            return array
        raise ValueError(
            f"{label} must have shape ({self.config.num_layers}, {self.config.kv_heads}, seq_len, {self.config.head_dim}) "
            f"or a compatible single-token form, got {array.shape}",
        )

    def _apply_append_rotation(self, values, *, rotate: bool):
        if not rotate:
            return values
        return apply_structured_rotation_mlx(
            values,
            signs_left=self._rotation_signs_left,
            signs_right=self._rotation_signs_right,
        )

    def _quantize_and_pack_rows(self, values, *, centroids, boundaries):
        import mlx.core as mx

        row_shape = tuple(int(dim) for dim in values.shape[:-1])
        flat = mx.reshape(values.astype(mx.float32), (-1, self.config.head_dim))
        if self._use_append_pack_kernel:
            packed = quantize_and_pack_rows_metal(
                flat,
                boundaries,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
        else:
            packed = pack_indices_batched_mlx(
                quantize_mlx(flat, centroids, boundaries=boundaries),
                self.config.bit_width,
            )
        return mx.reshape(packed, (*row_shape, self._layout.packed_words))

    def _quantize_rows(self, values, *, centroids, boundaries):
        import mlx.core as mx

        flat = mx.reshape(values, (-1, self.config.head_dim))
        indices = quantize_mlx(flat, centroids, boundaries=boundaries)
        quantized = centroids[indices]
        return mx.reshape(quantized, values.shape)

    def _quantize_and_pack_rows_dual(self, key_values, value_values):
        import mlx.core as mx

        row_shape = tuple(int(dim) for dim in key_values.shape[:-1])
        flat_keys = mx.reshape(key_values.astype(mx.float32), (-1, self.config.head_dim))
        flat_values = mx.reshape(value_values.astype(mx.float32), (-1, self.config.head_dim))
        packed_keys, packed_values = quantize_and_pack_rows_dual_metal(
            flat_keys,
            flat_values,
            self._key_boundaries,
            self._value_boundaries,
            bit_width=self.config.bit_width,
            head_dim=self.config.head_dim,
        )
        return (
            mx.reshape(packed_keys, (*row_shape, self._layout.packed_words)),
            mx.reshape(packed_values, (*row_shape, self._layout.packed_words)),
        )

    def _decode_packed_and_tail_batch(
        self,
        *,
        query_rot_flat,
        layer_indices,
        kv_head_indices,
        packed_length: int,
        extra_tail_keys=None,
        extra_tail_values=None,
    ):
        import mlx.core as mx

        profile = self.last_decode_profile if self.profile_decode_components else None
        scale = 1.0 / np.sqrt(self.config.head_dim)
        tail_mask = None
        if self._tail_length > 0:
            if self.config.num_layers == 1:
                tail_keys = self._tail_keys[0, kv_head_indices, : self._tail_length, :]
                tail_values = self._tail_values[0, kv_head_indices, : self._tail_length, :]
            else:
                tail_keys = self._tail_keys[layer_indices, kv_head_indices, : self._tail_length, :]
                tail_values = self._tail_values[layer_indices, kv_head_indices, : self._tail_length, :]
        else:
            tail_keys = None
            tail_values = None
        if extra_tail_keys is not None:
            tail_keys = extra_tail_keys if tail_keys is None else mx.concatenate([tail_keys, extra_tail_keys], axis=1)
            tail_values = extra_tail_values if tail_values is None else mx.concatenate([tail_values, extra_tail_values], axis=1)

        if tail_keys is None or tail_values is None:
            if self.config.num_layers == 1:
                batched_k_codes = self._key_codes[0, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[0, kv_head_indices, :packed_length, :]
            else:
                batched_k_codes = self._key_codes[layer_indices, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[layer_indices, kv_head_indices, :packed_length, :]
            return self._decode_packed_batch(
                query_rot_flat=query_rot_flat,
                batched_k_codes=batched_k_codes,
                batched_v_codes=batched_v_codes,
            )

        tail_token_count = int(tail_keys.shape[1])
        if tail_token_count == 1:
            tail_started = perf_counter() if profile is not None else 0.0
            tail_scores = mx.sum(
                tail_keys[:, 0, :].astype(mx.float32) * query_rot_flat,
                axis=1,
                keepdims=True,
            )
            self._record_profile_timing(profile, "tail_seconds", tail_started, tail_scores)

            if packed_length == 0:
                softmax_started = perf_counter() if profile is not None else 0.0
                tail_weights = mx.softmax(tail_scores * scale, axis=1)
                self._record_profile_timing(profile, "softmax_seconds", softmax_started, tail_weights)
                tail_started = perf_counter() if profile is not None else 0.0
                tail_out = tail_values[:, 0, :].astype(mx.float32) * tail_weights
                self._record_profile_timing(profile, "tail_seconds", tail_started, tail_out)
                return tail_out

            if self.config.num_layers == 1:
                batched_k_codes = self._key_codes[0, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[0, kv_head_indices, :packed_length, :]
            else:
                batched_k_codes = self._key_codes[layer_indices, kv_head_indices, :packed_length, :]
                batched_v_codes = self._value_codes[layer_indices, kv_head_indices, :packed_length, :]
            score_started = perf_counter() if profile is not None else 0.0
            packed_scores = score_packed_query_batched(
                query_rot_flat,
                batched_k_codes,
                self._key_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
            self._record_profile_timing(profile, "packed_score_seconds", score_started, packed_scores)
            softmax_started = perf_counter() if profile is not None else 0.0
            all_scores = mx.concatenate([packed_scores, tail_scores], axis=1)
            all_weights = mx.softmax(all_scores * scale, axis=1)
            self._record_profile_timing(profile, "softmax_seconds", softmax_started, all_weights)
            value_started = perf_counter() if profile is not None else 0.0
            packed_out = accumulate_packed_values_batched(
                all_weights[:, :packed_length],
                batched_v_codes,
                self._value_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
            self._record_profile_timing(profile, "packed_value_seconds", value_started, packed_out)
            tail_started = perf_counter() if profile is not None else 0.0
            tail_out = tail_values[:, 0, :].astype(mx.float32) * all_weights[:, packed_length : packed_length + 1]
            self._record_profile_timing(profile, "tail_seconds", tail_started, tail_out)
            return packed_out + tail_out

        tail_query = mx.expand_dims(query_rot_flat.astype(self._tail_storage_dtype), axis=-1)
        tail_started = perf_counter() if profile is not None else 0.0
        tail_scores = mx.squeeze(mx.matmul(tail_keys, tail_query), axis=-1).astype(mx.float32)
        self._record_profile_timing(profile, "tail_seconds", tail_started, tail_scores)

        if packed_length == 0:
            softmax_started = perf_counter() if profile is not None else 0.0
            tail_weights = mx.softmax(tail_scores * scale, axis=1)
            self._record_profile_timing(profile, "softmax_seconds", softmax_started, tail_weights)
            tail_started = perf_counter() if profile is not None else 0.0
            tail_out = mx.squeeze(
                mx.matmul(
                    mx.expand_dims(tail_weights.astype(self._tail_storage_dtype), axis=1),
                    tail_values,
                ),
                axis=1,
            ).astype(mx.float32)
            self._record_profile_timing(profile, "tail_seconds", tail_started, tail_out)
            return tail_out

        if self.config.num_layers == 1:
            batched_k_codes = self._key_codes[0, kv_head_indices, :packed_length, :]
            batched_v_codes = self._value_codes[0, kv_head_indices, :packed_length, :]
        else:
            batched_k_codes = self._key_codes[layer_indices, kv_head_indices, :packed_length, :]
            batched_v_codes = self._value_codes[layer_indices, kv_head_indices, :packed_length, :]
        score_started = perf_counter() if profile is not None else 0.0
        packed_scores = score_packed_query_batched(
            query_rot_flat,
            batched_k_codes,
            self._key_centroids,
            bit_width=self.config.bit_width,
            head_dim=self.config.head_dim,
        )
        self._record_profile_timing(profile, "packed_score_seconds", score_started, packed_scores)
        softmax_started = perf_counter() if profile is not None else 0.0
        all_scores = mx.concatenate([packed_scores, tail_scores], axis=1)
        all_weights = mx.softmax(all_scores * scale, axis=1)
        self._record_profile_timing(profile, "softmax_seconds", softmax_started, all_weights)
        value_started = perf_counter() if profile is not None else 0.0
        packed_out = accumulate_packed_values_batched(
            all_weights[:, :packed_length],
            batched_v_codes,
            self._value_centroids,
            bit_width=self.config.bit_width,
            head_dim=self.config.head_dim,
        )
        self._record_profile_timing(profile, "packed_value_seconds", value_started, packed_out)
        tail_started = perf_counter() if profile is not None else 0.0
        tail_out = mx.squeeze(
            mx.matmul(
                mx.expand_dims(
                    all_weights[:, packed_length:].astype(self._tail_storage_dtype),
                    axis=1,
                ),
                tail_values,
            ),
            axis=1,
        ).astype(mx.float32)
        self._record_profile_timing(profile, "tail_seconds", tail_started, tail_out)
        return packed_out + tail_out

    def _grouped_query_indices(self, *, query_heads: int, kv_group_ratio: int):
        import mlx.core as mx

        cached = self._query_index_cache.get(query_heads)
        if cached is not None:
            return cached

        if self.config.num_layers == 1:
            kv_head_indices = mx.array(np.arange(query_heads, dtype=np.int32) // kv_group_ratio)
            self._query_index_cache[query_heads] = kv_head_indices
            return kv_head_indices

        layer_indices = mx.array(np.repeat(np.arange(self.config.num_layers, dtype=np.int32), query_heads))
        query_head_indices = np.tile(np.arange(query_heads, dtype=np.int32), self.config.num_layers)
        kv_head_indices = mx.array(query_head_indices // kv_group_ratio)
        self._query_index_cache[query_heads] = (layer_indices, kv_head_indices)
        return self._query_index_cache[query_heads]

    def _normalize_query_heads_device(self, values, label: str):
        import mlx.core as mx

        array = mx.array(values, dtype=mx.float32)
        if len(array.shape) == 1 and array.shape == (self.config.head_dim,):
            if (self.config.num_layers, self.config.kv_heads) != (1, 1):
                raise ValueError(f"{label} shape {array.shape} only valid for 1 layer / 1 head")
            return mx.reshape(array, (1, 1, self.config.head_dim))
        if len(array.shape) == 2 and array.shape[1] == self.config.head_dim:
            if self.config.num_layers == 1:
                return mx.reshape(array, (1, array.shape[0], self.config.head_dim))
        if len(array.shape) == 3 and array.shape[0] == self.config.num_layers and array.shape[2] == self.config.head_dim:
            return array
        raise ValueError(
            f"{label} must have shape ({self.config.num_layers}, query_heads, {self.config.head_dim}) "
            f"or a compatible 1-layer form, got {array.shape}",
        )

    def resolve_decode_strategy(self, *, sequence_length: int | None = None) -> DecodeStrategy:
        if self.decode_strategy != "auto":
            return self.decode_strategy
        return "split_batched"

    def _decode_packed_batch(self, *, query_rot_flat, batched_k_codes, batched_v_codes):
        import mlx.core as mx

        strategy = self.resolve_decode_strategy()
        profile = self.last_decode_profile if self.profile_decode_components else None
        if strategy == "split_batched" and profile is not None:
            scale = 1.0 / np.sqrt(self.config.head_dim)
            score_started = perf_counter()
            scores = score_packed_query_batched(
                query_rot_flat,
                batched_k_codes,
                self._key_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
            self._record_profile_timing(profile, "packed_score_seconds", score_started, scores)
            softmax_started = perf_counter()
            weights = mx.softmax(scores * scale, axis=1)
            self._record_profile_timing(profile, "softmax_seconds", softmax_started, weights)
            value_started = perf_counter()
            out = accumulate_packed_values_batched(
                weights,
                batched_v_codes,
                self._value_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
            self._record_profile_timing(profile, "packed_value_seconds", value_started, out)
            return out
        if strategy == "split_batched":
            return decode_packed_attention_split_batched(
                query_rot_flat,
                batched_k_codes,
                batched_v_codes,
                self._key_centroids,
                self._value_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )

        outputs = [
            decode_packed_attention(
                query_rot_flat[batch_idx],
                batched_k_codes[batch_idx],
                batched_v_codes[batch_idx],
                self._key_centroids,
                self._value_centroids,
                bit_width=self.config.bit_width,
                head_dim=self.config.head_dim,
            )
            for batch_idx in range(int(query_rot_flat.shape[0]))
        ]
        return mx.stack(outputs, axis=0)
