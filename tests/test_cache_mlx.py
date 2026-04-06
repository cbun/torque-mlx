import numpy as np
import pytest

from torque_mlx.cache import TorqueKVCache
from torque_mlx.cache_mlx import TorqueKVCacheMLX
from torque_mlx.config import TorqueConfig
from torque_mlx.quantization import Codebook, pack_indices_batched, quantize
from torque_mlx.reference import streaming_attention_decode


def _uniform_codebook(bit_width: int) -> Codebook:
    levels = 1 << bit_width
    return Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, levels, dtype=np.float32),
        name="uniform",
    )


def test_mlx_cache_append_tracks_capacity() -> None:
    pytest.importorskip("mlx.core")

    cache = TorqueKVCacheMLX(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=2, kv_heads=2),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=2,
    )
    payload = np.zeros((2, 2, 64), dtype=np.float32)
    for _ in range(3):
        cache.append(key=payload, value=payload)

    assert cache.sequence_length == 3
    assert cache.capacity >= 3


def test_mlx_cache_append_many_matches_numpy_reference() -> None:
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    rng = np.random.default_rng(3)
    cache = TorqueKVCacheMLX(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=2, kv_heads=2),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=8,
    )
    keys = rng.uniform(-1.0, 1.0, size=(2, 2, 3, 64)).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=(2, 2, 3, 64)).astype(np.float32)

    cache.append_many_mlx(key=mx.array(keys), value=mx.array(values))
    mx.eval(cache._key_codes, cache._value_codes)

    expected_keys = pack_indices_batched(
        quantize(cache.rotation.apply(keys), cache.key_codebook),
        cache.config.bit_width,
    )
    expected_values = pack_indices_batched(
        quantize(cache.rotation.apply(values), cache.value_codebook),
        cache.config.bit_width,
    )

    np.testing.assert_array_equal(
        np.array(cache._key_codes[:, :, :3, :]),
        expected_keys,
    )
    np.testing.assert_array_equal(
        np.array(cache._value_codes[:, :, :3, :]),
        expected_values,
    )
    assert cache.sequence_length == 3


def test_mlx_cache_grouped_decode_matches_reference() -> None:
    pytest.importorskip("mlx.core")
    from torque_mlx.mlx_ops import metal_available

    if not metal_available():
        pytest.skip("Metal toolchain unavailable")

    rng = np.random.default_rng(9)
    config = TorqueConfig(bit_width=4, head_dim=64, num_layers=1, kv_heads=2)
    cache = TorqueKVCacheMLX(
        config=config,
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=8,
    )
    reference_cache = TorqueKVCache(
        config=config,
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )

    for _ in range(6):
        key = rng.uniform(-1.0, 1.0, size=(2, 64)).astype(np.float32)
        value = rng.uniform(-1.0, 1.0, size=(2, 64)).astype(np.float32)
        cache.append(key=key, value=value)
        reference_cache.append(key=key, value=value)

    grouped_query = rng.uniform(-1.0, 1.0, size=(4, 64)).astype(np.float32)
    actual = cache.decode_mlx(query=grouped_query)

    keys, values = reference_cache.export_dequantized()
    expected = np.zeros_like(grouped_query)
    kv_group_ratio = grouped_query.shape[0] // config.kv_heads
    for query_head_idx in range(grouped_query.shape[0]):
        kv_head_idx = query_head_idx // kv_group_ratio
        out_rot = streaming_attention_decode(
            reference_cache.rotation.apply(grouped_query[query_head_idx]),
            keys[0, kv_head_idx],
            values[0, kv_head_idx],
        )
        expected[query_head_idx] = reference_cache.rotation.inverse(out_rot)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mlx_cache_grouped_decode_matches_reference_with_dense_tail_flush() -> None:
    pytest.importorskip("mlx.core")
    from torque_mlx.mlx_ops import metal_available

    if not metal_available():
        pytest.skip("Metal toolchain unavailable")

    rng = np.random.default_rng(19)
    config = TorqueConfig(bit_width=4, head_dim=64, num_layers=1, kv_heads=2)
    cache = TorqueKVCacheMLX(
        config=config,
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=8,
        decode_tail_capacity=2,
    )
    reference_cache = TorqueKVCache(
        config=config,
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )

    for _ in range(5):
        key = rng.uniform(-1.0, 1.0, size=(2, 64)).astype(np.float32)
        value = rng.uniform(-1.0, 1.0, size=(2, 64)).astype(np.float32)
        cache.append(key=key, value=value)
        reference_cache.append(key=key, value=value)

    grouped_query = rng.uniform(-1.0, 1.0, size=(4, 64)).astype(np.float32)
    actual = cache.decode_mlx(query=grouped_query)

    keys, values = reference_cache.export_dequantized()
    expected = np.zeros_like(grouped_query)
    kv_group_ratio = grouped_query.shape[0] // config.kv_heads
    for query_head_idx in range(grouped_query.shape[0]):
        kv_head_idx = query_head_idx // kv_group_ratio
        out_rot = streaming_attention_decode(
            reference_cache.rotation.apply(grouped_query[query_head_idx]),
            keys[0, kv_head_idx],
            values[0, kv_head_idx],
        )
        expected[query_head_idx] = reference_cache.rotation.inverse(out_rot)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mlx_cache_auto_strategy_prefers_split_batched() -> None:
    pytest.importorskip("mlx.core")

    cache = TorqueKVCacheMLX(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=1, kv_heads=1),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=4,
        decode_strategy="auto",
    )

    assert cache.resolve_decode_strategy(sequence_length=128) == "split_batched"
    assert cache.resolve_decode_strategy(sequence_length=256) == "split_batched"


def test_mlx_cache_default_strategy_is_split_batched() -> None:
    pytest.importorskip("mlx.core")

    cache = TorqueKVCacheMLX(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=1, kv_heads=1),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
        initial_capacity=4,
    )

    assert cache.resolve_decode_strategy(sequence_length=32) == "split_batched"
