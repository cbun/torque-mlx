import numpy as np
import pytest

from torque_mlx.cache import TorqueKVCache
from torque_mlx.config import TorqueConfig
from torque_mlx.quantization import Codebook
from torque_mlx.reference import streaming_attention_decode


def _uniform_codebook(bit_width: int) -> Codebook:
    levels = 1 << bit_width
    return Codebook(
        bit_width=bit_width,
        centroids=np.linspace(-1.0, 1.0, levels, dtype=np.float32),
        name="uniform",
    )


def test_cache_append_and_reset() -> None:
    cache = TorqueKVCache(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=2, kv_heads=3),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )
    payload = np.zeros((2, 3, 64), dtype=np.float32)
    cache.append(key=payload, value=payload)
    assert cache.sequence_length == 1
    cache.reset()
    assert cache.sequence_length == 0


def test_decode_with_single_token_matches_single_value_path() -> None:
    cache = TorqueKVCache(
        config=TorqueConfig(bit_width=4, head_dim=64),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )
    key = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    value = np.linspace(1.0, -1.0, 64, dtype=np.float32)
    query = key.copy()
    cache.append(key=key, value=value)
    output = cache.decode(query=query)

    _, values = cache.export_dequantized()
    expected = cache.rotation.inverse(values[0, 0, 0])
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_cache_metadata_exposes_variant() -> None:
    cache = TorqueKVCache(
        config=TorqueConfig(bit_width=3, head_dim=128, fused_weights=True),
        key_codebook=_uniform_codebook(3),
        value_codebook=_uniform_codebook(3),
    )
    metadata = cache.metadata
    assert metadata.variant_id == "b3-h128-hadamard-fused"
    assert metadata.codebook_name == "uniform"


def test_decode_shape_validation() -> None:
    cache = TorqueKVCache(
        config=TorqueConfig(bit_width=4, head_dim=64, num_layers=1, kv_heads=2),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )
    payload = np.zeros((2, 64), dtype=np.float32)
    cache.append(key=payload, value=payload)
    with pytest.raises(ValueError):
        cache.decode(query=np.zeros((64,), dtype=np.float32))


def test_decode_matches_dequantized_reference_attention() -> None:
    rng = np.random.default_rng(4)
    cache = TorqueKVCache(
        config=TorqueConfig(bit_width=4, head_dim=64),
        key_codebook=_uniform_codebook(4),
        value_codebook=_uniform_codebook(4),
    )
    for _ in range(5):
        cache.append(
            key=rng.uniform(-1.0, 1.0, size=(64,)).astype(np.float32),
            value=rng.uniform(-1.0, 1.0, size=(64,)).astype(np.float32),
        )

    query = rng.uniform(-1.0, 1.0, size=(64,)).astype(np.float32)
    output = cache.decode(query=query)
    keys, values = cache.export_dequantized()
    out_rot = streaming_attention_decode(
        cache.rotation.apply(query),
        keys[0, 0],
        values[0, 0],
    )
    expected = cache.rotation.inverse(out_rot)
    np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)
