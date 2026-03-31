from torque_mlx import TorqueConfig, TorqueKVCache, fuse_attention_weights


def test_cache_scaffold_imports() -> None:
    config = TorqueConfig(bit_width=4, head_dim=128)
    cache = TorqueKVCache(config=config)
    assert cache.sequence_length == 0
    assert callable(fuse_attention_weights)
