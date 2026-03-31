from torque_mlx import TorqueConfig, TorqueKVCache


def test_cache_scaffold_imports() -> None:
    config = TorqueConfig(bit_width=4, head_dim=128)
    cache = TorqueKVCache(config=config)
    assert cache.sequence_length == 0

