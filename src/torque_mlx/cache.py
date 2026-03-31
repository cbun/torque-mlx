from dataclasses import dataclass, field

from torque_mlx.config import TorqueConfig


@dataclass(slots=True)
class TorqueKVCache:
    """Scaffold for the rotation-native KV cache runtime."""

    config: TorqueConfig = field(default_factory=TorqueConfig)
    sequence_length: int = 0

    def __post_init__(self) -> None:
        self.config.validate()

    def append(self, *, key, value) -> None:
        """Append one decode step worth of KV data.

        This is a placeholder contract. The real implementation will rotate,
        quantize, pack, and stage data for fused decode kernels.
        """
        del key, value
        self.sequence_length += 1

    def reset(self) -> None:
        self.sequence_length = 0

