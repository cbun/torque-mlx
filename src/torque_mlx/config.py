from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TorqueConfig:
    """Top-level runtime configuration for torque-mlx."""

    bit_width: int = 4
    head_dim: int = 128
    fused_weights: bool = False
    use_hadamard_rotation: bool = True

    def validate(self) -> None:
        if self.bit_width not in {2, 3, 4}:
            raise ValueError(f"Unsupported bit width: {self.bit_width}")
        if self.head_dim not in {64, 128}:
            raise ValueError(f"Unsupported head dimension: {self.head_dim}")

