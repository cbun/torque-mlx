from dataclasses import dataclass

from torque_mlx.layout import PackedKVLayout, build_variant_id
from torque_mlx.quantization import SUPPORTED_BIT_WIDTHS
from torque_mlx.rotation import RotationMode, validate_head_dim


@dataclass(frozen=True, slots=True)
class TorqueConfig:
    """Top-level runtime configuration for torque-mlx."""

    bit_width: int = 4
    head_dim: int = 128
    num_layers: int = 1
    kv_heads: int = 1
    fused_weights: bool = False
    rotation_mode: RotationMode = RotationMode.HADAMARD
    rotation_seed: int = 0

    def validate(self) -> None:
        if self.bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"Unsupported bit width: {self.bit_width}")
        validate_head_dim(self.head_dim)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")

    @property
    def layout(self) -> PackedKVLayout:
        self.validate()
        return PackedKVLayout(bit_width=self.bit_width, head_dim=self.head_dim)

    @property
    def variant_id(self) -> str:
        self.validate()
        return build_variant_id(
            bit_width=self.bit_width,
            head_dim=self.head_dim,
            fused_weights=self.fused_weights,
            rotation_mode=self.rotation_mode.value,
        )
