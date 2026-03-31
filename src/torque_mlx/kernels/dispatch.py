from torque_mlx.config import TorqueConfig
from torque_mlx.kernels import KernelSpec


def select_decode_kernel(config: TorqueConfig) -> KernelSpec:
    config.validate()
    return KernelSpec(
        bit_width=config.bit_width,
        head_dim=config.head_dim,
        fused_weights=config.fused_weights,
        pack_mode="packed",
    )
