from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelSpec:
    bit_width: int
    head_dim: int
    fused_weights: bool = False

