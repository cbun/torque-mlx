from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelSpec:
    bit_width: int
    head_dim: int
    fused_weights: bool = False
    pack_mode: str = "packed"

    @property
    def variant_id(self) -> str:
        fused = "fused" if self.fused_weights else "unfused"
        return f"b{self.bit_width}-h{self.head_dim}-{self.pack_mode}-{fused}"
