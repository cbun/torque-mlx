from dataclasses import dataclass

from torque_mlx.quantization import packed_words_for_head_dim


def build_variant_id(
    *,
    bit_width: int,
    head_dim: int,
    fused_weights: bool,
    rotation_mode: str,
) -> str:
    fused = "fused" if fused_weights else "unfused"
    return f"b{bit_width}-h{head_dim}-{rotation_mode}-{fused}"


@dataclass(frozen=True, slots=True)
class PackedKVLayout:
    bit_width: int
    head_dim: int

    @property
    def packed_words(self) -> int:
        return packed_words_for_head_dim(self.head_dim, self.bit_width)

    @property
    def packed_bytes(self) -> int:
        return self.packed_words * 4

    @property
    def kv_bytes_per_token_per_head(self) -> int:
        return self.packed_bytes * 2

    def tensor_shape(self, *, num_layers: int, kv_heads: int, seq_len: int) -> tuple[int, int, int, int]:
        return (num_layers, kv_heads, seq_len, self.packed_words)


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    variant_id: str
    bit_width: int
    head_dim: int
    num_layers: int
    kv_heads: int
    rotation_mode: str
    fused_weights: bool
    codebook_name: str
