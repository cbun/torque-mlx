from enum import StrEnum


class RotationMode(StrEnum):
    HADAMARD = "hadamard"


def validate_head_dim(head_dim: int) -> None:
    if head_dim not in {64, 128}:
        raise ValueError(f"Unsupported head dimension for rotation: {head_dim}")

