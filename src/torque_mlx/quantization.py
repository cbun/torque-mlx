from math import ceil


SUPPORTED_BIT_WIDTHS = {2, 3, 4}


def packed_words_for_head_dim(head_dim: int, bit_width: int) -> int:
    """Return the number of 32-bit words required for one packed vector."""
    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"Unsupported bit width: {bit_width}")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return ceil((head_dim * bit_width) / 32)

