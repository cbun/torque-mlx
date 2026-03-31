import pytest

from torque_mlx.quantization import packed_words_for_head_dim


@pytest.mark.parametrize(
    ("head_dim", "bit_width", "expected_words"),
    [
        (64, 2, 4),
        (64, 3, 6),
        (64, 4, 8),
        (128, 2, 8),
        (128, 3, 12),
        (128, 4, 16),
    ],
)
def test_packed_words_for_head_dim(
    head_dim: int,
    bit_width: int,
    expected_words: int,
) -> None:
    assert packed_words_for_head_dim(head_dim, bit_width) == expected_words

