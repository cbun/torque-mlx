import numpy as np

from torque_mlx.quantization import build_gaussian_codebook, dequantize, quantize
from torque_mlx.rotation import RotationSpec


def test_rotated_quantization_error_stays_bounded_on_synthetic_vectors() -> None:
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(256, 128)).astype(np.float32)
    spec = RotationSpec.from_seed(head_dim=128, seed=3)
    codebook = build_gaussian_codebook(4, sample_size=50_000, iterations=16, seed=2)

    rotated = spec.apply(vectors)
    reconstructed = dequantize(quantize(rotated, codebook), codebook)
    mse = float(np.mean((rotated - reconstructed) ** 2))

    assert mse < 0.05
