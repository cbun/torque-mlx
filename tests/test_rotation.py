import numpy as np
import pytest

from torque_mlx.rotation import RotationSpec, apply_structured_rotation


@pytest.mark.parametrize("head_dim", [64, 128])
def test_rotation_inverse_roundtrip(head_dim: int) -> None:
    rng = np.random.default_rng(0)
    vector = rng.normal(size=head_dim).astype(np.float32)
    spec = RotationSpec.from_seed(head_dim=head_dim, seed=11)
    restored = spec.inverse(spec.apply(vector))
    np.testing.assert_allclose(restored, vector, rtol=1e-5, atol=1e-5)


def test_rotation_preserves_norm() -> None:
    rng = np.random.default_rng(1)
    vector = rng.normal(size=128).astype(np.float32)
    rotated = apply_structured_rotation(vector, head_dim=128, seed=5)
    np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(vector), rtol=1e-5, atol=1e-5)

