import numpy as np
import pytest

from torque_mlx.rotation import (
    RotationSpec,
    apply_structured_rotation,
    apply_structured_rotation_mlx,
    inverse_structured_rotation_mlx,
)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_rotation_inverse_roundtrip(head_dim: int) -> None:
    rng = np.random.default_rng(0)
    vector = rng.normal(size=head_dim).astype(np.float32)
    spec = RotationSpec.from_seed(head_dim=head_dim, seed=11)
    restored = spec.inverse(spec.apply(vector))
    np.testing.assert_allclose(restored, vector, rtol=1e-5, atol=1e-5)


def test_rotation_preserves_norm() -> None:
    rng = np.random.default_rng(1)
    vector = rng.normal(size=256).astype(np.float32)
    rotated = apply_structured_rotation(vector, head_dim=256, seed=5)
    np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(vector), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_mlx_rotation_matches_numpy_rotation(head_dim: int) -> None:
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 2, head_dim)).astype(np.float32)
    spec = RotationSpec.from_seed(head_dim=head_dim, seed=13)

    rotated = apply_structured_rotation_mlx(
        mx.array(values),
        signs_left=mx.array(spec.signs_left),
        signs_right=mx.array(spec.signs_right),
    )
    restored = inverse_structured_rotation_mlx(
        rotated,
        signs_left=mx.array(spec.signs_left),
        signs_right=mx.array(spec.signs_right),
    )
    mx.eval(rotated, restored)

    np.testing.assert_allclose(np.array(rotated), spec.apply(values), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.array(restored), values, atol=1e-5, rtol=1e-5)
