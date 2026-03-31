import numpy as np

from torque_mlx.conversion import fuse_attention_weights
from torque_mlx.rotation import RotationSpec


def test_weight_fusion_matches_explicit_rotation() -> None:
    rng = np.random.default_rng(0)
    head_dim = 64
    hidden_dim = 32
    output_dim = 48
    rotation = RotationSpec.from_seed(head_dim=head_dim, seed=13)

    w_q = rng.normal(size=(head_dim, hidden_dim)).astype(np.float32)
    w_k = rng.normal(size=(head_dim, hidden_dim)).astype(np.float32)
    w_v = rng.normal(size=(head_dim, hidden_dim)).astype(np.float32)
    w_o = rng.normal(size=(output_dim, head_dim)).astype(np.float32)
    hidden = rng.normal(size=(hidden_dim,)).astype(np.float32)
    attn = rng.normal(size=(head_dim,)).astype(np.float32)

    fused = fuse_attention_weights(
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        rotation=rotation,
    )

    np.testing.assert_allclose(fused.w_q @ hidden, rotation.apply(w_q @ hidden), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(fused.w_k @ hidden, rotation.apply(w_k @ hidden), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(fused.w_v @ hidden, rotation.apply(w_v @ hidden), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(fused.w_o @ attn, w_o @ rotation.inverse(attn), atol=1e-4, rtol=1e-4)
