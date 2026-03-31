from torque_mlx.config import TorqueConfig


def fuse_model_weights(model, config: TorqueConfig):
    """Offline rotation/weight-fusion entrypoint placeholder."""
    config.validate()
    del model
    raise NotImplementedError("Offline weight fusion is not implemented yet.")

