# Torque Model Artifact Contract

## Purpose

`torque-mlx` artifacts package the offline-converted attention weights together with the runtime metadata needed to execute the model on the torque decode path.

The artifact is the boundary between:

- offline conversion
- runtime cache selection
- benchmark and evaluation tooling

## Directory Layout

The v1 artifact is a directory with:

- `manifest.json`: versioned metadata and runtime profile
- `weights.npz`: fused attention weights

## Manifest Fields

Required fields:

- `format_name`: must be `torque-mlx-artifact`
- `format_version`: currently `1`
- `model_name`
- `architecture`
- `source_format`
- `runtime_config`
- `key_codebook`
- `value_codebook`
- `weights_file`
- `weight_names`

## Runtime Config

The runtime profile must contain:

- `bit_width`
- `head_dim`
- `num_layers`
- `kv_heads`
- `fused_weights`
- `rotation_mode`
- `rotation_seed`

For v1 artifacts, `fused_weights` must be `true`.

## Weight Payload

The default `weights.npz` file stores:

- `w_q`
- `w_k`
- `w_v`
- `w_o`

These tensors are already fused with the shared rotation:

- `W_Q' = Pi W_Q`
- `W_K' = Pi W_K`
- `W_V' = Pi W_V`
- `W_O' = W_O Pi^T`

## Codebooks

Artifacts carry both key and value codebooks so the runtime can build a matching `TorqueKVCache` without requiring side-channel configuration.

Codebooks must follow the same constraints as the runtime contract:

- sorted ascending centroids
- `float32` centroid storage
- `2^bit_width` entries

## Compatibility

Artifacts are v1-compatible only when:

- `head_dim` is `64`, `128`, or `256`
- `bit_width` is `2`, `3`, or `4`
- `rotation_mode` is `hadamard`

Unsupported configurations must fail during load or config validation.
