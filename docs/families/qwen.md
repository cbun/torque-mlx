# Qwen Family Workflow

`torque-mlx` treats Qwen support as a curated family workflow, not a generic architecture promise.

## What To Inspect

Inspect the local Hugging Face snapshot first:

```bash
torque-mlx plan qwen --model-dir /path/to/qwen-snapshot
```

The planner reads `config.json` and reports:

- `head_dim`
- `num_hidden_layers`
- `num_key_value_heads`
- `layer_types`
- `full_attention_indices`
- blocking issues that prevent safe conversion

## Conversion Rule

For Qwen, only `full_attention` layers are candidates for torque conversion.

- `full_attention`: potential torque target
- `linear_attention`: copy through unchanged

If `head_dim` is not `64`, `128`, or `256`, the planner must fail the runtime support check because the current rotation and kernel path only supports those dimensions.

## Manual Conversion Workflow

1. Run the planner on the exact checkpoint you want to convert.
2. Extract `w_q`, `w_k`, `w_v`, and `w_o` for one supported `full_attention` layer into an `.npz` file.
3. Convert that layer with:

```bash
torque-mlx convert-qwen-layer \
  --model-dir /path/to/qwen-snapshot \
  --layer-idx 3 \
  --input-weights ./layer3.npz \
  --output-dir ./artifacts/qwen-layer3
```

4. Validate the resulting artifact before publishing.

For a full local snapshot rewrite, use:

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen3.5-27b-torque \
  --model-name qwen3.5-27b-torque
```

This command:

- reads the local `safetensors` shard map
- rewrites only the `full_attention` layer projections
- copies non-attention tensors through unchanged
- emits `torque_qwen_manifest.json` in the output snapshot

## Publishing Guidance

Only publish artifacts for model revisions you personally inspected and validated. Support is explicit per converted artifact, not inherited across all Qwen models.
