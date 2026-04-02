# CLI

`torque-mlx` ships a small CLI intended to grow into the main user-facing surface for conversion, benchmarking, and evaluation.

## Commands

### `convert`

Fuse attention weights from an `.npz` checkpoint and write a torque artifact directory.

Expected arrays in the checkpoint:

- `w_q`
- `w_k`
- `w_v`
- `w_o`

Example:

```bash
torque-mlx convert \
  --input-weights ./attention.npz \
  --output-dir ./artifacts/tiny-attention \
  --model-name tiny-attention \
  --bit-width 4
```

### `inspect`

Print JSON metadata for a converted artifact.

```bash
torque-mlx inspect --artifact ./artifacts/tiny-attention
```

### `plan qwen`

Inspect a local Qwen Hugging Face snapshot and emit a curated conversion report.

```bash
torque-mlx plan qwen --model-dir /path/to/qwen-snapshot
```

### `convert-qwen-layer`

Convert one extracted Qwen `full_attention` layer from `.npz` into a torque artifact.

```bash
torque-mlx convert-qwen-layer \
  --model-dir /path/to/qwen-snapshot \
  --layer-idx 3 \
  --input-weights ./layer3.npz \
  --output-dir ./artifacts/qwen-layer3
```

### `convert-qwen-model`

Convert a local Qwen Hugging Face snapshot into a merged torque snapshot with rewritten `full_attention` weights and a `torque_qwen_manifest.json`.

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen3.5-27b-torque \
  --model-name qwen3.5-27b-torque
```

### `inspect-qwen-model`

Inspect the manifest for a converted Qwen torque snapshot.

```bash
torque-mlx inspect-qwen-model --artifact ./artifacts/qwen3.5-27b-torque
```

### `eval-qwen-text`

Run text perplexity evaluation on a local Qwen snapshot or a converted torque snapshot.

```bash
torque-mlx eval-qwen-text \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --text-file ./wiki.test.raw \
  --context-length 2048 \
  --stride 2048
```

Notes:

- this command requires `torch` and a `transformers` build that can load `qwen3_5`
- it reports text perplexity and safetensor size
- it is intended for quality validation of curated Qwen artifacts, not for MLX runtime benchmarking

### `benchmark`

Available benchmark modes:

- `synthetic`
- `mlx-packed`
- `mlx-lm`

Example:

```bash
torque-mlx benchmark synthetic --seq-len 512 --head-dim 128 --kv-heads 8 --bit-width 4
```

### `eval`

Evaluate an artifact with the synthetic decode harness using the artifact's runtime profile.

```bash
torque-mlx eval --artifact ./artifacts/tiny-attention --seq-len 512
```
