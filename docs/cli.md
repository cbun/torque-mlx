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

Convert a local Qwen Hugging Face snapshot into either a merged torque snapshot or a smaller delta artifact with rewritten `full_attention` weights and a `torque_qwen_manifest.json`.

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen3.5-27b-torque \
  --model-name qwen3.5-27b-torque
```

To emit a delta artifact instead of a full rewritten snapshot:

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen3.5-27b-torque-delta \
  --model-name qwen3.5-27b-torque-delta \
  --artifact-layout delta_npz
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
- it reports text perplexity, safetensor size, and basic wall-clock timing
- when `torque_qwen_manifest.json` is present, it applies the matching torque runtime correction before evaluation
- when the manifest layout is `delta_npz`, it loads the referenced source model and applies the stored override tensors before evaluation
- it is intended for quality validation of curated Qwen artifacts, not for MLX runtime benchmarking

### `benchmark`

Available benchmark modes:

- `synthetic`
- `mlx-packed`
- `mlx-lm`
- `qwen-text`
- `qwen-decode`
- `qwen-generate`

Example:

```bash
torque-mlx benchmark synthetic --seq-len 512 --head-dim 128 --kv-heads 8 --bit-width 4
```

Compare a source and converted Qwen snapshot on the same text workload:

```bash
torque-mlx benchmark qwen-text \
  --source-model-dir ./artifacts/qwen3.5-0.8b-source \
  --torque-model-dir ./artifacts/qwen3.5-0.8b-torque \
  --text-file ./artifacts/wiki.test.raw \
  --context-length 512 \
  --context-length 2048 \
  --max-tokens 2048
```

This report includes, per context length:

- source and torque perplexity
- source and torque safetensor size
- source and torque loader/evaluation/total wall-clock time
- evaluated tokens per second
- source-vs-torque deltas for those metrics

Benchmark the MLX decode hot path using real Qwen geometry:

```bash
torque-mlx benchmark qwen-decode \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prefill-tokens 2048 \
  --decode-steps 128 \
  --decode-strategy split_batched
```

This report includes:

- the derived Qwen geometry and converted-layer count
- the grouped-query ratio between attention heads and KV heads
- the requested torque decode strategy (`split_batched` by default; `auto` currently aliases it)
- projected FP16 vs torque KV cache bytes
- MLX-LM FP16 decode timing
- MLX-LM quantized decode timing
- `TorqueKVCache` decode timing
- timing breakdowns for cache update/append, decode compute, and residual host overhead
- hot-path numerical error against the FP16 baseline

This is the benchmark mode aligned with the repo's long-context decode thesis. Unlike `qwen-text`, it does not run full dense forwards or report perplexity.

Run the experimental end-to-end MLX Qwen generation path:

```bash
torque-mlx benchmark qwen-generate \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prompt "hello" \
  --max-tokens 32 \
  --prefill-step-size 512 \
  --profile-runtime
```

This report includes:

- whether the loaded artifact is torque-converted
- the artifact layout (`merged_snapshot` or `delta_npz`)
- converted full-attention layer indices from the manifest
- prompt tokens and prompt tokens/sec
- generated tokens and generation tokens/sec
- peak MLX memory during the run
- optional converted-layer timing breakdowns for dense prefill, prompt append, decode append, aggregate append, and torque decode when `--profile-runtime` is set

This command exercises the repo-local MLX adapter for `qwen3_5` snapshots. It is the current end-to-end generation smoke path for converted Qwen artifacts, but it should still be treated as experimental.

### `eval`

Evaluate an artifact with the synthetic decode harness using the artifact's runtime profile.

```bash
torque-mlx eval --artifact ./artifacts/tiny-attention --seq-len 512
```
