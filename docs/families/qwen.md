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
- vision and other non-text components: copy through unchanged

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
- preserves multimodal Qwen snapshots by copying the vision stack and processor assets through unchanged
- emits `torque_qwen_manifest.json` in the output snapshot

If you want a smaller converted artifact that stores only the rewritten tensors, use the delta layout:

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen3.5-27b-torque-delta \
  --model-name qwen3.5-27b-torque-delta \
  --artifact-layout delta_npz
```

The delta layout:

- stores only converted tensor overrides in `torque_qwen_delta_weights.npz`
- copies non-weight assets such as config/tokenizer files
- keeps a manifest pointer back to the original source snapshot
- is smaller on disk than a full rewritten snapshot, but still depends on the base source model to run

For multimodal Qwen snapshots such as `Qwen3.5-0.8B`, the current support boundary is:

- text `full_attention` layers are converted
- text `linear_attention` layers are copied through
- vision weights and configs are copied through
- the converter emits metadata indicating that the source snapshot had a vision stack

That is curated conversion support, not a claim that the repository already provides a finished end-to-end MLX multimodal runtime.

## Text Evaluation Workflow

For text-only quality checks on a local source or converted snapshot, run:

```bash
torque-mlx eval-qwen-text \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --text-file ./wiki.test.raw \
  --context-length 2048 \
  --stride 2048
```

The evaluator:

- tokenizes the raw text file
- runs sliding-window perplexity over the local snapshot
- applies the matching torque runtime correction for converted `vo_only_runtime_qk_rotation` layers
- can load `delta_npz` artifacts by applying the stored overrides to the referenced source model first
- reports safetensor size in bytes and GiB
- reports loader time, evaluation time, and evaluated tokens per second
- includes `torque_qwen_manifest.json` metadata when present

This is a publishability and correctness check for curated Qwen artifacts. It does not imply that the repository already has a production MLX runtime for converted Qwen snapshots, and it does not yet produce a smaller full-model weight format on disk.

## Experimental MLX Runtime

For an end-to-end MLX generation smoke on a local source or converted artifact, run:

```bash
torque-mlx benchmark qwen-generate \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prompt "hello" \
  --max-tokens 32 \
  --prefill-step-size 512 \
  --ignore-eos \
  --profile-runtime
```

The current adapter:

- normalizes local `qwen3_5` snapshots onto a repo-local MLX `qwen3_next` compatibility model
- supports the hybrid Qwen3.5 text stack, including `linear_attention` passthrough layers and converted `full_attention` layers
- can load merged snapshots directly
- can load `delta_npz` artifacts by loading the referenced base snapshot and applying the stored overrides first
- uses dense-cache prompt prefill for converted full-attention layers, then switches those layers onto `TorqueKVCacheMLX` for single-token decode
- buffers single-token decode appends through a small dense tail before flushing them into packed storage
- exposes the tail size as `--decode-tail-capacity`; if omitted, the runtime now chooses a Qwen-specific default from the text hidden size
- can emit runtime timing breakdowns with `--profile-runtime` to separate converted full-attention layers, linear layers, passthrough attention layers, dense prefill, prompt append, decode append, aggregate append, torque decode, packed score, softmax/merge, packed value accumulation, and dense-tail costs
- supports fixed-length decode benchmarking with `--ignore-eos`, which keeps generation running to `max_tokens` even if EOS is emitted early
- reports explicit converted-layer KV estimates for cache tokens, FP16 bytes, packed bytes, and bytes saved so the memory story is visible even when total process memory is dominated by model weights

This is intentionally narrow. It is an experimental generation/runtime path for curated Qwen3.5 artifacts, not a generic MLX loader for arbitrary Hugging Face checkpoints.

For a direct source-vs-torque comparison, use:

```bash
torque-mlx benchmark qwen-text \
  --source-model-dir ./artifacts/qwen3.5-0.8b-source \
  --torque-model-dir ./artifacts/qwen3.5-0.8b-torque \
  --text-file ./artifacts/wiki.test.raw \
  --context-length 512 \
  --context-length 2048 \
  --max-tokens 2048
```

This benchmark is the current model-level report for curated Qwen artifacts. It shows whether the converted snapshot preserves perplexity, whether its on-disk size changed, and whether the current evaluation runtime got faster or slower.

For a decode-path benchmark aligned with `torque-mlx`'s actual thesis, use:

```bash
torque-mlx benchmark qwen-decode \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prefill-tokens 2048 \
  --decode-steps 128 \
  --decode-strategy split_batched
```

This benchmark:

- derives the decode geometry from the local Qwen snapshot or torque manifest
- benchmarks only the autoregressive KV-growing decode path on MLX
- compares MLX-LM FP16, MLX-LM quantized cache, and `TorqueKVCache`
- defaults to the batched split kernel and can still force the prior per-head fused kernel for comparison
- resolves the dense decode-tail size automatically from the Qwen text hidden size unless `--decode-tail-capacity` is set explicitly
- reports projected KV cache bytes saved against FP16

It is the right benchmark for answering whether torque improves the cache hot path. The text benchmark above is still useful, but it is a correctness check rather than a decode-kernel performance claim.

To compare that packed hot path against the real end-to-end MLX runtime, run:

```bash
torque-mlx benchmark qwen-runtime-compare \
  --model-dir ./artifacts/qwen3.5-2b-torque-delta \
  --prompt "hello" \
  --max-tokens 128 \
  --prefill-step-size 128 \
  --ignore-eos
```

This mode runs `qwen-generate` first, then reuses the observed prompt and generated token counts for `qwen-decode`. The result makes the remaining gap between the packed decode path and the full MLX Qwen runtime explicit.

## Publishing Guidance

Only publish artifacts for model revisions you personally inspected and validated. Support is explicit per converted artifact, not inherited across all Qwen models.
