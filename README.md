# torque-mlx

`torque-mlx` is a model conversion and KV-cache optimization toolkit for MLX on Apple silicon.

The project is built around one core idea: KV-cache quantization only helps if decode attention operates directly on packed cached values instead of expanding them back into dense FP16 buffers. `torque-mlx` stores keys and values as packed quantized codes in a rotated basis and targets fused packed-code decode kernels for long-context inference.

## Current Status

The repo now demonstrates three things on real Qwen3.5 snapshots:

- converted Qwen artifacts can be loaded and evaluated end-to-end
- the packed torque KV path cuts converted-layer KV bytes by about `75%` at `4` bits
- on `Qwen3.5-2B`, the current MLX runtime is near source-speed parity on fixed-length generation while keeping that KV reduction

The most recent real `Qwen3.5-2B` MLX generation comparison in this repo, using the same `972`-token prompt and `64` generated tokens, came out to:

- source: about `15.15 tok/s`
- torque: about `14.55 tok/s`
- converted-layer KV estimate: `12.73 MB` FP16 vs `3.18 MB` packed

The most important current limitation is architectural rather than purely torque-specific: on Qwen3.5, the hybrid linear-attention layers still dominate end-to-end runtime. The converted full-attention layers are no longer the only meaningful bottleneck.

## What It Does Today

The repo currently provides:

- rotation, quantization, packing, and reference decode primitives
- a `TorqueKVCache` runtime for packed rotated KV decode
- offline attention-weight fusion
- a versioned artifact format for converted attention blocks
- a packaged CLI for conversion, inspection, benchmarking, and evaluation
- a curated Qwen workflow for planning and rewriting local Hugging Face snapshots

This is still pre-alpha infrastructure. It is useful for experiments, conversion workflows, and validation, but it is not yet a polished end-user MLX deployment stack.

## Current Support Model

Support is curated, not universal.

- Families are added explicitly.
- Unsupported architectures should fail clearly.
- Converted artifacts should be validated before being published.

At the moment, the strongest family-specific path is Qwen:

- inspect a local snapshot with `torque-mlx plan qwen`
- rewrite supported `full_attention` layers with `torque-mlx convert-qwen-model`
- inspect the converted manifest with `torque-mlx inspect-qwen-model`
- preserve multimodal Qwen vision components unchanged when `vision_config` is present

## Requirements

For conversion workflows:

- Apple Silicon Mac is the intended platform
- Python `>= 3.11`
- `numpy`
- `safetensors`

For MLX runtime and packed-kernel benchmarks:

- Apple Silicon
- `mlx`
- Xcode / Metal toolchain available to MLX

For Qwen text perplexity evaluation:

- `torch`
- a `transformers` build with native `qwen3_5` support
- a local Qwen snapshot plus a raw text file such as `wiki.test.raw`

The current runtime envelope supports:

- head dimensions: `64`, `128`, `256`
- bit widths: `2`, `3`, `4`
- structured Hadamard rotation
- batch-1 decode oriented flows

## Install

From the repo root:

```bash
pip install -e .
```

For tests:

```bash
pip install -e .[dev]
```

For Qwen text perplexity evaluation:

```bash
pip install -e .[qwen-eval]
```

## Quick Start

Convert a generic attention checkpoint:

```bash
torque-mlx convert \
  --input-weights ./attention.npz \
  --output-dir ./artifacts/tiny-attention \
  --model-name tiny-attention \
  --bit-width 4

torque-mlx inspect --artifact ./artifacts/tiny-attention
```

Run the reference benchmark:

```bash
torque-mlx benchmark synthetic \
  --seq-len 512 \
  --head-dim 128 \
  --kv-heads 8 \
  --bit-width 4
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

Benchmark the MLX decode hot path using real Qwen geometry:

```bash
torque-mlx benchmark qwen-decode \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prefill-tokens 2048 \
  --decode-steps 128 \
  --decode-strategy split_batched
```

Evaluate an artifact:

```bash
torque-mlx eval --artifact ./artifacts/tiny-attention --seq-len 512
```

Programmatic use:

```python
from torque_mlx import load_torque_artifact

artifact = load_torque_artifact("./artifacts/tiny-attention")
cache = artifact.build_cache()
```

## Qwen Workflow

Inspect a local Qwen snapshot:

```bash
torque-mlx plan qwen --model-dir /path/to/qwen-snapshot
```

Convert a full local Qwen snapshot:

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen-torque \
  --model-name qwen-torque
```

Or emit a smaller delta artifact that stores only converted tensor overrides and references the source snapshot:

```bash
torque-mlx convert-qwen-model \
  --model-dir /path/to/qwen-snapshot \
  --output-dir ./artifacts/qwen-torque-delta \
  --model-name qwen-torque-delta \
  --artifact-layout delta_npz
```

Inspect the converted Qwen manifest:

```bash
torque-mlx inspect-qwen-model --artifact ./artifacts/qwen-torque
```

Run text perplexity evaluation against a local source or converted snapshot:

```bash
torque-mlx eval-qwen-text \
  --model-dir ./artifacts/qwen-torque \
  --text-file ./wiki.test.raw \
  --context-length 2048 \
  --stride 2048
```

Run an experimental MLX generation pass against a local source or converted artifact:

```bash
torque-mlx benchmark qwen-generate \
  --model-dir ./artifacts/qwen3.5-0.8b-torque \
  --prompt "hello" \
  --max-tokens 32 \
  --prefill-step-size 512 \
  --ignore-eos \
  --profile-runtime
```

Compare the end-to-end MLX generation path against the matching synthetic KV decode hot path:

```bash
torque-mlx benchmark qwen-runtime-compare \
  --model-dir ./artifacts/qwen3.5-2b-torque-delta \
  --prompt "hello" \
  --max-tokens 128 \
  --prefill-step-size 128 \
  --ignore-eos
```

The current Qwen converter:

- reads a local Hugging Face-style `safetensors` snapshot
- identifies `full_attention` layers from `config.json`
- rewrites only supported `q/k/v/o` attention projection weights
- can either rewrite a full merged snapshot or emit a delta artifact with only converted tensor overrides
- copies non-converted tensors through unchanged when writing a merged snapshot
- preserves multimodal vision tensors and processor/config assets unchanged
- emits `torque_qwen_manifest.json` in the output snapshot

It assumes standard Qwen-style tensor suffixes and should be treated as a curated family workflow, not a generic HF converter.

The current Qwen evaluator:

- runs text-only perplexity over a raw text file
- loads either the original local snapshot or a converted torque snapshot
- applies the matching torque runtime correction when a converted Qwen manifest is present
- can load a `delta_npz` torque artifact by applying stored override tensors on top of the referenced source model
- reports safetensor size alongside perplexity
- reports loader time, evaluation time, and evaluated tokens per second
- is useful for correctness and publishability checks
- does not mean torque is already reducing full-model weight size on disk

The Qwen text benchmark:

- compares a source snapshot against a converted torque snapshot
- runs the same workload at one or more context lengths
- reports perplexity, size, and timing deltas per context length
- is the current correctness and model-behavior check, not the main MLX performance benchmark

The experimental Qwen MLX generation benchmark:

- loads local `qwen3_5` snapshots through a repo-local MLX adapter that normalizes them onto the `qwen3_next` text runtime
- can load both merged converted snapshots and `delta_npz` artifacts
- can optionally synchronize and report converted-layer dense prefill, prompt append, decode append, aggregate append, and torque decode timings via `--profile-runtime`
- also breaks profiling into converted full-attention layer time, linear-layer time, and optional passthrough-attention time
- further breaks torque decode into packed score, softmax/merge, packed value accumulation, and dense-tail work when profiling is enabled
- uses dense cache prefill for hybrid Qwen3.5 prompt chunks, then routes converted `full_attention` decode through `TorqueKVCacheMLX`
- buffers single-token decode appends through a small dense tail before flushing them into packed storage
- can override that tail size through `--decode-tail-capacity`; by default the runtime now chooses it automatically from the Qwen text hidden size (`8` for smaller models like `0.8B`, `0` for `2B`-class models)
- reports prompt throughput, generation throughput, generated token count, peak memory, and explicit converted-layer KV estimates for FP16 bytes, packed bytes, and bytes saved
- supports fixed-length decode benchmarking with `--ignore-eos`, so source-vs-torque runs can compare the same number of generated tokens
- is the first end-to-end MLX runtime path for converted Qwen artifacts in this repo
- should still be treated as experimental until longer prompts and larger models are benchmarked systematically

The Qwen runtime comparison benchmark:

- runs `qwen-generate` first on a converted torque artifact
- reuses the observed prompt token count and generated token count to run the synthetic `qwen-decode` hot-path benchmark on matching geometry
- reports the gap between `hot_path_tokens_per_sec` and end-to-end `generation_tokens_per_sec`
- nests both the generation result and the synthetic decode result so the remaining runtime overhead is explicit
- now makes it clear whether the remaining gap is mostly inside converted torque layers or in the rest of the model/runtime

The Qwen decode benchmark:

- derives head dimension, KV heads, and converted-layer count from a local Qwen snapshot or torque manifest
- models grouped-query attention using the source model's `num_attention_heads` to `num_key_value_heads` ratio
- benchmarks the KV-growing autoregressive decode path on MLX
- compares MLX-LM FP16 cache decode, MLX-LM quantized cache decode, and `TorqueKVCache`
- reports projected KV cache bytes against FP16 so the memory story is explicit
- breaks timing into cache update/append, decode compute, and residual host overhead
- is the benchmark that lines up most directly with the repo's core performance thesis

## How It Works

`torque-mlx` combines three ideas:

- Basis-native attention: decode runs directly on packed quantized KV codes in rotated space.
- Offline weight fusion: the shared rotation can be fused into `W_Q`, `W_K`, `W_V`, and `W_O`.
- Structured rotations: Hadamard-based transforms keep the method compatible with Apple GPU-friendly execution patterns.

The intended hot path is:

1. Rotate the fresh query.
2. Read packed cached keys and values.
3. Compute attention scores from centroid lookups.
4. Run numerically stable streaming softmax.
5. Accumulate rotated values.
6. Optionally avoid inverse rotation with fused output weights.

## Scope and Limits

The repo is focused on decoder-only transformer inference through MLX on Apple silicon.

In scope now:

- packed rotated KV cache primitives
- offline fusion tooling
- generic attention artifacts
- curated Qwen snapshot rewriting
- synthetic and MLX benchmark harnesses

Not in scope yet:

- universal model-family conversion
- production-ready MLX loader integration for every converted model
- arbitrary head dimensions
- non-Apple backends

## Repository Guide

- [docs/cli.md](./docs/cli.md): command reference
- [docs/contracts/model-artifact.md](./docs/contracts/model-artifact.md): generic artifact contract
- [docs/families/qwen.md](./docs/families/qwen.md): curated Qwen workflow
- [docs/runtime-status.md](./docs/runtime-status.md): current real-model runtime status and bottlenecks
- [docs/model-support.md](./docs/model-support.md): support matrix and limits
- [docs/architecture/runtime-boundary.md](./docs/architecture/runtime-boundary.md): Python vs compiled hot-path boundary
- [PRD.md](./PRD.md): product requirements
- [TASKS.md](./TASKS.md): implementation backlog

## References

Upstream ecosystems:

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [Qwen on Hugging Face](https://huggingface.co/Qwen)

Design references:

- [QuaRot](https://arxiv.org/abs/2404.00456)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [RotateKV](https://arxiv.org/abs/2501.16383)

## License

The `torque-mlx` code in this repository is MIT-licensed. Converted model artifacts remain subject to the licenses and usage terms of their upstream model sources.
