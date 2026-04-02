# torque-mlx

`torque-mlx` is a model conversion and KV-cache optimization toolkit for MLX on Apple silicon.

The project is built around one core idea: KV-cache quantization only helps if decode attention operates directly on packed cached values instead of expanding them back into dense FP16 buffers. `torque-mlx` stores keys and values as packed quantized codes in a rotated basis and targets fused packed-code decode kernels for long-context inference.

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

The current Qwen converter:

- reads a local Hugging Face-style `safetensors` snapshot
- identifies `full_attention` layers from `config.json`
- rewrites only supported `q/k/v/o` attention projection weights
- copies non-converted tensors through unchanged
- preserves multimodal vision tensors and processor/config assets unchanged
- emits `torque_qwen_manifest.json` in the output snapshot

It assumes standard Qwen-style tensor suffixes and should be treated as a curated family workflow, not a generic HF converter.

The current Qwen evaluator:

- runs text-only perplexity over a raw text file
- loads either the original local snapshot or a converted torque snapshot
- reports safetensor size alongside perplexity
- is useful for correctness and publishability checks
- does not mean torque is already reducing full-model weight size on disk

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
