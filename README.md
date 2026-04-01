# torque-mlx

`torque-mlx` is a model conversion and KV-cache optimization toolkit for MLX on Apple silicon.

It is built for the failure mode that makes many quantized KV caches disappointing in practice: they save bits at rest, then give the bandwidth back by dequantizing cached keys and values into dense FP16 buffers during decode. `torque-mlx` avoids that path entirely. Keys and values are stored as packed quantized codes in a rotated basis, and decode attention runs directly on those packed codes inside fused Metal kernels.

This repository contains original `torque-mlx` code and design documents, but it is intentionally built for the Apple MLX ecosystem and interoperates with upstream model families such as Qwen through curated conversion workflows. That means the README should distinguish between:

- original project code and docs in this repository
- upstream runtimes and model ecosystems this project targets
- related research that informed the design

The product direction is not just “a cache class.” The goal is an end-to-end workflow:

- convert supported models into torque-aware artifacts
- load those artifacts with a matching runtime profile
- run long-context decode through fused packed-code kernels
- benchmark and evaluate the result against baseline MLX paths

The result is a deployment-oriented toolkit designed to make quantization pay off where it matters most on Apple hardware: long-context decode, where unified-memory bandwidth becomes the bottleneck.

## Current Status

The repository is still pre-alpha, but it now has the beginnings of the intended product surface:

- a versioned converted-model artifact format
- a `torque-mlx` CLI with `convert`, `benchmark`, `eval`, and `inspect`
- a runtime cache implementation for packed rotated KV decode
- offline attention-weight fusion utilities

Today, the converter targets generic attention checkpoints stored as `.npz` files with `w_q`, `w_k`, `w_v`, and `w_o` arrays. That is a foundation for future MLX/Hugging Face model-specific adapters, not the final UX.

The intended support model is curated, not universal:

- explicit family workflows for models you have actually inspected
- explicit failure on unsupported architectures or unsupported head dimensions
- publish validated converted artifacts, not blanket compatibility claims

## Intended Workflow

The intended user flow is:

1. Convert a supported checkpoint into a torque artifact.
2. Load the artifact and use its runtime profile for decode.
3. Benchmark and evaluate the artifact against baseline MLX paths.

Example:

```bash
torque-mlx convert \
  --input-weights ./attention.npz \
  --output-dir ./artifacts/tiny-attention \
  --model-name tiny-attention \
  --bit-width 4

torque-mlx inspect --artifact ./artifacts/tiny-attention

torque-mlx benchmark synthetic --seq-len 512 --head-dim 128 --kv-heads 8 --bit-width 4

torque-mlx eval --artifact ./artifacts/tiny-attention --seq-len 512
```

You can also load the artifact programmatically:

```python
from torque_mlx import load_torque_artifact

artifact = load_torque_artifact("./artifacts/tiny-attention")
cache = artifact.build_cache()
```

## Why It Exists

On Apple silicon, long-context inference is often limited by KV-cache reads rather than raw compute. The KV cache grows linearly with sequence length, and every decode step rereads prior keys and values. In that regime:

- FP16 KV caches consume too much bandwidth and memory.
- Naive quantized KV caches often regress on speed because they dequantize on fetch.
- Apple GPU performance depends heavily on data layout, SIMD-group shape, register pressure, and avoiding extra passes through memory.

`torque-mlx` is designed around that systems constraint rather than treating KV quantization as only a math problem.

## Core Idea

The project uses a shared orthogonal rotation to move queries, keys, and values into a basis that is easier to quantize. Attention is invariant under that shared rotation:

`q^T k = (Pi q)^T (Pi k)`

That means attention scoring and value accumulation can happen directly in the rotated basis:

- rotate the fresh query once
- keep cached keys and values packed in rotated coordinates
- compute dot products from packed indices via centroid lookup
- run streaming softmax
- accumulate rotated values
- optionally avoid even the final inverse rotation by fusing it into the output projection offline

The system never needs to reconstruct the cached KV tensors as dense FP16 arrays during decode.

## Design Overview

`torque-mlx` combines three main ideas:

- Basis-native attention: attention operates directly on packed quantized KV codes in rotated space.
- Offline weight fusion: the shared rotation can be absorbed into `W_Q`, `W_K`, `W_V`, and `W_O` so runtime rotation cost goes to zero.
- Structured rotations: Walsh-Hadamard-based transforms replace dense random rotations so the method maps cleanly to Apple GPU execution.

In product terms, those become:

- offline conversion into a torque-aware artifact
- runtime execution through a torque-aware KV cache
- proof via benchmark and evaluation commands

## Architecture

The project is organized around a small number of product-critical components:

- converted model artifacts: versioned bundles that carry fused weights and runtime metadata
- `TorqueKVCache`: drop-in KV-cache implementation for MLX/MLX-LM-style decode flows
- fused Metal decode kernels: the hot path for `q_len = 1` decode
- quantization and packing logic: turns rotated KV tensors into packed code streams
- rotation and conversion tooling: supports structured rotations and offline weight fusion
- benchmark/eval harness: validates throughput, latency, memory, and quality against FP16 and other baselines

## CLI

The repository now exposes a packaged CLI:

- `torque-mlx convert`: fuse attention weights from an `.npz` checkpoint and emit a versioned artifact
- `torque-mlx inspect`: print artifact metadata, runtime config, and weight shapes
- `torque-mlx plan qwen`: inspect a local Qwen snapshot and emit a curated conversion plan
- `torque-mlx convert-qwen-layer`: convert one extracted Qwen `full_attention` layer into an artifact
- `torque-mlx convert-qwen-model`: rewrite a local Qwen HF snapshot with fused `full_attention` weights and emit a model manifest
- `torque-mlx inspect-qwen-model`: inspect a converted Qwen model manifest
- `torque-mlx benchmark synthetic`: run the reference decode benchmark
- `torque-mlx benchmark mlx-packed`: run the MLX packed-kernel benchmark
- `torque-mlx benchmark mlx-lm`: compare against MLX-LM FP16 and quantized cache baselines
- `torque-mlx eval`: run artifact-level evaluation using the synthetic decode harness

The on-disk artifact contract is defined in [docs/contracts/model-artifact.md](./docs/contracts/model-artifact.md).
Curated family workflows are documented under [docs/families](./docs/families/qwen.md).

## Decode Path

The critical runtime path is fused decode attention. A single kernel pass over the cache performs:

1. packed-index unpacking and centroid-lookup dot products for attention logits
2. numerically stable streaming softmax
3. weighted rotated-value accumulation
4. output writeback in rotated space, or direct consumption by a fused output projection

This is the product’s defining optimization. If decode materializes intermediate dense KV, the design has failed.

## Data Representation

The cache stores rotated keys and values as packed codebook indices rather than floating-point tensors.

Canonical runtime layout:

- `K_codes`: `[layers, kv_heads, seq_len, packed_words]`
- `V_codes`: `[layers, kv_heads, seq_len, packed_words]`

Initial packing targets:

- 2-bit: 16 indices per `uint32`
- 3-bit: aligned multiword packing for SIMD-friendly decode
- 4-bit: 8 indices per `uint32`

The small centroid tables used for lookup are intended to live in Metal constant memory so codebook access is effectively free relative to cache traffic.

## Rotation Strategy

`torque-mlx` uses structured Hadamard-based rotations instead of dense QR-style random rotations.

Why:

- head dimensions `64`, `128`, and `256` are natural fits for Hadamard structure
- the cost drops from dense `O(d^2)` work to structured `O(d log d)` work
- butterfly-style transforms map well to Apple GPU SIMD groups
- the method preserves the outlier-suppression benefits needed for low-bit scalar quantization

The default design is a sign-flipped Hadamard family of the form `Pi = D1 H D2`.

## Weight Fusion

For deployments that want zero runtime rotation overhead, `torque-mlx` supports offline fusion of the shared rotation into the attention weights:

- `W_Q' = Pi W_Q`
- `W_K' = Pi W_K`
- `W_V' = Pi W_V`
- `W_O' = W_O Pi^T`

After fusion, the model natively produces and consumes rotated representations, and the only approximation left in the attention block is KV quantization itself.

The current converter writes those fused attention weights into a versioned artifact directory so runtime configuration, codebooks, and fused weights stay in sync.

## Metal Execution Model

The kernels are designed around Apple GPU constraints rather than treating Metal as a thin backend:

- SIMD width `32`
- threadgroup sizing in multiples of `32`
- careful control of register pressure
- minimal threadgroup-memory staging unless it clearly reduces bandwidth
- compile-time specialization for hot-path variants

Kernel specialization is expected across:

- bit width: `2`, `3`, `4`
- head dimension: `64`, `128`, `256`
- packing mode
- fused-weight vs non-fused mode

## Quality and Performance Targets

Design targets for the system are:

- `4x+` KV memory reduction at practical quality-preserving settings
- decode throughput that exceeds FP16 at long contexts
- no meaningful quality loss at `>= 3.5` bits per channel

At those settings, the project is designed to turn KV quantization into a real decode-speed optimization rather than only a storage optimization.

## Scope

The primary scope is decoder-only transformer inference on Apple silicon through MLX.

The highest-priority supported configuration is:

- batch-1 decode
- head dimensions `64`, `128`, and `256`
- 2-bit, 3-bit, and 4-bit packed KV formats
- Hadamard-based structured rotation
- optional offline fused-weight deployment

Lower-bit residual correction, mixed-bit outlier splits, and cold-cache tiering are natural extensions, but the core product is the fused rotated-basis decode path.

Near-term product scope is:

- generic converted attention artifacts from `.npz` checkpoints
- curated family planners and converters for hand-reviewed model families
- merged local Qwen snapshots with fused `full_attention` layers and a torque manifest
- runtime cache profile reconstruction from the artifact manifest
- synthetic and MLX benchmark entrypoints through the CLI

Universal model-family support for MLX-LM and Hugging Face checkpoints is out of scope for now.

## Evaluation Philosophy

`torque-mlx` is a systems project, so benchmark and evaluation work is part of the product, not an appendix. The project should be judged on:

- tokens/sec and per-token latency
- time-to-first-token
- peak KV memory and bytes per cached token
- perplexity and token-agreement behavior versus FP16
- performance crossover against FP16 and dequantize-on-fetch baselines

If the kernel architecture does not beat the naive quantized path and eventually beat FP16 at long context, the design has not met its goal.

## Repository Documents

- [PRD.md](./PRD.md): product requirements and release criteria
- [tqmlx.pdf](./tqmlx.pdf): original technical design write-up
- [TASKS.md](./TASKS.md): implementation backlog
- [docs/repo-map.md](./docs/repo-map.md): repository structure and code boundaries
- [docs/contracts/model-artifact.md](./docs/contracts/model-artifact.md): converted model artifact contract
- [docs/families/qwen.md](./docs/families/qwen.md): curated Qwen conversion workflow

## Attribution and Citations

Project-local design source:

- [tqmlx.pdf](./tqmlx.pdf) and [tqmlx.tex](./tqmlx.tex) are the original `torque-mlx` design write-up this repository builds from.

Upstream runtimes and ecosystems:

- [MLX](https://github.com/ml-explore/mlx): Apple’s array framework and runtime target for the kernels and cache path in this repo.
- [MLX-LM](https://github.com/ml-explore/mlx-lm): comparison and integration target for the `TorqueKVCache` API and benchmark baselines.
- [Qwen on Hugging Face](https://huggingface.co/Qwen): upstream model family referenced by the curated Qwen planning and conversion workflow in this repo.

Related research that informs the design direction:

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456): key reference for rotation-based quantization and Hadamard-style structured transforms.
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): important baseline for KV-cache quantization and memory/throughput tradeoffs.
- [RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations](https://arxiv.org/abs/2501.16383): reference for rotation placement and low-bit KV-cache quantization behavior in modern decoder models.

These citations are intended as attribution for the ideas and ecosystems `torque-mlx` builds on. They should not be read as claims of endorsement or direct code derivation unless a file in this repository says so explicitly.

## License and Model Rights

`torque-mlx` package metadata declares this project’s code as MIT-licensed. That does not change the license terms of any third-party model snapshot you convert with it.

If you use the Qwen or other curated family workflows:

- upstream model weights, tokenizer assets, configs, and safetensors shards remain subject to their original licenses and usage terms
- converted artifacts should preserve attribution to the original model family and publisher
- benchmark reports should name both `torque-mlx` and the upstream model/runtime stack they were produced from
