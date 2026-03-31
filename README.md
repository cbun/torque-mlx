# torque-mlx

`torque-mlx` is a rotation-native KV-cache compression library for MLX on Apple silicon.

It is built for the failure mode that makes many quantized KV caches disappointing in practice: they save bits at rest, then give the bandwidth back by dequantizing cached keys and values into dense FP16 buffers during decode. `torque-mlx` avoids that path entirely. Keys and values are stored as packed quantized codes in a rotated basis, and decode attention runs directly on those packed codes inside fused Metal kernels.

The result is a KV-cache backend designed to make quantization pay off where it matters most on Apple hardware: long-context decode, where unified-memory bandwidth becomes the bottleneck.

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

## Architecture

The project is organized around a small number of product-critical components:

- `TorqueKVCache`: drop-in KV-cache implementation for MLX/MLX-LM-style decode flows
- fused Metal decode kernels: the hot path for `q_len = 1` decode
- quantization and packing logic: turns rotated KV tensors into packed code streams
- rotation and conversion tooling: supports structured rotations and offline weight fusion
- benchmark/eval harness: validates throughput, latency, memory, and quality against FP16 and other baselines

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

- head dimensions `64` and `128` are natural fits for Hadamard structure
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

## Metal Execution Model

The kernels are designed around Apple GPU constraints rather than treating Metal as a thin backend:

- SIMD width `32`
- threadgroup sizing in multiples of `32`
- careful control of register pressure
- minimal threadgroup-memory staging unless it clearly reduces bandwidth
- compile-time specialization for hot-path variants

Kernel specialization is expected across:

- bit width: `2`, `3`, `4`
- head dimension: `64`, `128`
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
- head dimensions `64` and `128`
- 2-bit, 3-bit, and 4-bit packed KV formats
- Hadamard-based structured rotation
- optional offline fused-weight deployment

Lower-bit residual correction, mixed-bit outlier splits, and cold-cache tiering are natural extensions, but the core product is the fused rotated-basis decode path.

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
