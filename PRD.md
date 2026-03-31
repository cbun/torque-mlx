# torque-mlx Product Requirements Document

## Document Status

- Status: Draft v1
- Date: 2026-03-31
- Source: Derived from `tqmlx.pdf` and `tqmlx.tex`
- Audience: Product, research, ML systems, kernel, and MLX integration engineers

## 1. Executive Summary

`torque-mlx` is a KV-cache compression system for MLX on Apple silicon. Its core product thesis is that KV-cache quantization only improves inference if the runtime never expands cached keys and values back into dense FP16 buffers during decode. The product therefore stores keys and values in a rotated basis as packed quantized codes and performs attention scoring and value accumulation directly on those codes inside fused Metal kernels.

The product is intended to make long-context decode on Apple silicon materially faster and cheaper in memory than FP16 KV caching, while preserving model quality at practical bit widths. The initial target is a drop-in MLX/MLX-LM-compatible cache implementation with offline tooling for optional weight fusion and a benchmark suite that demonstrates clear wins over FP16 and naive quantized-cache baselines.

## 2. Problem Statement

On Apple silicon, long-context inference becomes memory-bandwidth bound because the KV cache grows linearly with context length and decode repeatedly rereads the cache. Existing quantized KV-cache implementations frequently underperform FP16 because they dequantize packed codes on fetch before running attention, which reintroduces bandwidth amplification and extra intermediate materialization.

This creates a gap in the market:

- Users want the memory savings of 2-4 bit KV caches.
- Existing MLX paths do not reliably turn those savings into decode speedups.
- Apple GPU execution constraints require a purpose-built kernel architecture rather than a generic CUDA-style translation.

`torque-mlx` solves this by keeping the entire attention computation in the rotated basis and operating directly on packed quantized codes.

## 3. Product Vision

Enable long-context LLM inference on Apple silicon where quantized KV caches are both:

- Faster than FP16 at sufficiently long contexts.
- Small enough to extend usable context windows on memory-constrained machines.

The product should feel like infrastructure, not research code: drop-in, benchmarked, predictable, and easy to integrate into MLX-LM-based serving and experimentation flows.

## 4. Target Users

### Primary Users

- MLX-LM developers serving or benchmarking decoder-only LLMs on Apple silicon
- Researchers experimenting with long-context inference and KV-cache compression
- Performance engineers optimizing decode throughput on Mac hardware

### Secondary Users

- Model packagers producing MLX-native model variants for Apple deployments
- Tooling authors who want a quantized cache backend with a stable interface

## 5. Jobs To Be Done

### Core Jobs

- As an MLX-LM user, I want to swap in a new KV cache with minimal model-code changes.
- As a performance engineer, I want decode throughput to improve at long contexts instead of regressing.
- As a model packager, I want to optionally fuse rotations into weights offline so runtime overhead goes to zero.
- As a researcher, I want clean baselines, ablations, and metrics across bit widths and context lengths.

### Supporting Jobs

- As an integrator, I want a well-defined cache API and compatibility matrix.
- As a kernel engineer, I want specialized variants per bit width and head dimension without runtime branching.
- As a maintainer, I want clear acceptance criteria and a rollout plan that de-risks the hardest features.

## 6. Goals

### Product Goals

- Deliver a production-usable quantized KV-cache path for MLX on Apple silicon.
- Eliminate dequantize-on-fetch from decode attention.
- Support rotation-native attention as the default product path.
- Support optional offline weight fusion for zero runtime rotation.
- Provide compelling benchmark evidence against FP16 and existing quantized baselines.

### Performance Goals

- Achieve at least 4x KV payload reduction at 3.5 bits per channel.
- Achieve decode speedups over FP16 at long contexts on supported Apple silicon SKUs.
- Approach payload-bound speedup limits closely enough that kernel overhead is not the dominant limiter.

### Quality Goals

- Be quality-neutral at 3.5 bits and above on target evaluation tasks.
- Bound degradation at lower bit widths and provide a recovery path via residual correction where implemented.

## 7. Non-Goals

The initial product will not:

- Retrain models or require calibration data as part of the default pipeline.
- Target non-Apple GPU backends.
- Optimize every prefill regime in v1; decode is the critical path.
- Ship cold-cache entropy tiering in the first release unless benchmark data shows it is necessary.
- Promise support for arbitrary head dimensions outside the initial specialization set.

## 8. Product Scope

### In Scope for v1

- Rotated-basis KV quantization and storage
- Fused decode attention kernel operating directly on packed KV codes
- Support for 2-bit, 3-bit, and 4-bit kernel families
- Support for head dimensions 64 and 128
- Hadamard-based structured rotations as the default rotation family
- `TorqueKVCache` runtime API compatible with existing MLX-LM cache usage patterns
- Offline conversion tool for optional weight fusion
- Benchmark and evaluation suite

### Likely v1.1 or Later

- Mixed-bit or outlier-split formats for 2.5-bit and 3.5-bit effective payloads
- Residual bias correction for ultra-low-bit regimes
- Prefill-specialized kernels
- Cold-cache tiering and entropy-coded pages
- Broader model-family compatibility automation

## 9. Core Product Principles

- Never materialize decompressed KV during decode.
- Keep runtime control flow simple and push specialization to compile time.
- Favor structured rotations that map well to Apple GPU hardware.
- Preserve model equivalence where possible by using offline weight fusion instead of runtime patchwork.
- Treat MLX integration overhead as a first-class product constraint, not an implementation detail.

## 10. User Experience Requirements

The product must support a straightforward integration flow:

1. User loads an MLX or MLX-LM model.
2. User swaps the default cache implementation for `TorqueKVCache`.
3. User selects a quantization mode or accepts a default profile.
4. Decode runs through fused Metal kernels automatically.
5. User can benchmark throughput, latency, memory, and quality versus baselines.

For deployments that want zero runtime rotation:

1. User runs an offline conversion tool on model weights.
2. Tool emits a fused-weight model artifact and required metadata.
3. Runtime loads the fused model and skips explicit runtime rotation.

The experience must minimize new call-site complexity. The preferred adoption model is constructor or config substitution rather than attention-module rewrites throughout user code.

## 11. Functional Requirements

### FR-1: Quantized Rotated KV Storage

- The system must store keys and values in a rotated basis.
- The system must store rotated keys and values as packed codebook indices rather than dense floating-point tensors.
- The system must support separate `K_codes` and `V_codes` buffers with layout compatible with per-layer, per-head, per-token traversal.
- The system must maintain enough metadata to interpret packing layout, bit width, rotation mode, and optional outlier partitions.

### FR-2: Rotation-Native Decode Attention

- The decode path must compute attention scores directly from packed quantized key codes.
- The decode path must perform numerically stable streaming softmax without materializing the full score matrix.
- The decode path must accumulate rotated values directly from packed value codes.
- The decode path must output either rotated attention output or post-rotation output depending on fusion mode.

### FR-3: Structured Rotation Support

- The default rotation implementation must use structured Hadamard-based transforms.
- The rotation implementation must support head dimensions 64 and 128 in v1.
- The rotation path must be usable both online for fresh queries and offline for model-weight fusion.

### FR-4: Offline Weight Fusion

- The product must provide an offline tool that fuses the shared rotation into `W_Q`, `W_K`, `W_V`, and `W_O`.
- The tool must emit artifacts that are mathematically equivalent to the original model in exact arithmetic aside from subsequent KV quantization.
- The tool must run before any downstream weight quantization pipeline if both are used.

### FR-5: MLX Runtime Integration

- The product must expose a `TorqueKVCache` object that can substitute for existing MLX-LM cache classes with minimal caller changes.
- The runtime integration must preserve MLX lazy evaluation semantics as much as possible.
- The integration must avoid forcing intermediate tensor materialization in Python.
- The product must provide a prototype path using `mlx.core.fast.metal_kernel` and a production path that can migrate into MLX core C++ if Python overhead is too high.

### FR-6: Kernel Specialization

- The Metal kernel library must use function constants or equivalent compile-time specialization.
- Specialized variants must exist for bit width, head dimension, packing layout, and fused-weight mode.
- Runtime dispatch must choose the correct compiled kernel variant without per-element branching inside the hot path.

### FR-7: Benchmarking and Evaluation

- The project must ship a benchmark suite covering throughput, latency, memory, and quality.
- The suite must compare against FP16 KV cache and at least one dequantize-on-fetch quantized baseline.
- The suite must run across multiple context lengths and at least one long-context regime where bandwidth dominates.

### FR-8: Documentation and Operability

- The project must document supported configurations, model assumptions, and known limitations.
- The project must document how rotation placement interacts with RoPE.
- The project must document how to run offline conversion, runtime inference, and evaluation.

## 12. Non-Functional Requirements

### NFR-1: Performance

- The decode path must be designed for Apple GPU SIMD width 32.
- Threadgroup sizing must align with Apple GPU occupancy constraints.
- The hot kernel must minimize register pressure and unnecessary threadgroup staging.
- The implementation must avoid extra device-memory passes over KV data.

### NFR-2: Memory Efficiency

- The runtime must reduce per-token KV footprint in line with configured bit width.
- Codebooks must live in constant memory or an equally cheap access path where feasible.
- Metadata overhead must remain small enough that payload savings are not erased.

### NFR-3: Accuracy

- At 3.5 bits and above, quality degradation must be statistically negligible on selected evaluation tasks.
- At 2-3 bits, the product must either meet explicit degradation bounds or mark the configuration experimental.

### NFR-4: Reliability

- Unsupported model/config combinations must fail clearly rather than silently falling back to incorrect kernels.
- Kernel dispatch must validate packing parameters and head dimensions.
- Numerical stability must be maintained for long streaming-softmax runs.

### NFR-5: Maintainability

- Kernel variants must be generated or organized to avoid manual duplication explosion.
- Data formats must be documented tightly enough for offline tooling and runtime code to remain in sync.

## 13. Technical Product Specification

### 13.1 System Components

The product consists of five main components:

- Python package surface for user-facing APIs and configuration
- `TorqueKVCache` runtime cache implementation
- Metal kernel library for fused decode attention
- Offline conversion tool for weight fusion and codebook preparation
- Benchmark/evaluation harness

### 13.2 Data Model

Required runtime artifacts:

- Packed key-code tensor: `[layers, kv_heads, seq_len, packed_words]`
- Packed value-code tensor: `[layers, kv_heads, seq_len, packed_words]`
- Global or per-mode codebooks for keys and values
- Rotation metadata per supported head configuration
- Optional outlier-channel metadata for mixed-bit or split-channel modes

### 13.3 Supported Packing Modes

Initial required packing modes:

- 2-bit packed into `uint32` words
- 3-bit packed into aligned multiword groups
- 4-bit packed into `uint32` words

Optional later modes:

- 2.5-bit effective via outlier split
- 3.5-bit effective via outlier split

### 13.4 Kernel Architecture

The primary v1 hot path is a fused decode kernel for `q_len = 1` that performs:

1. Packed-index unpacking and centroid-lookup dot products
2. Streaming softmax update
3. Weighted rotated-value accumulation
4. Final normalization and optional post-rotation handling

This kernel is the defining product differentiator and must remain the optimization priority.

### 13.5 Prefill Strategy

Prefill is out of the critical path for v1 product success. v1 may use a simpler implementation if it does not compromise correctness, but the design should preserve an upgrade path to a prefill-specialized kernel family.

### 13.6 Weight Fusion Behavior

When fused mode is enabled:

- Queries, keys, and values are produced directly in rotated space by modified weights.
- The output projection consumes rotated outputs via fused inverse rotation.
- Runtime avoids explicit rotation for both query generation and output restoration, except where model-specific handling requires otherwise.

## 14. API Requirements

### Required Public Interfaces

The exact API can evolve, but the product must expose:

- A cache class or factory equivalent to `TorqueKVCache`
- Configuration for bit width, head dimension, and fusion mode
- A conversion command or API for offline weight fusion
- Benchmark commands or scripts for reproducible evaluation

### API Behavior

- Cache append/update semantics must match normal autoregressive decode expectations.
- Runtime selection of kernel variants must be transparent to callers.
- Unsupported settings must produce actionable errors with remediation guidance.

## 15. Compatibility Requirements

### Model Assumptions

- Initial focus is decoder-only transformer models running under MLX-LM.
- Initial focus is head dimensions 64 and 128.
- RoPE interaction must be validated per supported model family.

### Hardware Assumptions

- Apple silicon GPUs only
- Unified memory architecture
- Metal backend available

The benchmark matrix must include multiple Apple silicon SKUs if available so the product can validate behavior across bandwidth tiers and occupancy profiles.

## 16. Success Metrics

### Primary Metrics

- Decode throughput in tokens/sec
- Per-token decode latency
- Peak memory consumed by KV cache
- KV bytes per token
- Quality metrics relative to FP16 baseline

### Target Outcomes

- 3.5-bit mode achieves roughly 4.57x KV payload reduction versus FP16.
- Long-context decode is faster than FP16 on supported hardware.
- 3.5-bit mode is quality-neutral on selected evaluation tasks.

### Secondary Metrics

- Time-to-first-token
- Kernel occupancy and SIMD utilization
- Variant compile count and binary size
- Conversion-tool runtime and artifact size

## 17. Benchmark and Evaluation Plan

### Baselines

- FP16 KV cache
- Existing MLX-LM quantized KV cache where applicable
- Naive quantized cache with dequantize-on-fetch
- External methods such as KIVI or RotateKV where feasible and reproducible

### Benchmark Matrix

- Context lengths from short to very long
- Batch 1 decode as the priority regime
- Moderate batch sizes as secondary coverage
- Bit widths 2, 3, 4 in v1; mixed-bit if implemented

### Accuracy Matrix

- WikiText-2 perplexity
- LongBench or equivalent long-context tasks
- Token agreement versus FP16

### Ablations

- No rotation versus structured rotation
- Fused decode versus dequantize-on-fetch
- Weight fusion on versus off
- 3-bit packing options
- Codebook residence in constant memory versus alternative staging if tested

## 18. Release Plan

### Milestone 1: Feasibility Prototype

- Implement packed KV format
- Implement a Python-prototyped fused decode kernel
- Verify correctness against FP16 attention on small test cases
- Demonstrate no dequantize-on-fetch path in decode

### Milestone 2: End-to-End MLX-LM Integration

- Ship `TorqueKVCache` with minimal integration friction
- Run decode benchmarks on at least one representative model
- Establish first performance crossover against FP16 at long context

### Milestone 3: Optimization and Hardening

- Add kernel specialization coverage for required modes
- Reduce Python dispatch overhead or move hot path into C++ MLX core if needed
- Validate occupancy, memory behavior, and correctness across SKUs

### Milestone 4: Weight Fusion and Packaging

- Ship offline weight-fusion tool
- Produce fused-model workflow docs
- Finalize benchmark suite and release docs

## 19. Acceptance Criteria

The v1 release is accepted only if all of the following are true:

- The package provides a working `TorqueKVCache` drop-in path for at least one MLX-LM-supported model family.
- Decode uses packed quantized KV directly without dequantize-on-fetch in the hot path.
- The product is faster than FP16 decode at long contexts on at least one supported Apple silicon SKU and does not regress catastrophically on others in the tested matrix.
- 3.5-bit or nearest supported quality-preserving mode is effectively neutral versus FP16 on selected quality benchmarks.
- The benchmark suite and docs are sufficient for an external engineer to reproduce the main claims.

## 20. Risks

### R-1: RoPE Placement Risk

Applying the rotation at the wrong point relative to positional encoding may degrade quality, especially at low bits. This must be resolved per model family, documented, and covered by validation tests.

### R-2: Occupancy Collapse

Overly aggressive unrolling, accumulator sizing, or staging may reduce occupancy and erase the expected bandwidth win. Metal profiling is required before claiming product readiness.

### R-3: Metadata and Unpacking Overhead

Packed formats, outlier metadata, and unpacking logic may become the new bottleneck if the implementation causes excess memory touches or branchiness.

### R-4: MLX Integration Overhead

Python-level custom-kernel dispatch may add enough overhead to reduce decode gains. The product therefore needs a clear migration path to MLX core C++ if prototype overhead is too high.

### R-5: Structured Rotation Limits

Hadamard-based rotations do not provide identical theoretical guarantees to dense random rotations. Very low-bit regimes may expose failure cases, which should be handled by scoping or residual correction rather than optimistic claims.

## 21. Open Questions

- What is the exact v1 definition of "quality-neutral" by model family and benchmark threshold?
- Should 3.5-bit support ship in v1 via outlier split, or should v1 ship 4-bit first and treat 3.5-bit as a near-term follow-up?
- Which model families will be explicitly supported first?
- Can the Python prototype achieve acceptable per-token dispatch overhead, or is C++ integration required before external release?
- Is post-RoPE or pre-RoPE rotation placement correct for each target model family?
- How much benchmark coverage across Apple SKUs is required before claiming broad support?

## 22. Implementation Priorities

Priority order for engineering work:

1. Correct fused decode kernel for packed rotated KV
2. Stable `TorqueKVCache` integration in MLX-LM
3. Performance profiling and specialization hardening
4. Offline weight fusion
5. Lower-bit enhancements such as residual correction
6. Cold-cache tiering

## 23. Launch Readiness Checklist

- Runtime cache path implemented
- Kernel correctness tests passing
- Benchmark suite checked into repo
- At least one reproducible performance report generated
- Quality report versus FP16 generated
- Docs for setup, conversion, inference, and benchmarking written
- Supported model and hardware matrix documented
- Known limitations and experimental flags documented

## 24. Deliverables

- `torque-mlx` Python package
- Fused Metal kernel library with required specializations
- Offline conversion tool for weight fusion and codebook preparation
- Benchmark and evaluation suite
- Documentation covering integration, performance claims, and limitations

## 25. Summary Decision

`torque-mlx` should be built as a decode-first MLX infrastructure product that makes quantized KV caches actually useful on Apple silicon. The defining requirement is architectural rather than purely mathematical: direct attention over packed rotated codes in fused Metal kernels. Everything in the roadmap should be judged by whether it reinforces that outcome.
