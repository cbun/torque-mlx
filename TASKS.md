# torque-mlx Task List

This task list is derived from [PRD.md](./PRD.md) and is ordered so the project proves correctness and performance before taking on lower-priority features.

## Phase 0: Repository Bootstrap

- [x] Initialize git repository
- [x] Add base Python package scaffold
- [x] Install agentic work harness
- [x] Write product requirements document
- [x] Create implementation backlog

## Phase 1: Contracts and Architecture

- [x] Define the public `TorqueKVCache` API and lifecycle
- [x] Specify packed KV tensor layouts for `K_codes` and `V_codes`
- [x] Specify codebook format, metadata layout, and variant identifiers
- [x] Document supported head dimensions, bit widths, and failure modes
- [x] Resolve rotation placement relative to RoPE for the first target model family
- [x] Write an architecture note for Python prototype vs MLX core C++ migration

## Phase 2: Quantization and Rotation Primitives

- [x] Implement Hadamard-based structured rotation utilities for head dims 64 and 128
- [x] Implement quantization codebook generation and serialization
- [x] Implement packed index encoders for 2-bit, 3-bit, and 4-bit modes
- [x] Add packing/unpacking reference tests against known vectors
- [x] Add error-budget tests for rotated quantization on synthetic vectors

## Phase 3: Runtime Cache Skeleton

- [x] Implement `TorqueKVCache` append/update interfaces
- [x] Implement cache metadata validation and variant selection
- [x] Add a pure-Python reference attention path for correctness checking
- [x] Add fixtures for small deterministic decode cases
- [x] Define the runtime boundary between Python orchestration and Metal kernels

## Phase 4: Fused Metal Decode Prototype

- [ ] Implement first fused decode kernel for `q_len=1` (blocked: full Metal toolchain unavailable)
- [ ] Support direct centroid-lookup dot products on packed key codes (blocked: compiled kernel path unavailable)
- [ ] Support numerically stable streaming softmax (implemented in Python reference path; Metal path blocked)
- [ ] Support rotated value accumulation on packed value codes (implemented in Python reference path; Metal path blocked)
- [x] Add correctness tests versus FP16 reference attention
- [ ] Measure per-token overhead of Python kernel dispatch (blocked: no compiled MLX kernel path yet)

## Phase 5: Kernel Specialization and Hardening

- [ ] Add specialized variants for bit widths 2, 3, and 4 (partial: variant selection scaffold exists)
- [ ] Add specialized variants for head dims 64 and 128 (partial: config + dispatch scaffold exists)
- [x] Add packing-layout-aware dispatch
- [ ] Validate register pressure and occupancy with Metal profiling (blocked: full Xcode/Metal profiling unavailable)
- [ ] Benchmark codebook access strategies and keep the winning path (blocked: compiled kernel path unavailable)
- [x] Decide whether the hot path must move into MLX core C++

## Phase 6: Offline Weight Fusion

- [x] Implement offline rotation fusion for `W_Q`, `W_K`, `W_V`, and `W_O`
- [x] Define artifact format and metadata for fused-weight models
- [x] Add equivalence tests against unfused exact-arithmetic reference paths
- [x] Document ordering constraints with downstream weight quantization pipelines

## Phase 7: Benchmarks and Evals

- [x] Build benchmark harness for tokens/sec, decode latency, TTFT, and KV memory
- [x] Add FP16 KV baseline
- [x] Add naive dequantize-on-fetch quantized baseline
- [ ] Add MLX-LM quantized baseline where available
- [ ] Run quality evaluation on WikiText-2
- [ ] Run at least one long-context evaluation benchmark
- [x] Publish a reproducible benchmark report

## Phase 8: Packaging and Release

- [x] Write user-facing integration docs
- [x] Document supported hardware and model matrix
- [x] Mark experimental modes clearly
- [ ] Cut an initial tagged release once acceptance criteria are met

## Stretch Goals

- [ ] Add mixed-bit outlier split for 2.5-bit and 3.5-bit effective modes
- [ ] Add residual bias correction for ultra-low-bit regimes
- [ ] Add prefill-specialized kernels
- [ ] Add cold-cache tiering for very long contexts
