# torque-mlx Task List

This task list is derived from [PRD.md](./PRD.md) and is ordered so the project proves correctness and performance before taking on lower-priority features.

## Phase 0: Repository Bootstrap

- [x] Initialize git repository
- [x] Add base Python package scaffold
- [x] Install agentic work harness
- [x] Write product requirements document
- [x] Create implementation backlog

## Phase 1: Contracts and Architecture

- [ ] Define the public `TorqueKVCache` API and lifecycle
- [ ] Specify packed KV tensor layouts for `K_codes` and `V_codes`
- [ ] Specify codebook format, metadata layout, and variant identifiers
- [ ] Document supported head dimensions, bit widths, and failure modes
- [ ] Resolve rotation placement relative to RoPE for the first target model family
- [ ] Write an architecture note for Python prototype vs MLX core C++ migration

## Phase 2: Quantization and Rotation Primitives

- [ ] Implement Hadamard-based structured rotation utilities for head dims 64 and 128
- [ ] Implement quantization codebook generation and serialization
- [ ] Implement packed index encoders for 2-bit, 3-bit, and 4-bit modes
- [ ] Add packing/unpacking reference tests against known vectors
- [ ] Add error-budget tests for rotated quantization on synthetic vectors

## Phase 3: Runtime Cache Skeleton

- [ ] Implement `TorqueKVCache` append/update interfaces
- [ ] Implement cache metadata validation and variant selection
- [ ] Add a pure-Python reference attention path for correctness checking
- [ ] Add fixtures for small deterministic decode cases
- [ ] Define the runtime boundary between Python orchestration and Metal kernels

## Phase 4: Fused Metal Decode Prototype

- [ ] Implement first fused decode kernel for `q_len=1`
- [ ] Support direct centroid-lookup dot products on packed key codes
- [ ] Support numerically stable streaming softmax
- [ ] Support rotated value accumulation on packed value codes
- [ ] Add correctness tests versus FP16 reference attention
- [ ] Measure per-token overhead of Python kernel dispatch

## Phase 5: Kernel Specialization and Hardening

- [ ] Add specialized variants for bit widths 2, 3, and 4
- [ ] Add specialized variants for head dims 64 and 128
- [ ] Add packing-layout-aware dispatch
- [ ] Validate register pressure and occupancy with Metal profiling
- [ ] Benchmark codebook access strategies and keep the winning path
- [ ] Decide whether the hot path must move into MLX core C++

## Phase 6: Offline Weight Fusion

- [ ] Implement offline rotation fusion for `W_Q`, `W_K`, `W_V`, and `W_O`
- [ ] Define artifact format and metadata for fused-weight models
- [ ] Add equivalence tests against unfused exact-arithmetic reference paths
- [ ] Document ordering constraints with downstream weight quantization pipelines

## Phase 7: Benchmarks and Evals

- [ ] Build benchmark harness for tokens/sec, decode latency, TTFT, and KV memory
- [ ] Add FP16 KV baseline
- [ ] Add naive dequantize-on-fetch quantized baseline
- [ ] Add MLX-LM quantized baseline where available
- [ ] Run quality evaluation on WikiText-2
- [ ] Run at least one long-context evaluation benchmark
- [ ] Publish a reproducible benchmark report

## Phase 8: Packaging and Release

- [ ] Write user-facing integration docs
- [ ] Document supported hardware and model matrix
- [ ] Mark experimental modes clearly
- [ ] Cut an initial tagged release once acceptance criteria are met

## Stretch Goals

- [ ] Add mixed-bit outlier split for 2.5-bit and 3.5-bit effective modes
- [ ] Add residual bias correction for ultra-low-bit regimes
- [ ] Add prefill-specialized kernels
- [ ] Add cold-cache tiering for very long contexts

