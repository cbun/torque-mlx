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

- [x] Implement first fused decode kernel for `q_len=1`
- [x] Support direct centroid-lookup dot products on packed key codes
- [x] Support numerically stable streaming softmax
- [x] Support rotated value accumulation on packed value codes
- [x] Add correctness tests versus FP16 reference attention
- [x] Measure per-token overhead of Python kernel dispatch

## Phase 5: Kernel Specialization and Hardening

- [x] Add specialized variants for bit widths 2, 3, and 4
- [x] Add specialized variants for head dims 64 and 128
- [x] Add packing-layout-aware dispatch
- [ ] Validate register pressure and occupancy with Metal profiling
- [ ] Benchmark codebook access strategies and keep the winning path
- [x] Decide whether the hot path must move into MLX core C++
- [x] Replace the Python-list MLX cache path with a GPU-resident packed slab
- [x] Batch decode across converted layers and KV heads to remove per-head dispatch overhead
- [x] Move query rotation and output unrotation onto the MLX device path
- [x] Add a runtime strategy switch so batched split and per-head fused kernels can be compared explicitly
- [x] Evaluate a batched split-kernel path against the current fused kernel and promote the faster design
- [x] Add a dedicated packed-plus-current-row decode path so single-token generation does not always fall back to the generic tail merge
- [ ] Make the split-batched kernel saturate GPU occupancy on Apple silicon
- [ ] Reduce bit-unpack and codebook-lookup overhead in the split-batched kernel path
- [ ] Revisit threadgroup/grid layout and SIMD-group work partitioning for long-context decode
- [ ] Decide whether the split-batched hot path must move into MLX core C++ / precompiled Metal

## Phase 6: Offline Weight Fusion

- [x] Implement offline rotation fusion for `W_Q`, `W_K`, `W_V`, and `W_O`
- [x] Define artifact format and metadata for fused-weight models
- [x] Add equivalence tests against unfused exact-arithmetic reference paths
- [x] Document ordering constraints with downstream weight quantization pipelines
- [x] Add a true compressed converted-model artifact format instead of dense `safetensors` rewrites
- [x] Define loader/runtime contracts for compressed whole-model artifacts
- [ ] Support exact Qwen-family fusion beyond `vo_only_runtime_qk_rotation` where architecture permits
- [ ] Investigate and reduce the observed Qwen3.5 `2B` quality drift in real-model evals

## Phase 7: Benchmarks and Evals

- [x] Build benchmark harness for tokens/sec, decode latency, TTFT, and KV memory
- [x] Add FP16 KV baseline
- [x] Add naive dequantize-on-fetch quantized baseline
- [x] Add MLX-LM quantized baseline where available
- [x] Make the Qwen decode benchmark model grouped-query attention faithfully
- [ ] Run quality evaluation on WikiText-2
- [ ] Run at least one long-context evaluation benchmark
- [x] Re-run Qwen decode benchmarks at multiple prefills after the GPU-resident cache upgrade
- [x] Add a real MLX generation benchmark for converted Qwen snapshots
- [x] Benchmark real-model MLX generation on Qwen3.5 `0.8B` and `2B`
- [x] Add a runtime-comparison benchmark that aligns real generation runs with the synthetic decode hot path
- [x] Attribute profiled MLX generation time across converted full-attention layers, linear layers, and torque decode sub-stages
- [ ] Benchmark real-model MLX generation on a larger target model
- [ ] Measure GPU occupancy/utilization alongside throughput and KV bytes/token
- [ ] Benchmark `4096+` token prefills on the split-batched default path
- [ ] Publish a benchmark report that clearly separates synthetic decode geometry from real-model runtime results
- [x] Publish a reproducible benchmark report

## Phase 8: Packaging and Release

- [x] Write user-facing integration docs
- [x] Document supported hardware and model matrix
- [x] Mark experimental modes clearly
- [x] Ship a real MLX runtime adapter that loads converted Qwen artifacts end-to-end
- [ ] Publish at least one compressed converted artifact once the runtime and quality bar are met
- [ ] Cut an initial tagged release once acceptance criteria are met

## Stretch Goals

- [ ] Add mixed-bit outlier split for 2.5-bit and 3.5-bit effective modes
- [ ] Add residual bias correction for ultra-low-bit regimes
- [ ] Add prefill-specialized kernels
- [ ] Add cold-cache tiering for very long contexts
