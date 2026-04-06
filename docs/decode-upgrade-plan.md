# Decode Upgrade Plan

This note explains why `torque-mlx` is currently losing on the Qwen decode benchmark, which parts of the original thesis still hold, and what the next runtime upgrade should be.

## Current Findings

As of April 6, 2026, the `qwen-decode` benchmark on local `Qwen3.5-0.8B` artifacts shows:

| Prefill | Decode Steps | MLX-LM FP16 tok/s | MLX-LM quant tok/s | torque tok/s | KV cache reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 681.6 | 1115.4 | 8.57 | 75% |
| 512 | 32 | 385.0 | 400.9 | 3.94 | 75% |
| 2048 | 32 | 527.2 | 469.1 | 1.23 | 75% |

At the same time, `qwen-text` still shows the converted artifact is quality-preserving enough to keep optimizing:

- `c=2048`, first `2048` tokens: exact parity, `PPL 12.389556903649764`
- `c=512`, first `2048` tokens: very close, source `24.7036`, torque `24.6057`

So the current state is:

- the memory thesis is partially validated for KV cache size
- the quality path is usable
- the performance thesis is not validated by the current runtime

## Root Cause Analysis

### 1. The MLX decode path still rebuilds the cache on every token

The current `TorqueKVCache.decode_mlx()` implementation in [src/torque_mlx/cache.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/cache.py) stores packed codes as Python lists of NumPy arrays, then does this on every decode step:

- `np.stack(packed_keys, axis=0)`
- `np.stack(packed_values, axis=0)`
- `mx.array(...)` upload for keys
- `mx.array(...)` upload for values

That means the runtime is paying `O(seq_len)` host-side work and host-to-device transfer per token. This defeats the whole point of keeping KV in a compact format.

### 2. The fused decode kernel has essentially zero GPU occupancy

The current fused kernel in [src/torque_mlx/mlx_ops.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/mlx_ops.py) launches with:

- `grid=(1, 1, 1)`
- `threadgroup=(1, 1, 1)`

and a single thread loops over:

- every token in the sequence
- every dimension in the head
- score computation
- online softmax
- value accumulation

That is a correctness prototype, not a competitive kernel.

### 3. The runtime dispatch is too fragmented

The current `decode_mlx()` loop dispatches once per:

- layer
- KV head

For the current Qwen benchmark profile, that means repeated small dispatches and `mx.eval()` boundaries instead of one batched decode over all converted layers and heads.

### 4. Rotation is still happening on the CPU / NumPy side

The runtime currently:

- rotates the query in NumPy before dispatch
- pulls outputs back to NumPy
- inverse-rotates outputs in NumPy when `fused_weights` is false

That adds avoidable device/host boundaries around the hottest part of the decode loop.

### 5. The current Qwen benchmark is still an approximation

The `qwen-decode` benchmark in [src/torque_mlx/qwen_benchmark.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/qwen_benchmark.py) captures:

- head dimension
- KV heads
- converted layer count
- grouped-query ratio

but it does not yet model the full `num_attention_heads` query fanout in the runtime path. So it is already useful, but it is not yet a perfect architectural model of real Qwen grouped-query attention.

### 6. Qwen3.5 needs a hybrid runtime, not a pure offline-fusion story

For real Qwen3.5 full-attention layers:

- `q_norm` and `k_norm` prevent exact offline fusion of `Q` and `K`
- the output gate means attention output semantics are not the simple `attn_out -> o_proj` path

That means the right runtime model is:

- offline fusion where exact (`V` and `O`)
- runtime handling where exact offline fusion is unavailable (`Q` and `K`)

This is not a failure of the core idea, but it does mean the deployment story for Qwen must explicitly support hybrid fusion modes.

## Revised Thesis

The original thesis should be refined as follows:

1. `torque-mlx` reduces KV cache size and decode-time KV memory traffic, not full model weight size.
2. The rotation-native packed KV idea still makes sense.
3. The current implementation loses because it violates the intended execution model:
   - the cache is not GPU-resident
   - the kernel is not parallel enough
   - the runtime is not batched enough
4. On Apple GPUs, a split batched kernel may be the right intermediate step before a truly fused kernel. Fusion is only useful if occupancy stays high.

## Upgrade Goals

The next implementation should satisfy these goals:

- no `O(seq_len)` host-side restacking during decode
- no per-token host-to-device upload of the existing cache
- one batched decode path over all converted layers and heads
- on-device query rotation and output unrotation
- support for Qwen grouped-query attention
- preserve current quality parity on `qwen-text`

Target outcome:

- keep the 75% KV-cache reduction at 4-bit
- close the gap to MLX-LM quantized decode first
- then target long-context crossover on supported Qwen layers

## Upgrade Design

### Phase 0: Make the benchmark faithful

Before optimizing further, improve `qwen-decode` so it reflects real Qwen grouped-query attention more closely.

Changes:

- generate `num_attention_heads` queries and `num_key_value_heads` KV blocks
- benchmark grouped query fanout instead of only KV heads
- separate timing into:
  - cache append/update
  - decode kernel time
  - host orchestration overhead
- keep reporting projected KV bytes saved

Files:

- [src/torque_mlx/qwen_benchmark.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/qwen_benchmark.py)
- [tests/test_qwen_benchmark.py](/Users/chris/development/personal/torque-mlx/tests/test_qwen_benchmark.py)

### Phase 1: Replace the Python-list cache with a GPU-resident packed slab

The main structural fix is to stop rebuilding the cache every token.

Changes:

- add an MLX-specific cache backend, for example `TorqueKVCacheMLX`
- preallocate packed code storage on device:
  - `k_codes_dev[layer, kv_head, token, packed_word]`
  - `v_codes_dev[layer, kv_head, token, packed_word]`
- keep centroids and rotation matrices resident on device
- append one slice per token instead of appending Python list entries
- allow the benchmark/runtime to pass `mx.array` query inputs directly

Important boundary:

- keep the current NumPy-backed `TorqueKVCache` as the reference implementation
- introduce a distinct MLX backend rather than turning the reference cache into a mixed-mode class

Files:

- [src/torque_mlx/cache.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/cache.py) or a new `cache_mlx.py`
- [src/torque_mlx/artifact.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/artifact.py)
- [tests/test_cache.py](/Users/chris/development/personal/torque-mlx/tests/test_cache.py)

### Phase 2: Batch the decode path across layers and heads

The current per-layer, per-head dispatch pattern is too fine-grained.

Changes:

- add batched decode kernels that consume:
  - `query[layer_head, head_dim]`
  - `k_codes[layer_head, seq, packed_words]`
  - `v_codes[layer_head, seq, packed_words]`
- collapse converted layers and KV heads into a batch axis for the hot path
- move from `mx.eval()` inside nested Python loops to one or a few batched dispatches

This should reduce:

- dispatch count
- Python overhead
- device synchronization points

Files:

- [src/torque_mlx/mlx_ops.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/mlx_ops.py)
- [src/torque_mlx/kernels/torque_attn_decode.metal](/Users/chris/development/personal/torque-mlx/src/torque_mlx/kernels/torque_attn_decode.metal)
- [tests/test_mlx_ops.py](/Users/chris/development/personal/torque-mlx/tests/test_mlx_ops.py)

### Phase 3: Prefer a fast split kernel before revisiting full fusion

The current fused kernel is too serial. A batched split design is the better next step.

Recommended path:

- batched score kernel
- batched softmax using MLX tensor ops
- batched value accumulation kernel

Why:

- split kernels can use more threads immediately
- they are easier to profile and reason about
- the current synthetic reports already suggest split can outperform the existing fused prototype

Only after the split path is strong should the project revisit a fully fused online-softmax kernel.

### Phase 4: Move rotation and Qwen-specific output handling onto the device

For Qwen `vo_only_runtime_qk_rotation`:

- rotate fresh `Q` on device after the model's Q path produces it
- keep the value path in rotated basis
- unrotate the combined attention output on device
- then apply the gate and restored `O` projection in the model adapter

This keeps the hybrid Qwen mode exact while avoiding CPU rotation overhead.

Files:

- [src/torque_mlx/qwen_eval.py](/Users/chris/development/personal/torque-mlx/src/torque_mlx/qwen_eval.py)
- new Qwen runtime adapter module

### Phase 5: Escalate to MLX core C++ / precompiled Metal if needed

The repo already documents the migration trigger in [docs/architecture/runtime-boundary.md](/Users/chris/development/personal/torque-mlx/docs/architecture/runtime-boundary.md):

- if Python-orchestrated MLX dispatch overhead prevents crossover, move the hot path down into MLX core C++ with precompiled Metal libraries

The current benchmark results strongly suggest this may eventually be necessary, but only after:

- GPU-resident cache storage exists
- batching exists
- the split path has been optimized enough to establish a real baseline

## What Should Not Change

These benchmark results do **not** justify:

- abandoning rotation-native packed KV
- pivoting to weight-only compaction as the main story
- claiming full-model size reduction
- chasing arbitrary architectures before one family crosses over
- insisting on a fused kernel when a split batched kernel is clearly better on Apple hardware

## Immediate Backlog

1. Re-run `qwen-decode` at `prefill=128, 512, 2048, 4096` and compare batched split vs prior per-head fused behavior.
2. Decide whether the split path should replace the fused path as the default long-context benchmark/runtime path.
3. Reduce append cost in the MLX cache backend, which is now a visible secondary bottleneck.
4. Profile codebook access and memory layout inside the batched split kernels.
5. Only then decide whether a new fused kernel or MLX core C++ migration is the next bottleneck to attack.
