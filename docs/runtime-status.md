# Runtime Status

This note summarizes what the repository currently demonstrates on real Qwen3.5 artifacts.

## What Is Proven

- Converted Qwen artifacts can be planned, converted, loaded, and evaluated end-to-end.
- The torque runtime reduces converted-layer KV cache bytes by about `75%` at `4` bits relative to FP16.
- The MLX generation path now has enough instrumentation to separate:
  - converted full-attention layer time
  - linear-layer time
  - prompt-side append/prefill work
  - packed decode sub-stages such as score, softmax, value accumulation, and tail handling

## Current Real-Model Signal

On the current local `Qwen3.5-2B` workflow in this repo, using a fixed-length `64`-token generation over a `972`-token prompt:

- Source MLX generation is about `15.15 tok/s`
- Torque MLX generation is about `14.55 tok/s`
- Converted-layer KV estimate is about `12.73 MB` in FP16 versus `3.18 MB` packed

So the runtime is close to speed parity on `2B` while preserving the expected KV-memory win.

## What The Latest Profiling Changed

The latest `qwen-runtime-compare --profile-runtime` runs show that the remaining end-to-end gap on Qwen3.5 is not explained mainly by converted torque layers.

In the current profiled `2B` runs:

- converted full-attention layer time is substantial
- but linear-attention layers are an even larger bucket
- the packed torque decode path is no longer the only or even dominant limiter

That means more progress on this model family depends on improving the surrounding Qwen runtime, not just the packed KV kernels.

## What Still Matters Next

- Benchmark a larger target model where standard full-attention layers dominate more of runtime.
- Continue improving long-context split-batched occupancy and codebook/bit-unpack efficiency.
- Measure GPU occupancy/utilization directly next to throughput and KV bytes saved.
- Keep separating synthetic hot-path wins from full-model runtime results.
