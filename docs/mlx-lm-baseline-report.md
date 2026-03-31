# MLX-LM Baseline Smoke Report

## Scope

This report compares three synthetic decode paths on the same q_len=1 workload:

- MLX-LM `KVCache` with floating-point SDPA
- MLX-LM `QuantizedKVCache`
- `torque-mlx` packed-code MLX decode path

## Command

```bash
python benchmarks/mlx_lm_baseline.py --seq-len 64 --head-dim 64 --bit-width 4
```

## Output

```json
{
  "bit_width": 4.0,
  "head_dim": 64.0,
  "max_abs_error_quantized_vs_fp16": 0.012879990041255951,
  "max_abs_error_torque_vs_fp16": 0.04252122342586517,
  "mlx_fp16_decode_ms": 0.36833295598626137,
  "mlx_fp16_tokens_per_sec": 2714.934908070782,
  "mlx_lm_quantized_decode_ms": 0.43508398812264204,
  "mlx_lm_quantized_tokens_per_sec": 2298.4068071889574,
  "seq_len": 64.0,
  "torque_mlx_decode_ms": 2.0213749958202243,
  "torque_mlx_tokens_per_sec": 494.7127584281929
}
```

## Interpretation

- The MLX-LM floating-point baseline is still the fastest path in this tiny synthetic benchmark.
- The MLX-LM quantized baseline is slightly slower than FP16 here, which is consistent with the repo's motivating problem statement.
- `torque-mlx` is currently slower because the fused packed-code kernel is still a correctness-first implementation with minimal parallelism.
- The useful result is that the benchmark now exists and can measure progress against both MLX-LM baselines as kernel parallelism improves.
