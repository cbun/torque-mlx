# MLX Packed-Kernel Smoke Report

## Scope

This report captures the current MLX JIT Metal prototype, which now consists of:

- a split packed-code path with separate score and value kernels
- a fused single-kernel decode path with in-kernel score computation, softmax, and value accumulation

The fused kernel is correctness-first and not yet optimized for throughput.

## Command

```bash
python benchmarks/mlx_packed_decode.py --seq-len 64 --head-dim 64 --bit-width 4
```

## Output

```json
{
  "bit_width": 4.0,
  "head_dim": 64.0,
  "max_abs_diff_fused_vs_split": 5.960464477539063e-08,
  "max_abs_error_fused": 3.3527612686157227e-08,
  "max_abs_error_split": 5.960464477539063e-08,
  "mlx_fused_decode_ms": 1.3133339816704392,
  "mlx_fused_tokens_per_sec": 761.4209439156464,
  "mlx_split_decode_ms": 0.47874997835606337,
  "mlx_split_tokens_per_sec": 2088.7729403848966,
  "reference_decode_ms": 0.526249990798533,
  "reference_tokens_per_sec": 1900.2375629168139,
  "seq_len": 64.0
}
```

## Interpretation

- Both the split and fused MLX/Metal paths are correct against the dequantized reference path for this synthetic case.
- The split path is currently faster than the fused path because the fused kernel is a one-thread correctness-first implementation.
- The important milestone is that the repo now has a real fused decode kernel to optimize, profile, and parallelize instead of only a conceptual target.
