# MLX Packed-Kernel Smoke Report

## Scope

This report captures the current MLX JIT Metal prototype, which consists of:

- a packed-code query/key score kernel
- a packed-code value accumulation kernel
- softmax performed in MLX between the two kernels

It is not yet the final fully fused decode kernel, but it validates that packed-code attention work can run through real Metal kernels from Python.

## Command

```bash
python benchmarks/mlx_packed_decode.py --seq-len 64 --head-dim 64 --bit-width 4
```

## Output

```json
{
  "bit_width": 4.0,
  "head_dim": 64.0,
  "max_abs_error": 5.960464477539063e-08,
  "mlx_packed_decode_ms": 0.474542030133307,
  "mlx_packed_tokens_per_sec": 2107.2949001357856,
  "reference_decode_ms": 0.4818330053240061,
  "reference_tokens_per_sec": 2075.4078465993734,
  "seq_len": 64.0
}
```

## Interpretation

- The MLX/Metal prototype is correct against the dequantized reference path for this synthetic case.
- On this tiny synthetic benchmark it is roughly at parity with the reference path after warmup, which is a better starting point than the earlier ad hoc smoke measurement.
- This is still useful progress because it proves the packed-code path is executable through MLX custom Metal kernels and can now be profiled and fused further.
