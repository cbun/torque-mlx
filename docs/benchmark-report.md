# Synthetic Benchmark Report

## Scope

This report captures the current Python reference-path benchmark. It is useful for reproducibility and for validating data formats and correctness, but it is not a claim about the final fused Metal implementation.

## Command

```bash
python benchmarks/synthetic_decode.py --seq-len 32 --head-dim 64 --kv-heads 2 --bit-width 4
```

## Output

```json
{
  "bit_width": 4.0,
  "head_dim": 64.0,
  "kv_bytes_per_token": 128.0,
  "kv_heads": 2.0,
  "max_abs_error_vs_rotated_reference": 0.0800141841173172,
  "naive_quantized_decode_ms": 8.766333921812475,
  "naive_quantized_tokens_per_sec": 114.07277077499758,
  "reference_decode_ms": 0.713790999725461,
  "reference_tokens_per_sec": 1400.9703125769602,
  "seq_len": 32.0,
  "ttft_proxy_ms": 0.3694590413942933
}
```

## Interpretation

- This is the dequantize-on-fetch reference path, not the final fused packed-code Metal path.
- The result is directionally useful because it demonstrates the known failure mode: naive quantized decode can be slower than the floating-point reference path.
- The report is intentionally checked in so future kernel work has a fixed comparison point.
