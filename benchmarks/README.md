# Benchmarks

This directory will hold reproducible benchmark and evaluation entrypoints for:

- decode throughput
- per-token latency
- time-to-first-token
- KV-cache memory footprint
- quality comparisons against FP16 and quantized baselines

Included today:

- `synthetic_decode.py`: synthetic q_len=1 decode benchmark for the Python reference path
- `mlx_packed_decode.py`: MLX JIT Metal benchmark for both split-kernel and fused packed-code prototype paths
- `mlx_lm_baseline.py`: synthetic MLX-LM baseline comparison against `KVCache`, `QuantizedKVCache`, and `torque-mlx`
- `../docs/benchmark-report.md`: reproducible sample report from the current reference benchmark
- `../docs/mlx-kernel-report.md`: reproducible sample report from the current MLX packed-kernel smoke benchmark
- `../docs/mlx-lm-baseline-report.md`: reproducible sample report for the MLX-LM baseline comparison

The benchmark plan is defined in `PRD.md` and tracked in `TASKS.md`.

External model evals still remain blocked until model-specific evaluation harnesses exist.
