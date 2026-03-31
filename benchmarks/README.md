# Benchmarks

This directory will hold reproducible benchmark and evaluation entrypoints for:

- decode throughput
- per-token latency
- time-to-first-token
- KV-cache memory footprint
- quality comparisons against FP16 and quantized baselines

Included today:

- `synthetic_decode.py`: synthetic q_len=1 decode benchmark for the Python reference path
- `../docs/benchmark-report.md`: reproducible sample report from the current reference benchmark

The benchmark plan is defined in `PRD.md` and tracked in `TASKS.md`.

Real MLX/Metal kernel benchmarking and external evals remain blocked until the compiled runtime path and model-specific evaluation harnesses exist.
