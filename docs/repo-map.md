# Repository Map

## Purpose

This repository is organized around a decode-first implementation of rotation-native KV-cache compression for MLX.

## Top-Level Areas

- `src/torque_mlx`: product code
- `tests`: unit and reference correctness tests
- `benchmarks`: reproducible performance and evaluation harnesses
- `docs`: architecture notes, repo map, and future contracts
- `work`: event-driven work artifacts from the harness
- `agents`: reusable harness roles and project configuration

## Expected Code Boundaries

- Cache runtime and API surface live in `src/torque_mlx/cache.py`
- Quantization and packing logic live in `src/torque_mlx/quantization.py`
- Rotation helpers live in `src/torque_mlx/rotation.py`
- Runtime layout contracts live in `src/torque_mlx/layout.py`
- Reference attention math lives in `src/torque_mlx/reference.py`
- Metal kernel integration metadata lives in `src/torque_mlx/kernels/`
- Offline conversion tooling lives in `src/torque_mlx/conversion.py`

## Source Documents

- Product requirements: `PRD.md`
- Original design proposal: `tqmlx.tex`
