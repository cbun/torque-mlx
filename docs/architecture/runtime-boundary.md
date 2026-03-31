# Runtime Boundary: Python Prototype vs MLX Core

## Current Boundary

The repository currently contains a Python reference implementation for:

- structured rotation
- scalar quantization and codebooks
- packed KV storage
- streaming decode attention
- offline weight fusion math

This path is useful for:

- contract validation
- correctness tests
- packing/rotation experiments
- synthetic benchmarking

## What Stays In Python

- configuration and variant selection
- offline tooling
- benchmark orchestration
- correctness/reference paths
- serialization of metadata and codebooks

## What Must Move To A Compiled Hot Path

- direct consumption of packed KV inside decode attention
- centroid-lookup dot products in the critical loop
- streaming softmax and rotated value accumulation at token rate
- any path that needs Apple GPU occupancy tuning

## Migration Trigger

If Python-orchestrated MLX custom-kernel dispatch adds measurable per-token overhead that prevents FP16 crossover at long context, the decode hot path should move into MLX core C++ with precompiled Metal libraries and function-constant specialization.

## Current External Blocker

Real Metal compilation and profiling are blocked in the current environment because full Xcode toolchain support is not available. The checked-in `.metal` file is therefore a prototype artifact, not a validated compiled path.

