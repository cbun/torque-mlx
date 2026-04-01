# Fused Weight Artifact Contract

## Purpose

Offline fusion absorbs the shared rotation into the attention projections so runtime rotation can be removed from the hot path.

This document defines the math and ordering constraints for fusion itself. The full on-disk artifact layout is specified in [model-artifact.md](./model-artifact.md).

## Weight Convention

The current implementation assumes:

- `W_Q`, `W_K`, `W_V`: shape `[head_dim, input_dim]`
- `W_O`: shape `[output_dim, head_dim]`

## Fusion Rules

- `W_Q' = Pi W_Q`
- `W_K' = Pi W_K`
- `W_V' = Pi W_V`
- `W_O' = W_O Pi^T`

## Artifact Fields

Minimum artifact metadata:

- rotation seed
- rotation mode
- head dimension
- fused variant id
- serialization format version

## Ordering Constraint

If a downstream pipeline also quantizes weights, rotation fusion must happen before weight quantization. Otherwise the fused model will not be mathematically equivalent to the original floating-point model.
