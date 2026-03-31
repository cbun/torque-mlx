# Model Support and Rotation Placement

## Initial Target Family

The first target family is RoPE-based decoder-only transformer models with head dimensions `64` or `128`, such as Llama-style architectures.

## Rotation Placement Contract

For the initial target family:

- queries and keys are rotated after RoPE has been applied
- values are rotated after the value projection
- the rotated attention output is inverse-rotated once per head, or consumed directly by fused output weights

This placement preserves the positioned query-key dot product while still keeping the cache in the rotation-friendly basis used for quantization.

## Supported Runtime Modes

- bit widths: `2`, `3`, `4`
- head dimensions: `64`, `128`
- rotation mode: structured Hadamard
- fused or unfused output path

## Unsupported or Experimental Cases

- arbitrary head dimensions
- non-RoPE models
- mixed-bit outlier splits
- residual correction for sub-3-bit regimes
- cold-cache tiering

## Failure Policy

Unsupported configurations should fail explicitly during config validation or kernel selection rather than falling back silently.

