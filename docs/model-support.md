# Model Support and Rotation Placement

## Initial Target Family

The first target family is RoPE-based decoder-only transformer models with head dimensions `64`, `128`, or `256`, such as Llama-style and Qwen-style attention variants.

The repository uses a curated-support model:

- support is explicit per family or per published artifact
- unsupported families should fail during planning
- manual family-specific conversion is preferred over speculative generic conversion

## Rotation Placement Contract

For the initial target family:

- queries and keys are rotated after RoPE has been applied
- values are rotated after the value projection
- the rotated attention output is inverse-rotated once per head, or consumed directly by fused output weights

This placement preserves the positioned query-key dot product while still keeping the cache in the rotation-friendly basis used for quantization.

## Supported Runtime Modes

- bit widths: `2`, `3`, `4`
- head dimensions: `64`, `128`, `256`
- rotation mode: structured Hadamard
- fused or unfused output path

## Unsupported or Experimental Cases

- arbitrary head dimensions
- non-RoPE models
- hybrid architectures whose attention/KV behavior has not been explicitly reviewed
- mixed-bit outlier splits
- residual correction for sub-3-bit regimes
- cold-cache tiering

## Failure Policy

Unsupported configurations should fail explicitly during config validation or kernel selection rather than falling back silently.

## Curated Family Notes

### Qwen

Qwen support is treated as a curated workflow:

- inspect the local Hugging Face snapshot first
- convert only layers explicitly marked `full_attention`
- copy non-convertible layers through unchanged
- fail clearly if the model head dimension is outside the supported `64` / `128` / `256` runtime envelope

The family-specific workflow is documented in [families/qwen.md](./families/qwen.md).
