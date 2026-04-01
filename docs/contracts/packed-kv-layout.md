# Packed KV Layout Contract

## Tensor Shapes

The packed cache stores keys and values in the following canonical runtime layout:

- `K_codes`: `[layers, kv_heads, seq_len, packed_words]`
- `V_codes`: `[layers, kv_heads, seq_len, packed_words]`

The final axis stores packed scalar codebook indices for one head vector.

## Packing

Initial supported packing modes:

- 2-bit: 16 indices per `uint32`
- 3-bit: 96-bit aligned groups for 32 indices, represented as 3 `uint32` words
- 4-bit: 8 indices per `uint32`

The implementation uses a generic bitstream packer with deterministic `uint32` output. For supported head dimensions, the packed word counts are:

- head dim 64: 4 words at 2-bit, 6 words at 3-bit, 8 words at 4-bit
- head dim 128: 8 words at 2-bit, 12 words at 3-bit, 16 words at 4-bit
- head dim 256: 16 words at 2-bit, 24 words at 3-bit, 32 words at 4-bit

## Codebooks

Each quantized scalar references a small global codebook with `2^b` entries.

Contract:

- centroids are sorted ascending
- centroids are stored as `float32`
- codebooks are serialized as JSON-compatible payloads
- key and value codebooks may differ

## Variant Identifiers

Runtime and offline artifacts share a variant identifier:

`b{bit_width}-h{head_dim}-{rotation_mode}-{fused|unfused}`

Examples:

- `b4-h128-hadamard-unfused`
- `b3-h64-hadamard-fused`
