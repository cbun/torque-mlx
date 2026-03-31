# TorqueKVCache API Contract

## Purpose

`TorqueKVCache` is the public runtime boundary for the rotation-native KV cache.

## Construction

The cache is configured by `TorqueConfig`, which defines:

- `bit_width`
- `head_dim`
- `num_layers`
- `kv_heads`
- `rotation_mode`
- `rotation_seed`
- `fused_weights`

The config resolves to a single variant identifier of the form:

`b{bit_width}-h{head_dim}-{rotation_mode}-{fused|unfused}`

## Append Lifecycle

`append(key=..., value=...)` appends one decode step for every layer and KV head.

Input shapes:

- single layer, single head: `[head_dim]`
- single layer, multi-head: `[kv_heads, head_dim]`
- multi-layer: `[num_layers, kv_heads, head_dim]`

Append behavior:

1. rotate incoming keys and values into structured Hadamard space
2. quantize to scalar codebook indices
3. pack indices into `uint32` words
4. append packed words to the per-layer, per-head token stream

## Decode Lifecycle

`decode(query=...)` consumes a query tensor with the same layer/head conventions as `append`.

Decode behavior:

1. rotate fresh queries into the cache basis
2. read packed key codes token-by-token
3. compute attention scores from centroid lookups
4. run numerically stable streaming softmax
5. read packed value codes token-by-token and accumulate output
6. inverse-rotate the output unless fused weights are enabled

## Metadata

`metadata` returns a `CacheMetadata` object containing:

- variant id
- bit width
- head dimension
- layer count
- KV head count
- rotation mode
- fused-weight mode
- codebook name

## Failure Modes

The cache must fail fast for:

- unsupported bit widths
- unsupported head dimensions
- shape mismatches on append or decode
- malformed codebooks

