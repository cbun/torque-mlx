#include <metal_stdlib>
using namespace metal;

constant ushort BITW [[function_constant(0)]];
constant ushort HEAD_DIM [[function_constant(1)]];
constant bool FUSED_WO [[function_constant(2)]];

kernel void torque_attn_decode(
  device const half* q_rot [[buffer(0)]],
  device const uint* k_codes [[buffer(1)]],
  device const uint* v_codes [[buffer(2)]],
  constant half* cent_k [[buffer(3)]],
  constant half* cent_v [[buffer(4)]],
  device half* out_rot [[buffer(5)]],
  uint3 tg_pos [[threadgroup_position_in_grid]],
  uint simd_lane [[thread_index_in_simdgroup]],
  uint simd_gid [[simdgroup_index_in_threadgroup]]
) {
  // This kernel source is a checked-in prototype artifact, not a compiled path.
  // The intended production behavior is:
  // 1. unpack packed indices
  // 2. compute centroid-lookup dot products for scores
  // 3. maintain online softmax statistics
  // 4. accumulate rotated values directly from packed codes
  // 5. optionally hand off to fused output projection
  (void)q_rot;
  (void)k_codes;
  (void)v_codes;
  (void)cent_k;
  (void)cent_v;
  (void)out_rot;
  (void)tg_pos;
  (void)simd_lane;
  (void)simd_gid;
  (void)BITW;
  (void)HEAD_DIM;
  (void)FUSED_WO;
}
