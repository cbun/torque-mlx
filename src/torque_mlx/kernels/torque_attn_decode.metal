// Prototype kernels used by the MLX-backed packed-code path.
//
// These are mirrored in `src/torque_mlx/mlx_ops.py`, which JITs kernel bodies
// through `mlx.fast.metal_kernel`. This file exists as a checked-in reference
// for the current packed-index score and value-accumulation kernels.
//
// Score kernel:
// - one thread per token
// - unpacks BITW-sized codes from `uint32` words
// - performs centroid-lookup dot products against the rotated query
//
// Value kernel:
// - one thread per output dimension
// - reuses the packed layout to gather centroid values
// - accumulates a weighted sum using softmax weights produced outside the kernel
//
// The next production step is to fuse score computation, online softmax, and
// value accumulation into a single decode kernel.
