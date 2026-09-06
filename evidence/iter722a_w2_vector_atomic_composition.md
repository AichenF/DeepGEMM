# Iteration 722a — compose Hopper vector-atomic W2 producer combine

- **Hypothesis:** the previously correct scalar producer-combine path lost
  0.93% at M128 because every W2 lane issued four scalar FP32 atomics.  On
  SM90, adjacent FP32 elements can be updated with one element-wise `float2`
  atomic.  The WGMMA accumulator map places adjacent output columns four
  lanes apart while preserving the same route pair, so a warp shuffle plus
  half-lane vector issue halves the atomic instruction count.
- **Isolation:** new default-off
  `V4_SINGLE_LAUNCH_W2_PRODUCER_VECTOR_ATOMIC_COMBINE=1` implies the existing
  producer-combine workspace/communication path but changes only its atomic
  epilogue.  The ordinary single-launch and scalar producer-combine paths are
  unchanged when the flag is zero.
- **Numerics:** each vector operation remains element-wise FP32 atomic add.
  The candidate deliberately retains per-route BF16 rounding, route weight,
  and 1.5 routed scale.  As with scalar producer combine, reduction order may
  differ from the fixed-k6 control, so correctness uses the pre-existing
  tolerance-qualified graph gate.
- **Hard gate:** first require CUDA JIT, vector-atomic SASS, no resource
  regression, and local M8/M128 correctness.  Then compare against the
  matched bound-8 ordinary one-kernel path under replay-level cold L2.  Reject
  if M128 is not at least neutral or if M8 does not show a repeatable gain.
- **Benchmark contract:** caller-provided FP8-E4M3 activation plus FP32
  group-128 scales and MXFP4 weights; one business kernel including TP
  collective; CUDA Graph; separate excluded 256 MiB L2 clear before every
  timed replay.
