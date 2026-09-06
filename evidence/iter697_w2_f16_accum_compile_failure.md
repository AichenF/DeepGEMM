# Iteration 697: packed-FP16 fence overload compile failure

Fresh SM90a JIT of extension
`v4tp_790c6b763a4cf8954fae_v178mspec` stopped before any CUDA launch.  nvcc
reported two equivalent errors: the candidate's `uint32_t` packed FP16
accumulator could not bind to DeepGEMM's
`ptx::warpgroup_fence_operand(float&)`.

The included CuTe SM90 GMMA header separately defines
`cute::warpgroup_fence_operand(uint32_t&)`, matching the register class used
by `MMA_64x8x32_F16E4M3E4M3_RS_TN`.  The next source-only repair redirects
only the packed-FP16 fence calls to that overload and leaves every production
FP32 fence unchanged.

No device kernel executed, so this iteration contains no correctness,
resource, cold-L2, or latency claim.  Raw compiler output is retained in
`bench/results/iter697_w2_f16_accum_m128_correctness.log`.
