# SM90 Dense NVFP4 Single-Primary FP8 Staging Design

## Goal

Replace the dense SM90 NVFP4 fallback's primary/residual FP8 decomposition with one primary FP8 staging value per packed NVFP4 weight, then remeasure all six Cupra grouped-GEMM shapes under the unchanged correctness contract.

The persistent representation remains genuine NVFP4: packed E2M1 codes plus the original UE4M3 block scales and global scale. Expansion exists only transiently in shared memory while the kernel runs.

## Scope

- Change `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_grouped_gemm.cuh` in both its warp-specialized and non-warp-specialized dense paths.
- Change the dense shared-memory sizing in `csrc/jit_kernels/impls/sm90_nvfp4_grouped_gemm.hpp`.
- Leave the K <= 2048 complementary 2:4 sparse path unchanged.
- Leave the persistent packed-weight layout, Python API, and `prepare()` behavior unchanged.

## Data flow

For each dense B tile, the producer reads packed E2M1 codes and their UE4M3 block scales, uses the existing primary lookup table to form one normalized E4M3 value, and writes that value to a single shared-memory B staging buffer. The consumer issues one set of WGMMA operations against that buffer. The epilogue continues to apply the existing global normalization factor.

The residual lookup table, residual shared-memory buffer, residual producer decode, and second set of WGMMA instructions are removed from the dense kernel. Shared-memory allocation changes from A + 2B per stage and two lookup tables to A + B per stage and one lookup table.

## Correctness and risk

Some E2M1 x UE4M3 products are not exactly representable by E4M3, so single-primary staging introduces a bounded weight-rounding error. This change deliberately tests whether the task's existing elementwise tolerance accepts that error after the complete grouped SwiGLU computation. No tolerance, reference, input distribution, or checker is changed.

The full six-shape Cupra run is the acceptance test. In addition to PASS/FAIL, diagnostics must report the number of elements violating `abs(out-ref) <= 2 + 0.05*abs(ref)` and the maximum absolute error. If a shape fails, a later correction mechanism must be justified by that measured failure rather than assuming a residual path is necessary everywhere.

## Performance accounting

Report useful GEMM FLOPs from the original problem dimensions. The expected savings are roughly half of the dense B staging bytes and WGMMA issue count, plus lower shared-memory use and synchronization pressure. The resulting throughput must not count removed residual work as useful FLOPs.

## Validation

1. Build the Python 3.12 extension in `lmsysorg/sglang:dev` on the H20 host.
2. Run the six fixed Cupra shapes with a fresh DeepGEMM JIT cache.
3. Record latency, useful TFLOP/s, Cupra peak percentage, status, violation count, and maximum absolute error.
4. Confirm the public prepared tensors retain their original dtype, shape, and number of elements.
