# SM90 FP8 DeepEP baseline-only benchmark design

## Scope

Add a benchmark-only driver to the existing
`opt/megamoe-sm90-fp8-h200-retune` workspace.  The measured path is exactly:

1. DeepEP FP8 dispatch.
2. DeepGEMM contiguous grouped-FP8 L1 GEMM.
3. Triton SwiGLU, top-k weighting, and E4M3 quantization.
4. DeepGEMM contiguous grouped-FP8 L2 GEMM.
5. DeepEP combine.

The driver must not call, warm up, transform weights for, or otherwise launch
the PR323 or local fused MegaMoE kernel.

## Shapes and points

- Flash: H=4096, I=2048, E=256, top-k=6.
- Pro: H=7168, I=3072, E=384, top-k=6.
- MiMo-V2.5-Pro: H=6144, I=2048, E=384, top-k=8.
- M: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.

Use eight H200 ranks and deterministic seed 101.  Report the maximum rank for
each observation and the median of three observations, with 20 timed samples
per observation.

## Timing and validation

Use CUDA events around the complete five-stage pipeline.  Flush 8 GB per GPU
outside the timed interval before every sample, synchronize all ranks before
the timed call, and validate that every rank emits the expected sample count.
For each M point, set DeepEP's `num_max_tokens_per_rank` to M and retain its
default exact-extent `do_cpu_sync=1` behavior.  The CPU synchronization is
inside the end-to-end timing.  This avoids measuring an 8192-token graph
capacity at every small-M point while still including the cost required to
obtain the dynamic expert-expanded extent.
The driver checks output shape, dtype, and finiteness without invoking a fused
reference.  Raw per-rank samples and route statistics are emitted as marked
JSON for strict parsing.

The existing MegaMoE H200 results used `DG_BENCH_FLUSH_L2_BYTES=0`; therefore
the new cold-L2 baseline is not compared numerically against those warm-L2
results without a separately requested cold-L2 MegaMoE rerun.

## Integration

Keep the implementation self-contained under `tests/`, borrowing the proven
DeepEP compatibility and SM90 FP8 quantization logic from PR323 while importing
the current workspace's `deep_gemm`.  A static source audit rejects fused
entry-point references.  No production selector or kernel source is changed.
