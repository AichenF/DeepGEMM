# SM90 NVFP4 BN128 Split RS Mainloop Experiment

## Goal

Test the missing combination between the optimized W8A8 kernel and standard
NVFP4: phase-specific BN128 split kernels with a seven-stage pipeline, while
decoding lane-native NVFP4 directly into FP8 register-source WGMMA operands.

## Rationale

The retained NVFP4 N24 kernel uses BN256 fused execution and only two stages.
The optimized W8A8 Pro M64 reference uses separate BN128 L1/L2 kernels, seven
stages, and the same swapAB N buckets. Previous NVFP4 split tests materialized
decoded FP8 in shared memory and serialized a producer, while the previous RS
test retained the two-stage BN256 fused architecture. Neither tested the
combination that removes the decoded shared tile and restores W8's pipeline.

For BN128, two math warpgroups each own one 64-row weight half. A lossless
model-load prepack stores each warpgroup lane's sixteen E2M1 values in the
braided eight-byte format already verified exhaustively. Four lanes share the
four original UE4M3 scale bytes. No FP8 weight representation is cached.

## Mainloop Screening

Before building full MoE kernels, launch a 384-thread, two-math-warpgroup
microkernel with seven resident stages and compare:

1. W8A8 shared/shared WGMMA with four K32 instructions per BK128 stage.
2. NVFP4 RS with all four lane fragments decoded before WGMMA.
3. NVFP4 RS with `decode(K)` followed immediately by RS-WGMMA(K), allowing
   decode(K+1) to issue while earlier tensor-core work is in flight.

The two RS schedules must produce identical accumulators. SASS must show the
interleaved decoder between QGMMA instructions rather than after a dependency
barrier. Record cycles for N=8/16/24/64, registers, stack/local memory, and
instruction mix.

## Integration Gate

Proceed to full L1/L2 kernels only if the interleaved RS mainloop has no spills
and its incremental cycles over W8 are small enough to project below the
retained N24 endpoint. Full implementation must live in separate L1 and L2
kernel files, retain standard NVFP4 semantics, add no environment/runtime
control, and pass M=8/64 correctness before multi-seed ABBA.

No commit or push is allowed during the experiment.

## Result

The seven-stage, two-math-warpgroup microkernel passed bit-exact accumulator
comparison for all RS schedules, but it did not pass the integration gate.
For N=8/16/24/64, the W8 control measured 135.2/185.3/249.1/569.8 cycles per
BK128 stage. Decoding all four RS fragments before WGMMA measured
484.2/520.3/552.1/1054.1 cycles. Interleaving each decode and WGMMA was
slower at N <= 24 because each newly written register fragment required a new
`wgmma.fence`; the pairwise two-fence schedule only improved N=24 by about 1%.

Projecting the measured per-stage delta over the real Pro L1 and L2 task
counts gives approximately 1.42--1.44 ms, still slower than the retained N24
candidate at roughly 1.34 ms. The full split L1/L2 RS kernels are therefore
not implemented. Raw cycles, resources, and SASS are archived under
`/root/fac/scripts/megamoe/nvfp4_split_rs_mainloop_20260703/` on the H20 host.
