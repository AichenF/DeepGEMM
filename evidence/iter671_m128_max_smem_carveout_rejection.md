# Iteration 671 — standalone-like maximum shared carveout is neutral/slower

## Motivation

Saved NCU reports showed one launch-policy difference despite the shared GEMM
body: standalone W13 used a 233.472-KiB shared-memory configuration, while the
selected M128 one-kernel entry used 200.704 KiB. This experiment transferred
only that policy to the one-kernel M128 specialization.

## Static and launch gates

- A temporary M128-only switch called
  `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout,
  cudaSharedmemCarveoutMaxShared)` before the existing occupancy query.
- Device code, 702-CTA grid, task rotations, arithmetic and M<=64 were
  unchanged.
- Candidate cubin remained `REG56 STACK32 SHARED2048 LOCAL0`.
- NCU LaunchStats confirmed `702x128`, 56 registers, 18.432 KiB dynamic plus
  1.024 KiB reported static shared memory, one wave/SM, and an actual
  233.472-KiB configuration. The attribute therefore took effect.

## Correctness

M128 random routing with 1,992 padded rows and split-K2 was bitwise equal to
the independent same-source multi local routed-W2 tensor (`cosine=1`,
`rel_l2=0`, finite). Packed generations wrapped exactly to `[0,0,0,0]` after
an excluded 256 MiB cold-L2 clear.

## TP4 cold-L2 bracket

GPUs 0/5/6/7, CUDA Graph, OFF/ON/ON/OFF independent processes, two outer
batches x 20 samples per implementation/process, four warmups, replay-level
pairing, and a separate excluded 256 MiB L2 clear before every replay.

| Window | carveout | multi (ms) | one (ms) | one/multi |
|---|---:|---:|---:|---:|
| OFF_A | 200.704 KiB | 0.305312 | 0.349984 | 1.146316 |
| ON_A | 233.472 KiB | 0.305312 | 0.351472 | 1.151190 |
| ON_B | 233.472 KiB | 0.305344 | 0.350656 | 1.148397 |
| OFF_B | 200.704 KiB | 0.306048 | 0.351376 | 1.148107 |

OFF and ON mean paired ratios are 1.147212 and 1.149793. Maximum carveout is
0.225% slower after paired-baseline normalization; direct one-kernel means
are 0.350680 and 0.351064 ms (0.110% slower).

The prior reports also show identical 20.480-KiB per-block shared allocation.
The complete one-kernel report has only 120 excessive shared wavefronts out of
16,955,732 (0.000708%), too small to explain its issue-rate deficit.

## Verdict

Reject below the 1% gate and restore production source byte-identical. The
larger partition changes neither occupancy nor device instructions and
slightly reduces useful L1 capacity. Shared carveout is not the standalone
W13 advantage.

## Evidence

- `bench/results/iter671_m128_max_smem_carveout_correctness_20260906.log`
- `bench/results/iter671b_m128_max_smem_carveout_launchstats_20260906.log`
- `results/iter671b_m128_max_smem_carveout_launchstats.ncu-rep`
- `bench/results/iter671{c,d,e,f}_m128_max_smem_*_tp4_cold_20260906.log`
- `results/iter650b_production_m128_cold_sourcecounters.ncu-rep`
- `results/iter651b_standalone_w13_m128_cold_sourcecounters.ncu-rep`
