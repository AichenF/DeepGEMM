# W2 function-specific register-contract experiment

## Context

At TP4 M128 the frozen multi-kernel path runs W13/W2 in about
187.584/98.240 microseconds, while the selected one-launch path needs about
203.056/107.200 microseconds.  Both instantiate the same `route_gemm_task`.
The standalone W2 cubin uses 64 registers and emits 32 QGMMAs with 16
dependency waits; the fused entry and every tested device-call/lifetime
variant emit 32 waits.  A direct source cherry-pick therefore does not exist.

## Candidate

Add a default-off `V4_SINGLE_LAUNCH_W2_FUNC_MAXNREG64` experiment.  It is
legal only for TP4 M128 with the selected compact W13 path, the unbounded
M128 entry, and the whole-W2 phase outline.  Annotate only that outlined W2
device function with CUDA's function-specific `__maxnreg__(64)` attribute.
Lower M, TP8, the standalone multi-kernel kernels, arithmetic, task mapping,
intermediate boundaries, synchronization, communication and public ABI stay
unchanged.

This is preferred over a relocatable-device-code split because it tests the
same compiler-contract hypothesis without introducing a device-link build,
new translation-unit ABI, or shared-memory/TMA descriptor duplication.

## Gates

1. Python/JIT build and exact symbol isolation.
2. Static M128 gate: no fixed local spill and exact W2 callee schedule of 32
   QGMMAs / about 16 `WARPGROUP.DEPBAR` waits.  A 32/32 result is rejected
   before a business launch.
3. If static gate passes, compare every routed W2 output element against the
   independent same-source multi-kernel local reference after an excluded
   256 MiB cold-L2 clear.
4. If correct, run paired replay-level TP4 M128 cold-L2 screening, followed
   by all-M TP4 and TP8 run-through only for a positive local signal.
5. Formal selection still requires the original 10x200 all-M rank-max CUDA
   Graph protocol and exactly one timed business kernel per rank.

## Rejection and fallback

Compilation failure, local spill, or a 32/32 schedule closes this attribute
approach.  The next stronger version would compile the W2 callee as a
relocatable device-code translation unit and device-link it into the single
entry; that larger build change requires separate design and static proof.

