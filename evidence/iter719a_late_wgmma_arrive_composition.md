# Iteration 719a: move the FP16-pair WGMMA fence after operand preparation

## SASS-guided hypothesis

Iteration 718b proved that distinct early-clobber destinations alone are not
enough: ptxas still injects one `WARPGROUP.ARRIVE` and one dependency wait
around each W2 QGMMA.  Source inspection then found a semantic placement
difference rather than another instruction spelling:

- `route_gemm_task` fences the WGMMA accumulators and executes
  `wgmma.fence.sync.aligned` before loading packed weights/scales and decoding
  them into the register-source operands;
- DeepGEMM's own SM90 implementations explicitly load all shared inputs
  before `warpgroup_arrive`, with a source comment stating that those reads
  must precede the arrive.

This ordering explains why ptxas must inject additional register fences after
the late operand writes.  The prior pair/fence/inline/warp-sync experiments
did not move the actual `wgmma.fence`; they added compiler constraints or a
warp barrier while retaining the early hardware fence.

## Isolated change

- Add `V4_SINGLE_LAUNCH_W2_F16_PAIR_LATE_ARRIVE=1`, default off and requiring
  the Iteration-718 FP16 paired-inline path.
- For that M128-only specialization, suppress the original early
  `warpgroup_arrive`; after all eight FP8 source words are materialized and
  compiler-fenced, fence the four packed-FP16 accumulators and execute the
  same arrive immediately before the two-QGMMA asm block.
- Leave QGMMA math/order, commit/wait, K-loop, tasks, TMA, epilogue, route,
  k6 combine and TP collective unchanged.  Every other TP4/TP8 specialization
  retains the original fence location.

Candidate source hashes before JIT:

- `v4_flash_tp_wgmma.py`:
  `240e5832d74b9d46cf3340ef4cc9ed6a29e71da14315dafb93ffd5bb9e95698e`
- `bench/v4_flash_tp_single_vs_multi_graph.py`:
  `d6b36861537b6d20a3320f5c3566f2b6d564bbd29454360d1713072be93186b6`

## Gates

1. Fresh JIT and M128 resource inspection: REG <= 64, LOCAL0, no `STL`/`LDL`
   in the exact W2 interval, and no material growth beyond STACK32.
2. Exact W2 SASS: 32 FP16 QGMMAs and <=20 DEPBARs (target 16).  If the ratio
   stays 1:1, reject before CUDA execution and close this line definitively.
3. If static scheduling passes, require local M128 correctness against both
   the prior FP16 path and the FP32 multi reference before any timing.
4. Only a correctness-qualified candidate receives alternating cold-L2 W2
   phase and TP4 graph timing.  Promotion still requires a repeatable gain;
   the project objective remains >=1.10x over the multi-kernel baseline.

No CUDA launch, correctness, timing or speedup is claimed here.
