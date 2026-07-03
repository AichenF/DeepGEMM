# SM90 NVFP4 M16 K+1 Full-Row Overlap Experiment

## Goal

Hide the mandatory generic-to-WGMMA async-proxy publication cost together
with NVFP4 decode at M <= 16, while preserving the braided deployment layout,
the fused three-stage pipeline, and correctness for receive-heavy experts.

## Why This Differs From Rejected Lookahead

The old next-quad lookahead kept decoded fragments live in registers and
ptxas moved their shared stores behind `WARPGROUP.DEPBAR`. The recent
half-row experiment on the braided kernel proved that ptxas can interleave
PRMT/shared stores with an in-flight QGMMA, but splitting each row into two
waves destroyed decoder ILP and required two fences.

This experiment keeps the existing one-thread-per-row decoder unchanged. It
decodes a complete next-stage row per math thread and publishes the whole
stage with one proxy fence.

## Schedule

1. Wait for and fully decode stage zero as a prologue, then publish it to the
   WGMMA async proxy.
2. Submit the first 64-row weight-half QGMMA group for the current stage.
3. If K+1 exists, wait for its TMA full barrier, decode all 256 rows into the
   next ring slot, and execute one async-proxy fence while the current QGMMA
   group is pending.
4. Wait for and promote the current first-half accumulator, then execute the
   remaining current-stage QGMMA groups normally.
5. Release only the current stage's empty barrier and advance. The following
   iteration consumes the already decoded stage without repeating its wait or
   decode.

Both N128 math warpgroups follow the same schedule and each decodes its own
128 rows. The M16 candidate retains only N8/N16 swap-AB WGMMA shapes. Experts
with `valid_m > 16` use compile-time token bases 16/32/48, so the specialization
does not rely on a per-rank receive-count bound. Lookahead runs only in the
base-zero group and therefore exactly once per K stage.

## Alternatives

- Split each row and overlap the second half with WGMMA: already rejected by
  a 30.5% endpoint regression from lost ILP and two synchronization waves.
- Use dispatch or TMA warps as producers: already rejected because issue
  contention or waiting on full stages destroys TMA lookahead.
- Decode K+1 before submitting current WGMMA: correct but provides no tensor
  overlap and only moves the same critical-path work.

## Gates

1. Keep 168 registers, a 56-byte stack, and zero spill loads/stores.
2. SASS must place next-stage LUT/PRMT/STS plus `FENCE.VIEW.ASYNC.S` after
   current QGMMA submission and before its `WARPGROUP.DEPBAR`.
3. Pass exact-NVFP4 M8/M16 for `global_scale=none/expert`.
4. Pass the eight-rank adversarial case where one expert receives 128 tokens.
5. Pass seed-101 M16 quick A/B/B/A before any multi-seed gate.
6. Advance only with non-regressive M8/M16 seeds 101/202/303 and compare the
   winner against optimized W8A8.

The candidate stays in separate implementation files. It adds no persistent
environment variable or runtime argument and is not committed or pushed.
