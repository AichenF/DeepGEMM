# V4 Flash TP production dependency-ready tail overlap design

Date: 2026-09-07

## Decision

Keep the selected production kernel's 128-thread, one-warpgroup GEMM worker and
its occupancy-derived resident grid. Replace the fixed W13-to-activation,
activation-to-W2, and W2-to-local-reduce/communication phase transitions with
dependency-aware block publication. Permit cross-phase execution only during
the measured tail of W13 and for completed W2 communication chunks.

Do not add fixed producer/consumer warp specialization in this iteration. It
would change CTA width, register/shared-memory use, resident-worker count and
the cross-phase scheduler simultaneously, preventing causal attribution. A
warp-specialized worker remains a separate follow-up A/B experiment.

## Baseline and acceptance boundary

The source baseline is the current selected production path on branch
`tpmegamoe_single_launch`, not the incorrect exact-H20 experiment. It is one
persistent CUDA kernel per TP rank with 128 threads per CTA and an
occupancy-derived resident grid: normally eight CTAs per H20 SM and nine for
the selected M128 compact specialization.

The timed API begins with caller-owned FP8 activation and MXFP4 weights and
ends with the TP-summed BF16 result. Route alignment, W13, SwiGLU plus FP8
requantization, W2, local route reduction, and the existing embedded TP
collective remain in the same kernel launch. CUDA Graph replay and graph-stable
workspace semantics must not change.

Every comparison uses TP4, cold L2 before every measured replay, rank-maximum
latency, and M in `{8,16,32,64,128}`. TP8 is a mandatory correctness and
run-through gate. The experiment is selected only if it is correct and improves
the equal-weight five-M geometric mean without a material regression at any M.

## Why retain multiple thin CTAs

Persistent describes CTA lifetime, not a requirement to launch one CTA per SM.
Production uses a resident pool of thin CTAs, each owning one independent
N128 GEMM task at a time. The 8--9 CTAs resident on each SM expose 32--36
warps, hide TMA/WGMMA scoreboard latency with independent tiles, and reduce
the tail from irregular routed expert blocks.

The exact-style 384-thread worker has explicit dispatch/load/math roles but
uses enough shared memory and registers to permit only one CTA per SM. Prior
profiling showed long no-eligible-warp and barrier-stall intervals. Therefore
the first dependency-aware experiment retains inter-CTA latency hiding and
the already validated GEMM task body.

## Existing production flow

The selected schedule is globally phased:

```
route alignment
  -> grid barrier
W13 flat tasks
  -> grid barrier
activation + FP8 requantization
  -> grid barrier
W2 flat tasks
  -> grid barrier
local route reduction + embedded all-reduce
  -> replay cleanup
```

The GEMM worker already has an intra-task asynchronous TMA/mbarrier pipeline.
The optimization in this design is cross-task and cross-phase scheduling; it
does not alter MXFP4 decode, WGMMA issue, accumulator mapping, or numerical
boundaries.

## Dependency model

Use small graph-stable state arrays in the existing symmetric workspace. Each
entry includes a replay generation so stale completion from an earlier CUDA
Graph replay cannot satisfy a dependency.

### W13 publication

The readiness unit is one routed BM8 block and one W13 gate/up intermediate
group. Every W13 task that contributes to that unit increments its completion
counter after its global output store is visible. The last contributing task
performs a device-scope release publication.

Activation consumers use acquire loads. A consumer may claim an activation
unit only after publication; no resident CTA is allowed to claim an unready
unit and spin while holding the worker slot.

### Activation publication and W2 admission

Activation and FP8 requantization publish readiness for the corresponding
intermediate group. The initial W2 implementation remains a full-local-K task:
a routed block becomes W2-ready only when all intermediate groups required by
that W2 task have been published. This avoids adding split-K W2 reduction or a
new numerical boundary in the first experiment.

### W2 chunk publication

Partition the hidden output into the same chunks used by the selected embedded
collective. Each completed W2 output task increments the counter for its
`(token block, hidden chunk)`. The last local producer release-publishes that
chunk after all routed expert contributions needed by the local reduction are
visible.

The consumer performs the existing deterministic local weighted reduction and
then the existing M-dependent collective protocol: multicast one-shot for the
selected small-M cases and P2P two-shot for the selected M64/M128 cases. The
communication protocol and arithmetic order are not changed in this iteration.

## Hybrid bulk/tail scheduler

Do not replace the whole kernel with a global atomic work queue. Preserve the
current static/flat bulk schedule for locality and low overhead.

1. Route alignment still ends with a correctness barrier because all task
   bounds and route metadata must be visible before bulk W13 begins.
2. While W13 remaining work exceeds the compile-time tail threshold, every CTA
   executes only W13 tasks in the existing order.
3. In the final W13 wave, a CTA that has no more assigned W13 work probes a
   ready bitmap. It may claim a published activation unit, then a W2-ready
   routed block. A bounded priority rule always prefers remaining upstream
   W13 work, so downstream polling cannot starve producers.
4. W2 remains the only bulk weight stream once all W13 producers finish. This
   prevents arbitrary W13/W2 interleaving from doubling the cold-L2 working set.
5. Communication is admitted only when a W2 chunk is dependency-ready and the
   number of outstanding W2 tasks is below a per-M tail threshold. A bounded
   CTA subset performs communication so the remaining workers can drain W2.
6. A final local completion barrier remains before replay-state cleanup and
   kernel exit. It is not on a compute-phase transition and cannot expose stale
   generation state to the next graph replay.

Ready work is represented by bitmaps plus completion counters, not by CTAs
spinning on arbitrary dependency records. Claims use atomic bit operations and
exactly-once ownership. Release/acquire ordering is explicit at every global
producer-consumer edge.

## Deadlock and resource constraints

- Never let all resident CTAs wait on downstream readiness. W13 production has
  strict priority until the upstream remaining count reaches zero.
- A CTA checks readiness only between complete GEMM tasks, after all WGMMA and
  TMA operations associated with its reusable shared storage have completed.
- Scheduler state must not lengthen the hot GEMM function's live ranges. Keep
  counters/bitmaps in a small out-of-line scheduler helper and verify register,
  local-memory, shared-memory, and occupancy deltas after JIT compilation.
- Generation reset occurs only after local and remote communication users have
  finished. Debug builds use bounded progress/error words; selected builds
  remove timeout polling from the hot path.
- TP4 and TP8 use world-size-specific expected counts. No TP4 constant may be
  reused implicitly by TP8 communication readiness.

## Tunable surface

Keep the initial surface deliberately small:

- W13 tail start: disabled, one resident wave, or two resident waves.
- Maximum activation/W2 tail consumers: fixed small CTA count or a fraction of
  the resident grid.
- Communication start: disabled, final W2 wave, or final two W2 waves.
- Maximum communication consumers: protocol-specific fixed CTA subset.

All knobs are compile-time/JIT specialization choices keyed by M and world
size. No host-side routing-derived specialization is allowed inside a captured
graph.

## Verification sequence

1. Freeze a fresh same-source production control and record compiler resources.
2. Add state allocation and generation initialization with overlap disabled;
   prove bitwise-equivalent scheduling and unchanged latency within noise.
3. Replace only W13-to-activation readiness; test TP4 M8/M128 random and maximal
   skew routes under many graph replays, then screen cold-L2 performance.
4. Add activation-to-W2 admission at the W13 tail and repeat correctness,
   compute-sanitizer, deadlock stress, and resource checks.
5. Add W2-chunk-to-local-reduce/AR publication without overlap, then enable
   communication only at the tail and verify both one-shot and two-shot paths.
6. Run TP8 correctness/run-through for all five M values.
7. Run paired TP4 cold-L2 screens, followed by at least five independent formal
   repeats per M. Report min/median/max and equal-weight geometric mean against
   the unchanged multi-kernel and production controls.

If readiness bookkeeping regresses the overlap-disabled control, or no
tail/chunk setting improves the five-M geometric mean, reject the experiment
and restore the selected production schedule rather than carrying dormant hot
path overhead.
