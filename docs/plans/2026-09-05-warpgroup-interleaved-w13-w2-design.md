# Hopper-EP-inspired warpgroup-interleaved W13/requant/W2 design

## Context

The selected TP4 candidate is one CUDA kernel per rank.  It consumes
prequantized FP8-E4M3 activations plus FP32 group-128 scales, precomputed
routes and MXFP4 weights, then performs route preparation, W13, SwiGLU and
internal FP8 requantization, W2, ordered k6 reduction, TP allreduce and
cleanup.  The current 78-CTA specialization places one 1024-thread CTA on
each H20 SM and gives it eight independent 128-thread WGMMA warpgroups.

Iteration 407 measured the remaining global-phase implementation at M8 and
M128.  W13 plus W2 dominates: the stamped compute intervals are 79.104 and
331.008 us, while the complete embedded collective plus unstamped tail is
only about 9.120 and 18.160 us.  Communication-only work cannot reach the
required 1.10x win over the selected multi-kernel baseline.

The primary reference is the Hopper implementation on branch
`megamoe_nvfp4_dev_m`, specifically
`sm90_nvfp4_mega_moe_h200_fused_body.inl` and
`InterleavedMegaMoEScheduler` in `scheduler/mega_moe.cuh`.  It does not
execute all Linear1 work, cross a global boundary, and then execute all
Linear2 work.  A weight-loader warp dynamically claims persistent
`BlockPhase::Linear1` and `BlockPhase::Linear2` tasks, publishes each task
through a two-stage shared-memory mailbox, performs a minimal L1-only warm-up,
and then alternates L2/L1.  The L1 epilogue release-ORs a per-pool-block
readiness mask after its SwiGLU/FP8 TMA store; the L2 activation loader waits
on that mask with an acquire load.  EP dispatch/combine and the reference's
NVFP4 layout are not part of this port, but this Hopper scheduling and
readiness contract is the model for the TP implementation.

## Prior rejected designs

This design must not repeat these measured failures:

- Iterations 162-163 used a CTA-wide per-task global dispatcher.  Global
  atomic claims/completions, polling, all-CTA mailbox barriers and increased
  register pressure made M8 570-578 us.
- Iteration 164 used static groups of sixteen 128-thread CTAs with two cohort
  barriers per mblock.  M8 was 105.280 us, slower than the then-selected
  phase implementation.
- Iterations 236-237 mixed cold W13 and W2 traffic only in the residual W13
  tail.  The 32-CTA version added roughly 11 us of compute time.
- Iteration 292 fused activation and W2 in 16-CTA cohorts, but seven cohort
  barriers and constrained W2 assignment made its tail about 4.9 us slower.
- Iteration 413 let all 624 resident warpgroups claim individual W13,
  requant and W2 tasks through global atomics.  It was numerically correct but
  41-47x slower than control because roughly 19,440 short-task scheduling
  events at M128 contended on a handful of global counters.

## Alternatives

### 1. CTA-local coarse readiness pipeline (selected after Iteration 413)

Thread 0 of each resident CTA chooses one coarse L1 macro or ready L2 chunk
and broadcasts that payload once to all eight warpgroups.  A split-K4 L1
macro assigns the four gate and four up split tasks for one activation group
to the eight warpgroups.  A split-K2 macro assigns two activation groups at
once.  The same CTA immediately performs the matching eight-row
SwiGLU/requant epilogue, mirroring Hopper's ownership of the L1 epilogue.
Only a coarse per-mblock completion is globally published.

This is the closest adaptation of the Hopper reference that preserves the
already faster H20 MXFP4 task body.  The reference dedicates loader warps and
two math warpgroups inside a 384-thread CTA; our current task body instead
combines loading, register dequantization and math in one self-contained
128-thread warpgroup.  Therefore one mailbox fans a macro payload to eight
self-contained consumers rather than separate A/B-loader and math roles.
Once an mblock's four activation groups are ready, four W2 chunks are
published; one CTA consumes a chunk with one N128 tile per warpgroup.  The
CTA alternates one ready W2 chunk with its statically owned L1 macros.

### 2. Static two-CTA mblock cohorts

Two physical CTAs own each mblock, complete W13 and matching requant groups,
cross one cohort barrier, then execute W2.  This has fewer atomics but tends
to move all cohorts in lockstep and cannot deliberately mix W13/W2 warpgroups
on one SM.  It is a fallback if scheduler overhead dominates.

### 3. Full Hopper role-specialized MegaMoE port

Port the Hopper reference's dispatch-role slots, activation loader, weight
loader and two math warpgroups into a new H20 kernel, replacing its dispatch
work with TP-local route preparation/communication.  This offers the highest
architectural freedom but is a substantially larger rewrite and repeats
unresolved native MXFP4 register-dequant issues.  It is not the first
experiment.

## Dependency granularity

For TP4, `Intermediate/rank=512`, W13 produces 1024 columns in eight N128
tiles, the internal activation has four group-128 columns, and W2 produces
4096 columns in 32 N128 tiles.

For each `(mblock, activation_group)`:

1. Two W13 N128 tiles provide the gate/up pair.
2. Every split-K slice of those two tiles must finish, so the readiness count
   is `2 * SplitK`.
3. Those split tasks are assigned together to one CTA macro, so a CTA barrier
   directly establishes readiness without a global fragment counter.
4. The same eight warpgroups execute one requant task per routed BM8 row.
5. CTA thread 0 release-increments the mblock's ready-group count by one
   (split-K4) or two (split-K2).
6. When all four groups are complete, four W2 chunks are release-published;
   each chunk contains eight N128 tasks, one per warpgroup.

Thus requant can overlap later W13 groups.  W2 starts only after the full K512
activation for an mblock is ready, preserving the existing W2 task body and
numerical order.

## Scheduler

The replay-local scheduler contains:

- static per-CTA L1 macro cursors, requiring no global L1 claim;
- one completed-activation-group counter per possible mblock;
- a W2-ready chunk queue plus head/tail;
- a terminal W2-chunk count used only for loop exit.

Queue entries use zero as unpublished and store the encoded index plus one.
The route-preparation phase clears the scheduler slab and the existing phase-0
whole-grid barrier publishes both route metadata and zeroed scheduler state.

Each CTA scheduling iteration follows the Hopper scheduler's bounded
L1-warmup/alternation policy, adapted so a CTA never claims unavailable
downstream work:

1. issue one statically owned L1 macro;
2. try one acquire-visible W2 chunk;
3. alternate L1 and W2 while both exist;
4. after exhausting local L1 macros, drain published W2 chunks;
5. only nanosleep when local L1 is exhausted and the next W2 queue entry is
   not yet published.

No CTA claims an unpublished task and waits on it, avoiding the
producer-consumer cycle that requires a larger warm-up in the reference
implementation.

## Memory ordering

All task-body lanes finish their ordinary global stores before their
warpgroup's named barrier, then a CTA barrier joins all eight consumers.
CTA thread 0 performs a GPU-scope release atomic or release store to publish
readiness.  W2 consumers use GPU-scope acquire loads before reading
qactivation.  This follows the CTA-release pattern already validated in
Iteration 163, but publishes once per coarse macro/chunk rather than once per
N128 task.

There is no global barrier between W13, requant and W2.  The existing packed
whole-grid phase-3 barrier remains after the terminal W2 count so all `down`
writes are published before ordered k6 reduction and the embedded TP
collective.  The collective and its numerical order are unchanged.

## Resource control

The path is guarded by a new default-off compile-time flag and is initially
valid only with the 78-CTA/eight-WG specialization.  The scheduler loop must
not keep phase payloads live across the large inlined GEMM body.  Phase task
calls will use narrowly scoped wrappers if needed; the first gate rejects a
binary that exceeds 64 registers/thread, grows the selected stack materially,
or cannot retain one 1024-thread CTA per SM.

The existing real-SMID table remains available for the phase control.  The
coarse path uses static CTA-strided L1 ownership plus a dynamic W2-chunk
queue, so its placement must be profiled independently rather than claiming
the Iteration 399 mapping benefit carries over.

## Validation

The implementation proceeds through these gates:

1. static scheduler-state bounds, Python compilation and JIT/resource audit;
2. TP-disabled M8 and M128 cold-L2 execution against the selected local route
   tensor, including all routes rather than block 0;
3. repeated TP4 M8/M128 correctness with the existing embedded multicast and
   P2P two-shot collectives;
4. replay-interleaved same-process TP4 A/B, at least two batches x ten
   individually cold-L2 samples per arm for screening;
5. NCU comparison of duration, no-eligible cycles, SM/DRAM throughput, local
   spills and executed instructions;
6. only after an endpoint win, all M={8,16,32,64,128} cold-L2 formal timing,
   exactly-one-business-kernel Nsight proof and TP8 correctness/runability.

Any correctness drift, deadlock, occupancy loss or latency regression is
logged and committed before the next repair.  The selected phase scheduler
remains the fallback until the new path wins the required gates.
