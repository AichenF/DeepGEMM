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
- Iterations 414-417 reduced those claims with a CTA-wide macro mailbox and
  recovered 95% of the collapse, but remained about 2x slower.  NCU measured
  48.03% of the issue interval stalled at CTA barriers, 63.25% no-eligible
  cycles and only 27.44% DRAM throughput.

## Alternatives

### 1. Static-WG stripes with coarse readiness (revised after Iteration 421)

Each resident warpgroup retains a real-SMID static stripe for W13,
activation groups and W2.  There is no global task claim and no CTA-wide
task-loop barrier.  W13 tasks release-increment a distributed counter for
their exact `(mblock, activation_group)`.  The final gate/up split completion
only release-publishes that group.  Its statically assigned activation
warpgroup acquire-observes the flag, executes the group's eight routed-row
SwiGLU/requant operations, and then release-increments the mblock's
four-group counter.  The final group publishes one coarse W2-ready flag for
the mblock.

This is the closest adaptation of the Hopper reference that preserves the
already faster H20 MXFP4 task body.  The reference dedicates loader warps and
two math warpgroups inside a 384-thread CTA; our current task body instead
combines loading, register dequantization and math in one self-contained
128-thread warpgroup.  Each WG therefore owns a small named-barrier mailbox
and alternates its static W13 stripe with acquire-visible activation and W2
work.  Scheduler cursors and bounded completion masks live in shared memory,
so they are not kept live through the large noinline GEMM helpers.  Each WG
scans only its own small activation/W2 stripe, avoiding both a global claim
and head-of-line waiting on an unready mblock.

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
3. The final split completion acquires the distributed counter's release
   chain and release-publishes the group-ready flag.
4. The statically assigned activation warpgroup acquire-tests that flag and
   executes eight requant tasks, one per routed BM8 row.
5. Its lane 0 release-increments the mblock's ready-group count by one.
6. When all four groups are complete, one W2-ready flag is release-published;
   every statically assigned W2 task acquire-tests this flag.

Thus requant can overlap later W13 groups.  W2 starts only after the full K512
activation for an mblock is ready, preserving the existing W2 task body and
numerical order.

## Scheduler

The replay-local scheduler contains:

- one static W13 cursor plus activation-group and W2 completion masks per
  warpgroup, stored in shared memory and requiring no global task claim;
- four W13 split-completion counters per possible mblock;
- four W13 group-ready flags, one completed-activation-group counter and one
  W2-ready flag per possible mblock.

The route-preparation phase clears the scheduler slab and the existing phase-0
whole-grid barrier publishes both route metadata and zeroed scheduler state.

Each warpgroup scheduling iteration follows the Hopper scheduler's bounded
L1-warmup/alternation policy, adapted so a WG never claims unavailable
downstream work:

1. issue one statically owned W13 task;
2. scan the WG's bounded activation and W2 masks for acquire-ready work;
3. alternate W13 and downstream work while both exist;
4. after exhausting local W13, drain its ready activation and W2 tasks;
5. only nanosleep when local W13 is exhausted and all remaining downstream
   work is not yet published.

No CTA claims an unpublished task and waits on it, avoiding the
producer-consumer cycle that requires a larger warm-up in the reference
implementation.

## Memory ordering

All task-body lanes finish their ordinary global stores before the
warpgroup's named barrier.  Lane 0 performs a GPU-scope release atomic or
release store to publish readiness.  The separately assigned activation WG
acquire-loads its group flag; its task-mailbox named barrier transfers that
observation to all epilogue lanes without an extra post-W13 barrier.  W2
scheduler lane 0 similarly acquire-loads the mblock flag and publishes the
chosen static task through its named-barrier mailbox.  No CTA barrier occurs
inside the task loop.

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

The path reuses the existing real-SMID table for both static W13 and W2
stripes.  Its additional readiness checks and epilogue-owner imbalance must
still be profiled independently rather than claiming the Iteration 399
mapping benefit carries over.

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
