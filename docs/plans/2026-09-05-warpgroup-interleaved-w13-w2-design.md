# Warpgroup-interleaved W13/requant/W2 design

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

The B200 EP MegaMoE reference does not execute all Linear1 work, cross a
global boundary, and then execute all Linear2 work.  It schedules persistent
`BlockPhase::Linear1` and `BlockPhase::Linear2` tasks, publishes Linear1
epilogue readiness per pool block, and permits Linear2 only after the required
Linear1 fragments are release-published.  Its SM90 evolution uses a
two-stage task mailbox and an L1-only warm-up before alternating L2 and L1.
EP dispatch/combine and Blackwell tensor-memory details are not part of this
port; only its dependency and scheduling principles apply.

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

## Alternatives

### 1. Warpgroup-local dynamic readiness DAG (selected)

Each of the eight resident warpgroups independently selects a ready task.
Lane 0 of the warpgroup performs scheduler atomics and broadcasts the payload
with warp shuffles.  Task execution uses the existing private shared-memory
slab and named barrier for that warpgroup.  No CTA-wide mailbox or task-loop
`__syncthreads()` is allowed.

This is the closest H20 adaptation of the reference scheduler because a
128-thread WGMMA warpgroup is already a self-contained loader/math execution
unit.  It also permits W13 and W2 warpgroups to coexist on the same SM, which
can fill the 35.7% no-eligible cycles observed in Iteration 402.

### 2. Static two-CTA mblock cohorts

Two physical CTAs own each mblock, complete W13 and matching requant groups,
cross one cohort barrier, then execute W2.  This has fewer atomics but tends
to move all cohorts in lockstep and cannot deliberately mix W13/W2 warpgroups
on one SM.  It is a fallback if scheduler overhead dominates.

### 3. Full role-specialized MegaMoE port

Port the reference's producer, activation loader, weight loader and math roles
into a new H20 kernel.  This offers the highest architectural freedom but is
a substantially larger rewrite and repeats unresolved native MXFP4
register-dequant issues.  It is not the first experiment.

## Dependency granularity

For TP4, `Intermediate/rank=512`, W13 produces 1024 columns in eight N128
tiles, the internal activation has four group-128 columns, and W2 produces
4096 columns in 32 N128 tiles.

For each `(mblock, activation_group)`:

1. Two W13 N128 tiles provide the gate/up pair.
2. Every split-K slice of those two tiles must finish, so the readiness count
   is `2 * SplitK`.
3. The final W13 completion release-publishes that activation group.
4. Eight independent requant tasks, one per routed BM8 row, consume the ready
   group.
5. The final requant task increments the mblock's ready-group count.
6. When all four groups are complete, the mblock's 32 W2 N128 tasks become
   claimable.

Thus requant can overlap later W13 groups.  W2 starts only after the full K512
activation for an mblock is ready, preserving the existing W2 task body and
numerical order.

## Scheduler

The replay-local scheduler contains:

- monotonically claimed W13 task index;
- four W13-fragment completion counters per possible mblock;
- an activation-group ready queue plus head/tail;
- activation row completion counters and per-mblock ready-group counters;
- a W2-ready mblock queue plus head/tail;
- W2 N-tile claim/completion counters;
- a terminal W2 count used only for loop exit.

Queue entries use zero as unpublished and store the encoded index plus one.
The route-preparation phase clears the scheduler slab and the existing phase-0
whole-grid barrier publishes both route metadata and zeroed scheduler state.

Each warpgroup scheduling iteration follows a B200-like bounded policy:

1. immediately service a ready activation group;
2. service a ready W2 tile unless the warpgroup owes an upstream turn;
3. claim one W13 task if any remain;
4. after downstream work, force one upstream attempt before another W2 task;
5. only nanosleep when all W13 work is claimed and no published downstream
   work is currently available.

The policy is compile-time selectable between downstream-first and strict
alternation for cold-L2 A/B testing.  No warpgroup may claim an unpublished
task and wait on it, avoiding the producer-consumer cycle that requires a
larger warm-up in the reference implementation.

## Memory ordering

All task-body lanes finish their ordinary global stores before the
warpgroup's named barrier.  Lane 0 then performs a GPU-scope release atomic or
release store to publish readiness.  Consumers use GPU-scope acquire loads
before reading partials or qactivation.  This follows the CTA-release pattern
already validated in Iteration 163 but reduces its synchronization scope to
the participating warpgroup.

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
dynamic path intentionally replaces logical-worker striping with readiness
scheduling, so its placement must be profiled independently rather than
claiming the Iteration 399 mapping benefit carries over.

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
