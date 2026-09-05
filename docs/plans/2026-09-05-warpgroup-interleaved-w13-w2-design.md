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

Iteration 425 shows that this ownership change halves dynamic spill requests,
but the per-row activation helper still leaves 35.7% of the issue interval at
barriers.  One activation-group task currently invokes the 128-thread helper
eight times.  Each row performs two internal reduction/scale barriers plus a
terminal task barrier, in addition to the scheduler mailbox barrier.  This is
about 25 named-barrier encounters for one ready group.

#### Eight-row activation epilogue (selected after Iteration 425)

Match the Hopper L1 epilogue's tile-level behavior more closely: one WG task
consumes all eight sorted rows of an `(mblock, activation_group)` together.

1. Every lane computes one group-128 column for each valid row, applies the
   exact BF16 rounding/SwiGLU/BF16 rounding contract, overwrites the now-dead
   split-0 gate partial with that exact BF16-rounded float, optionally stores
   the public BF16 activation when its workspace is enabled, and contributes
   eight per-warp maxima to shared memory.
2. After one named barrier, warp 0 maps its 32 lanes to exactly eight
   `(row, source_warp)` values, performs eight independent width-4 max
   reductions, and publishes the eight FP8 scales.
3. After a second named barrier, all lanes reload the reclaimed partial slot,
   quantize all valid rows, and store FP8 bytes.  A terminal named barrier
   publishes completion before lane 0 updates mblock readiness.

Including the existing task-mailbox handoff, this reduces synchronization
from about 25 to four named barriers per activation group.  It adds at most
4 KiB each of coalesced float stores and reloads per group, negligible beside
cold expert-weight traffic, and deliberately avoids keeping eight gate/up
pairs live across a barrier.  The slot is safe to reclaim only after all
`2 * SplitK` producers for that group have release-published completion; no
later consumer reads W13 partials after requantization.  The public activation
workspace remains populated when enabled, while the default fused-quant path
correctly supports its zero-sized activation tensor.

Two fallbacks are intentionally narrower.  A two- or four-row helper lowers
register risk but retains two to four times as many barriers.  Staging all
eight rows in per-WG shared memory avoids the BF16 reload but consumes another
16 KiB per CTA and can create shared-bank pressure.  Use those only if the
eight-row/global-reload binary exceeds 64 registers, increases dynamic local
spills, or fails correctness/performance gates.

#### Static local phases without a task mailbox (selected after Iteration 434)

Source counters show that the first load after the per-task mailbox accounts
for 17.76% of all barrier-stall samples, while the terminal W2 convergence is
another 23.74%.  The dynamic alternation also lowers cold delivered bandwidth
relative to the mapped phase implementation.  Replace the scheduler loop with
three deterministic real-SMID stripes:

1. every WG completes all of its `worker + n * 624` W13 tasks and publishes
   group counters;
2. every WG walks its statically owned activation groups, acquire-waits on
   each exact group flag, runs the eight-row epilogue, and publishes mblock
   readiness;
3. every WG walks its statically owned W2 tasks, acquire-waits on the owning
   mblock flag, and runs W2.

All 128 lanes derive the same task indices directly, so there is no lane-0
task selection, shared task payload, or named-barrier mailbox.  The acquire
loops may reconverge at the first existing task-body synchronization.  The
ordering cannot deadlock: no WG waits for activation until it has exhausted
its own W13 work, and no WG waits for W2 until it has exhausted its own
activation work.  Producers on other WGs therefore always remain runnable.

This sacrifices fine L1/L2 alternation, which has already reduced cold
bandwidth without an endpoint win, but it is not a return to three globally
barriered kernels.  Faster WGs may enter downstream work while slower WGs
finish their producer stripe; the only CTA-wide and whole-grid convergence is
the terminal W2 boundary required by ordered k6 reduction and TP communication.

Iteration 436 shows that placing all three loops directly in the monolithic
entry grows its per-thread caller stack by 64 bytes.  The selected repair is
one noinline boundary per complete phase stripe, not one call per tile.  W13
and W2 phase callees inline their route-GEMM task bodies and preserve
task-to-task optimization; the activation phase similarly owns the full
static group loop.  The kernel entry retains only three sequential calls, so
cross-phase pointer/cursor state is dead at each boundary while task hot paths
do not pay the earlier rejected per-tile ABI overhead.

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

For the eight-row helper, reject immediately if cubin resources exceed 64
registers/thread or one-CTA-per-SM residency.  A passing binary must be
bitwise-equal on all routed rows at M8/M128, improve both endpoints over
Iteration 424 under replay-interleaved cold-L2 TP4 timing, and reduce NCU
barrier stalls without increasing local spill requests.  The final project
gate remains at least 1.10x over the multi-kernel plus CustomAllReduceV2
baseline across M={8,16,32,64,128}; an overlap micro-improvement alone is not
completion.

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

## 2026-09-05 Hopper mailbox correction after Iteration 441

Direct inspection of `megamoe_nvfp4_dev_m` confirms that its SM90 kernel does
not run eight independent GEMMs through three bulk W13/activation/W2 phases.
One resident CTA instead assigns fixed dispatch, A-loader, B-loader, and
math/epilogue roles.  A producer publishes an interleaved L1/L2 task stream
through a two-stage shared-memory mailbox; each role consumes the same payload,
and the L1 epilogue publishes the readiness consumed by L2.  The deadlocked
Iteration 439 bulk-phase outline is therefore not a faithful Hopper port.

The next bounded migration keeps the already-correct TP task bodies and eight
math warpgroups, but changes mailbox transport before attempting a full loader
split.  WG lane 0 remains the task producer.  It writes kind/index, performs a
block-scope release fence, and advances a per-WG shared epoch.  One leader in
each of the four consumer warps acquire-observes the epoch, reads the payload,
and broadcasts it with warp shuffle.  A real task's existing terminal WG
barrier proves that every warp consumed the payload before lane 0 can publish
the next task.  An unavailable-task retry retains one named WG barrier to
prevent producer overwrite; termination is followed by the existing CTA-wide
barrier.  This removes the named barrier and 128 duplicate shared loads before
every useful GEMM/activation task while preserving the dynamic L1/L2 order.

This is deliberately an intermediate Hopper adaptation.  It does not yet
split TMA A/B loading away from the math WGs, does not reuse a decoded weight
tile across multiple token consumers, and does not import EP dispatch/scatter
or NVFP4 arithmetic.  Those larger changes are justified only if the mailbox
transport passes resource, correctness, cold-L2, and NCU gates.

## 2026-09-05 direct Hopper-native audit after Iteration 446

The read-only `megamoe_nvfp4_dev_m` reference is the Hopper SM90/H200 path,
not a Blackwell implementation.  The owned experimental files
`v4_flash_tp_native_megamoe.py` and `v4_flash_tp_native_body.inl` already
retain its main structure nearly verbatim: two dispatch warps, separate A and
B TMA-loader warps, two math/epilogue warpgroups, full/empty stage mbarriers,
and a two-stage interleaved L1/L2 task mailbox.  It is therefore the correct
place to test Hopper role specialization; recreating that topology inside the
self-contained eight-WG path before exhausting the native resource knobs
would duplicate known code.

The direct port is not otherwise shape-equivalent.  The reference is H200
with 132 SMs, EP8, 48 experts per rank, H=6144, I=2048, topk8, and an M-based
BM8/16/24/64/128 heuristic.  V4 Flash TP4 runs on 78-SM H20, has all 256
experts resident on every rank, H=4096, I/rank=512, and topk6.  Its expected
routes per expert are only `6*M/256`, so copying the EP large-M block-M
heuristic would substantially increase padding.  The TP adaptation also
replaces remote expert ownership/return with local bucket formation plus one
final TP all-reduce, removes NVFP4 global scales, and must eventually keep the
selected route-weight-after-W2 numerical boundary.  The current native
experiment instead folds route weight before W2, uses a power-of-two internal
quantizer, and supports TP4 only; it is a performance/reference branch, not a
drop-in final implementation.

Measured native register-dequant resources explain the large gap: 78x384,
168 allocated registers/thread, 100 KiB dynamic shared, one CTA/SM, two math
WGs/SM, 18.78% achieved occupancy, and 55.94% no-eligible cycles at M128.  Its
best screened cold-L2 result remains 117.184 us at M8 and 495.280 us at M128,
versus the same-process multi-kernel control at 71.376 and 305.968 us.

The next bounded native experiment keeps BM8/BN256/BK128, register MXFP4
dequantization, and the exact Hopper task pipeline, but asks whether resource
pressure rather than the framework itself is the limiting factor.  Under an
opt-in two-CTA specialization, reduce the math-role `setmaxnreg` target from
208 to 96 and launch 156 persistent CTAs.  The weighted register budget is
`(2*48 + 2*64 + 8*96)*32 = 31,744` registers/CTA, or 63,488 for two CTAs,
while two 100 KiB shared allocations fit H20's 233,472-byte SM capacity.
Compilation must prove two-block residency and report no static local memory;
M8/M128 full-output correctness and runtime follow only if that resource gate
passes.  Any material dynamic spill or failure to retain two CTAs/SM rejects
the experiment before distributed timing.
