# DeepSeek-V4-Flash TP single-launch MXFP4 MegaMoE design

Date: 2026-09-04

## Goal and definition

Replace the selected five/six-kernel TP4 pipeline with one persistent CUDA
kernel per rank.  Per the corrected serving boundary, the timed operation
begins with FP8-E4M3 `X[M,4096]`, its FP32 group-128 scale `X_scale[M,32]`, and
precomputed `topk_idx/topk_weights [M,6]`, and returns the final TP-summed BF16
`Y[M,4096]`.  BF16-to-FP8 input quantization is upstream and untimed.  Route
preparation, both MXFP4 GEMMs, SwiGLU plus intermediate requantization, route
reduction, communication and replay-state cleanup all execute inside that
launch.  A multi-node CUDA Graph, CUDA dynamic parallelism or device-launched
child kernels do not qualify.

TP4 is optimized for `M={8,16,32,64,128}` with `I_rank=512`; TP8 with
`I_rank=256` must execute the same one-launch contract correctly.  Router/top-k
selection and weight transformation remain outside the layer benchmark.

The performance gate is a same-process candidate/control comparison against
the current selected custom implementation, not an old log: CUDA Graph,
TP-rank maximum, a separate excluded 256 MiB L2 eviction immediately before
every replay, and 10 batches x 200 samples for every M.  The equal-weight
geometric-mean speedup must be at least 1.10x, with all individual M results
reported.  The planning baseline from iteration 138 is 0.176213 ms geometric
mean, corresponding to approximately 0.160194 ms, but only the fresh paired
run can prove the target.

## Reference and ownership boundary

Read `megamoe_nvfp4_dev_m` with `git show` only.  Its useful properties are a
single `<<<num_sms,384>>>` launch, one persistent CTA per SM, warp-specialized
loading/math, a device-side Linear1/Linear2 scheduler, release/acquire
readiness and an in-kernel combine phase.  Its EP expert ownership,
cross-rank dispatch, return scatter, top-k combine and NVFP4 encoding are not
part of the TP implementation.

All new code lives in `DeepGEMM_tp`.  The dirty `DeepGEMM` checkout remains
untouched.  Reuse the selected SM90 MXFP4 dequant/WGMMA implementation and the
SGLang `CustomAllReduceV2` symmetric allocation/protocol already validated in
this worktree.

## Considered approaches

### A. Native-style TP persistent task DAG — selected

Port the reference kernel's 384-thread, one-CTA-per-SM execution skeleton and
rewrite its front and tail for TP.  A device-generated local expert pool feeds
interleaved W13/W2 tasks.  W13 epilogues publish FP8 SwiGLU tiles; W2 tasks
acquire those tiles and publish BF16 route rows.  A final in-kernel phase forms
the ordered weighted k=6 sum and executes the TP collective.

This is the largest implementation change, but it is the only option that can
both meet one launch and plausibly gain 10%: it removes launch boundaries and
intermediate scheduling while allowing W2 to overlap the W13 tail.

### B. Barrier-separated phases using the current kernels' tilings

Rewrite current route quant, W13, activation, W2 and reduction as device
functions separated by grid barriers in a 78-CTA kernel.  This is easier to
reason about, but static per-phase mapping loses the current split-K/task
parallelism.  Earlier persistent W13/W2 experiments underfilled HBM and tensor
pipes.  Retain only as a bring-up oracle, not the performance design.

### C. Parent kernel plus device child launches or graph wrapping

This leaves several GPU kernels and adds launch overhead.  It violates the
acceptance definition and is rejected.

## Kernel data flow

### Persistent launch and roles

Compile shape-specific TP4 specializations and launch exactly 78 CTAs with
384 threads so every block is resident before any grid-wide wait.  Following
the reference structure, dispatch/preparation warps own route metadata and
activation movement, loader warps own TMA and packed-weight dequantization,
and 256 math/epilogue threads own WGMMA and output transforms.  M may select a
different compile-time block policy while still issuing exactly one launch.

### In-kernel preparation

The prologue clears graph-stable route/scheduler state from within the launch;
there is no captured memset node.  It consumes the caller-provided FP8 X and
group-128 scales directly.  Preparation warps consume the precomputed top-k
IDs, count local routes for all 256 experts, form padded expert offsets and
write token/top-k metadata into a contiguous local route pool.  TP ranks use
identical routes and own all experts, so no token is sent to another rank.

Publication uses GPU-scope release stores and consumers use acquire loads.
Grid barriers are legal only because the launch is capped at one resident CTA
per physical SM.  Route IDs and weights may change between graph replays; no
host-derived active-expert count is allowed.

### Interleaved W13 and W2

Use local experts rather than `expert/rank` ownership.  W13 is
`[BLOCK_M,4096] x [1024,4096]`; its paired gate/up epilogue preserves the BF16
boundary, applies SwiGLU, preserves the second BF16 boundary and dynamically
requantizes to FP8 group-128.  It release-publishes completion for the
corresponding intermediate tile.

W2 is `[BLOCK_M,512] x [4096,512]`.  It is claimable only after every W13 N
fragment required by that expert block is ready.  W13 and W2 claims are
interleaved in expert waves using the reference two-stage task mailbox.  The
initial plan uses the reference full-K producer/math pipeline; if profiling
shows the previously observed small-M parallelism loss, add scheduler-visible
two-way/four-way W13 K partitions rather than falling back to another launch.

W2 writes BF16 route rows to graph-stable workspace indexed by original token
and top-k slot.  This preserves the current numerical boundary and gives each
element one writer.

### Local reduction and inline TP all-reduce

After all W2 tasks publish completion, all resident CTAs enter a local grid
barrier.  The tail reduces six BF16 route rows in fixed FP32 order using
`topk_weight * 1.5`, then rounds the local result to BF16.

For TP4 M<=32, inline the validated multicast one-shot push protocol from the
existing fused-k6 kernel.  For TP4 M>=64, write the local result directly to
the multicast-bound pull slab, synchronize ranks with SGLang-compatible
semaphores, and execute sharded NVLS two-shot reduction/broadcast inside the
same kernel.  Nonparticipating CTAs remain resident at a local barrier while a
tuned CTA subset performs the large-message tail; no child kernel is launched.

TP8 receives a world-size-eight specialization using the same symmetric-memory
ABI.  It is correctness-gated before performance tuning.

### Replay cleanup

Before returning, a designated persistent subset resets route counters,
readiness/task cursors and communication phase state after all local and remote
users have completed.  A final local barrier prevents any CTA from leaving
while another still accesses replay state.  Counter protocols remain
compatible with alternating Humming/CARv2 control graphs on the same
communicator.

## Correctness and failure handling

- Validate route counts, offsets, token/top-k metadata and task uniqueness for
  random, balanced and maximal-skew routes at every M.
- Mutate FP8 X/X scales, routes and weights in place between CUDA Graph
  replays to prove that no captured host specialization or stale scheduler
  state exists.
- Compare intermediate W13/SwiGLU quantization, W2 route rows, local k6 output
  and final TP result against the dequantized-MXFP4 golden.  Final gates are
  finite output, cosine >=0.999 and relative L2 <=0.02 on every rank.
- Add bounded debug epochs/error words for bring-up so a bad readiness or
  cross-rank protocol fails diagnostically instead of spinning forever; remove
  timeout overhead from the selected performance specialization.
- Unsupported shapes fall back only at the public dispatcher.  The benchmark
  for the target shapes must verify the one-launch path rather than silently
  accepting fallback.

## Verification and optimization sequence

1. Create an owned single-launch JIT wrapper/header and port only the reference
   scheduler/body pieces required for TP4 H20.
2. Bring up one persistent launch with in-kernel route preparation and direct
   FP8-X consumption plus a trace-only scheduler; prove graph replay and state
   cleanup.
3. Port native-style MXFP4 W13 plus fused SwiGLU/requant and validate its
   intermediate tensor.
4. Enable interleaved W2 and validate route rows/local k6 output.
5. Inline TP4 one-shot and two-shot communication, then run repeated
   distributed correctness and deadlock tests.
6. Use Nsight Systems to prove one timed kernel node and NCU once to choose
   occupancy, WGMMA pipeline and scheduler tuning.
7. Screen M8/M32/M128 with paired cold-L2 samples, then tune block-M/N,
   expert-wave size, W13 K partitioning and AR CTA count.
8. Run TP8 one-launch correctness/run-through.
9. Run the formal TP4 five-M 10x200 paired verdict.  Do not claim completion
   unless kernel count, correctness and >=1.10x geometric-mean speedup are all
   simultaneously proven.
