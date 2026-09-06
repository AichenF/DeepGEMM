# Iteration 721g: reject W13 ping-pong task state

## Candidate

`V4_SINGLE_LAUNCH_W13_COMPACT_PINGPONG_STATE=1` alternates two sets of W13
TMA mbarriers and route metadata in the M128 compact phase.  It retains the
ordinary nonpersistent K-loop generations and removes the caller's post-task
CTA rendezvous, leaving the following task's entry rendezvous as the sole
task-boundary synchronization.

## Static gate

The fresh TP4+FP16-W2 candidate extension is
`v4tp_f3a43f3d30c1dfc2e815_v178mspec`.  The exact M128 split-K2 and split-K4
single-kernel entries retain `REG56`, `SHARED2048`, and `LOCAL0`, so the
selected 702x128 (nine CTAs/SM) residency tier is unchanged.  The caller
stack nevertheless grows from the fresh flag-zero control's 32 bytes to 48
bytes.  The exact split-K2 compact-W13 callee contains one `STL` and one
`LDL`, versus no fixed local allocation.

## Correctness

On physical H20 GPU1, M128 random routes (seed 20260902), 1,992 padded rows,
and W13 split-K2, the compute-only cold-L2 check passes:

- cosine versus the independent FP32 multi local reference:
  `0.9999993512848424`;
- relative L2: `0.001139531250097178`;
- all finite;
- all packed grid-barrier words wrap from generation `2^22-1` to zero.

The nonzero error is the already-qualified opt-in packed-FP16 W2 accumulator,
which is enabled identically for this experiment; the W13 task-state change
adds no new numerical mode.

The complete TP4 graph, including embedded P2P two-shot all-reduce, also
passes the independent final-output check.  Against the multi-kernel graph
in that run it measures 0.365072 ms versus 0.320176 ms median, or 14.02%
slower.  That comparison is not used to attribute the task-state delta.

## Paired cold-L2 flag A/B

- physical H20 GPUs 1, 5, 6, and 7; maximum latency across ranks;
- one process loads fresh flag-zero and flag-one extensions;
- identical weights, activations, random routes, and one shared SGLang
  `CustomAllReduceV2` communicator;
- complete single-business-kernel MegaMoE graph including embedded TP4
  all-reduce in both arms;
- five batches x fifty samples = 250 samples per arm after ten alternating
  warmups;
- per-sample A/B then B/A ordering;
- a separate 256 MiB L2 clear immediately precedes every graph replay and is
  excluded from CUDA event timing.

| Arm | Min (ms) | Median (ms) | Max (ms) | Batch medians (ms) |
|---|---:|---:|---:|---|
| flag-zero control | 0.345760 | 0.358880 | 0.881568 | 0.348320, 0.349008, 0.364576, 0.360576, 0.359120 |
| ping-pong candidate | 0.346400 | 0.358944 | 0.367520 | 0.348384, 0.349120, 0.364208, 0.360272, 0.359424 |

`control / candidate = 0.9998217x`: the candidate is 0.0178% slower.  Batch
directions are mixed, so the result is neutral within noise rather than a
speedup.  Cross-implementation outputs are bitwise identical on every rank
(cosine 1, relative L2 0, max absolute difference 0, all finite).

## Decision

**Reject.**  Removing the static task-tail barrier does not reduce complete
graph latency.  The barrier is evidently hidden or negligible beside the
long cold-weight W13 work, while alternate-bank address selection introduces
one local spill pair.  This closes the bounded task-boundary optimization;
the next candidate must change phase execution/ownership rather than merely
remove this rendezvous.

## Artifacts

- `bench/results/iter721b_w13_pingpong_jit_20260906.log`
- `bench/results/iter721b_w13_pingpong_m128_resources_20260906.log`
- `bench/results/iter721b_w13_pingpong_split2_local_sass_20260906.log`
- `bench/results/iter721c4_w13_pingpong_m128_correctness_20260906.log`
- `bench/results/iter721d4_w13_pingpong_tp4_m128_paired_cold_20260906.log`
- `bench/results/iter721f_w13_pingpong_tp4_m128_flag_pair_cold_20260906.log`
