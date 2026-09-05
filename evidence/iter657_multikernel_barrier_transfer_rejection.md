# Iteration 657: reject cluster-aggregated phase barriers

## Motivation from the multi-kernel audit

The frozen multi-kernel branch contains no one-sided commits to cherry-pick,
but each standalone launch obtains hardware phase completion instead of making
every resident CTA join a software grid barrier.  This experiment tested the
narrowest safe transfer of that property while preserving the selected
128-thread CTA, logical task order, MXFP4/TMA/WGMMA bodies, phase ordering and
embedded TP transport.

A tempting residual-CTA-only publication was rejected at design time.  The
CTA owning a residual final-wave task can finish before a slower CTA with no
residual task has completed its preceding full-wave task.  Omitting the latter
CTA's release arrival therefore cannot prove that all W13/W2 global writes are
visible.  Passing ordinary tests would be timing-dependent rather than a valid
memory-order protocol.

## Safe cluster aggregation tested

The default-off prototype launched the unchanged flat grid as Hopper
thread-block clusters.  At each phase boundary every block executed a native
`barrier.cluster.arrive/wait`; cluster rank zero alone joined the existing
packed release-arrival chain; a second native cluster barrier distributed the
global publication back to every member CTA.  Thus all producers remained in
the happens-before chain while global arrivals were reduced by the cluster
size.  Native CuTe/PTX barrier instructions were used rather than
`cooperative_groups::cluster_group`, avoiding the latter's initial resource
inflation.

The experiment used the selected compact-W13 bundle.  Both cluster-size two
and three cubins preserved the production resources:

- M128 split-K2/4: `REG56 STACK32 SHARED2048 LOCAL0`;
- M8 split-K4: `REG63 STACK32 SHARED2048 LOCAL0`.

## Residency and correctness gates

- Cluster size three failed before executing the M128 kernel.  CUDA's active
  cluster query admitted only 504 resident blocks, versus the required
  unchanged 702-block production grid.  The host guard rejected the launch;
  no invalid or reduced-grid timing was collected.
- Cluster size two admitted the full 624-block M8 grid.  Random-route M8,
  seed 20260902, was bitwise equal to the independent same-source multi-kernel
  local result (`cosine=1`, `rel_l2=0`, finite), with 344 padded rows.  Seeding
  all four packed barrier generations at `2^22-1` returned exactly
  `[0,0,0,0]`, validating replay wrap and zero residual counts.

## Cold-L2 phase screen

Physical H20 GPU1, M8 random routes, seed 20260902, separate processes, and a
separate excluded 256 MiB clear immediately before every measured launch.
Inputs were caller-provided FP8-E4M3 activation plus FP32 group-128 scales and
MXFP4 weights.  TP communication was disabled only for this local phase
screen; no external X quantization was run or timed.

| path | route | W13 | requant | W2 | total (us) |
|---|---:|---:|---:|---:|---:|
| ordinary packed, sample 1 | 2.112 | 39.232 | 3.328 | 20.960 | 65.632 |
| cluster-2, sample 1 | 2.688 | 40.384 | 3.776 | 22.464 | 69.312 |
| cluster-2, sample 2 | 2.912 | 40.224 | 3.648 | 22.304 | 69.088 |
| ordinary packed, sample 2 | 2.336 | 39.552 | 3.008 | 21.536 | 66.432 |

The ordinary and cluster-2 mean totals are 66.032 and 69.200 us.  Hardware
cluster aggregation is therefore 4.80% slower in this screen; both candidate
samples lose to both controls.  The two cluster rendezvous per phase cost more
than halving the global packed arrivals.

## Decision

Reject before distributed TP4 timing and remove the experimental source path.
The production CUDA/Python source is restored byte-identical to the Iteration
656 head.  Multi-kernel's phase boundary remains a structural launch advantage,
not a software-barrier fragment that can be profitably transplanted into the
flat one-kernel topology.

Raw artifacts:

- `bench/results/iter657c_cluster3_selected_resources_20260906.log`
- `bench/results/iter657c_cluster3_m128_correctness_20260906.log`
- `bench/results/iter657d_cluster2_selected_resources_20260906.log`
- `bench/results/iter657d_cluster2_m8_correctness_20260906.log`
- `bench/results/iter657e_clusterbarrier_{off,on}_m8_phase_{1,2}_20260906.log`
