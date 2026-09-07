# V4 Flash TP Tile-Warp-Specialized Mega-Kernel Design

Date: 2026-09-07

## Goal

Add an isolated experimental MegaMoE path for H20 that performs TP-local route
preparation, MXFP4 W13, SwiGLU plus FP8 requantization, MXFP4 W2, ordered route
reduction, TP all-reduce, and replay cleanup in one persistent CUDA kernel per
rank. The caller supplies prequantized FP8 activation and canonical MXFP4
weights. There is no activation quantization, EP dispatch, EP combine, or
device-launched child kernel in the timed operation.

TP4 is the optimization target for M in {8, 16, 32, 64, 128}. TP8 must compile,
run, and pass correctness. All performance measurements use CUDA Graph replay
and an excluded cold-L2 eviction before every replay.

## Why the Existing Exact Path Is Not the Target

The Hopper EP reference and the current exact experiment already divide a
384-thread CTA into control/loading roles and two WGMMA warpgroups. However,
their outer scheduler advances expert waves and publishes W13 FP8 output to a
global pool before a separately scheduled W2 task reloads it. Removing EP
dispatch and combine does not change that dependency boundary.

The TP implementation instead schedules a flat tile DAG. Expert-major packing
is only a data-layout operation; it does not own FC1-to-FC2 scheduling.

## Selected Microtask

The persistent work item is:

```
(expert-local BM8 route block, TP-local intermediate K128 slice)
```

One work item performs, without relinquishing ownership:

```
W13 gate/up for one K128 output slice
  -> FP32 SwiGLU in registers/shared memory
  -> group-128 FP8 requantization in shared memory
  -> every W2 N128 output tile for that K128 input slice
  -> FP32 W2 partial scratch
```

The FP8 intermediate is never written to global memory. TP4 has four K128
slices per route block and TP8 has two. This exposes enough independent work at
M=8 while avoiding an expert-wave handoff between W13 and W2.

The first version intentionally allows W2 slice partials in graph-stable global
scratch. After all work items finish, the same kernel reduces the slice
partials and the six routes in the required numerical order. A later cluster
variant may replace this scratch with DSM; it is not required for bring-up.

## Persistent CTA and Warp Roles

Launch exactly 78 CTAs with 384 threads and one resident CTA per H20 SM:

- warp 0 owns the global task counter, task decode, and shared task mailbox;
- warp 1 owns activation/scaling movement;
- warps 2 and 3 own the paired MXFP4 packed-weight loads and Mode2 decode;
- warps 4 through 7 are WGMMA consumer warpgroup 0;
- warps 8 through 11 are WGMMA consumer warpgroup 1.

During W13, the two math warpgroups compute the paired gate/up tile. They hand
valid BM8 rows through a shared-memory mailbox for SwiGLU and requantization.
During W2, the two warpgroups consume the same resident FP8 K128 activation and
process two disjoint N128 output tiles at a time. Math warpgroups are repurposed
for the final reduction/communication tail only after the compute queue drains.

Register reconfiguration follows the reference 64/64/256 split. The initial
W13 queue is two-stage to keep the two decoded B operands, activation tile,
mailboxes, and barriers below the SM90 shared-memory limit.

## Synchronization and Termination

One producer publishes every decoded task through a double-buffered shared
mailbox, so all roles observe the same task sequence and termination marker.
TMA/WGMMA queues use separate full/empty mbarriers with exact producer and
consumer arrival counts. W13-to-W2 handoff is CTA-local and uses acquire/release
mailbox barriers; there is no global W13 readiness bitmap.

Each CTA repeatedly claims a microtask using one global atomic counter. On
completion it release-increments a CTA-done counter. The grid-wide transition
to reduction is legal because the launch is occupancy-checked and fixed at one
resident CTA per each of the 78 SMs. Debug builds expose an epoch/error word so
a broken task or barrier protocol fails diagnostically instead of silently
returning incomplete output.

## Reduction and TP Communication

After the compute barrier, persistent CTAs reduce FP32 W2 slice partials per
route, apply the existing route weights and fixed k=6 order, and form the
rank-local BF16 output. The tail reuses the validated production symmetric
workspace ABI and embedded collective:

- TP4 M<=32: multicast one-shot push;
- TP4 M>=64: the selected two-shot/NVLS-compatible large-message path;
- TP8: the supported world-size-eight specialization using the same single
  launch contract.

Communication, state cleanup, and the final local barrier remain inside this
kernel. No fallback kernel is permitted in the experimental benchmark path.

## Isolation

The implementation is exposed as a separately named, default-off
`tp_tile_ws` candidate with its own JIT identity. Production and the rejected
H20-exact experiment remain unchanged. Workspace sizing, 78-SM residency, TP
world size, and supported M values are checked before launch.

## Verification

1. Compile and record registers, stack, shared memory, and occupancy.
2. Validate W13 slice output, shared FP8 requantization, W2 FP32 partials, local
   route reduction, and final output against the existing MXFP4 reference.
3. Run TP4 M={8,16,32,64,128}, including repeated CUDA Graph replay with
   mutated inputs/routes to prove state cleanup.
4. Use Nsight Systems to prove one timed business-kernel node per rank.
5. Run TP8 M=8 and M=128 correctness/run-through before expanding to all M.
6. Only after correctness, compare against production with same-process paired
   cold-L2 samples and then profile the bottleneck. No performance claim is
   made from a smoke test.

## Rejected First-Version Alternatives

An entire route block per CTA eliminates W2 slice scratch but exposes too few
tasks at small M. A multi-CTA cluster/DSM pipeline can eliminate all global
partials but adds remote-shared handoff and cluster-residency risk before the
basic tile pipeline is qualified. Both remain possible follow-ups.
