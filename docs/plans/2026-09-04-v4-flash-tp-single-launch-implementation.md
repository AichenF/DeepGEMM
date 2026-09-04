# V4 Flash TP single-launch MegaMoE implementation plan

This plan implements the approved design in
`2026-09-04-v4-flash-tp-single-launch-megamoe-design.md`.  Every benchmarked
source change follows the AKO sequence: run, append `ITERATIONS.md`, commit.

## Phase 0: freeze evidence and interfaces

- Treat the iteration-138 selected custom graph as the control and preserve
  its exact weights, inputs, routes, numerical checks and CARv2 communicator.
- Add an Nsight launch-count check to the candidate harness.  The candidate
  must expose a path identifier and reject fallback for target shapes.
- Kernel ABI: FP8-E4M3 X plus FP32 group-128 X scales, INT32 top-k IDs, FP32
  top-k weights, transformed MXFP4 W13/W2 data and scales, owned persistent
  workspace, CARv2 symmetric pointers and counters, BF16 output, M/rank/world
  metadata. BF16-to-FP8 X quantization is an upstream, untimed operation.

## Phase 1: one-launch local MegaMoE skeleton

- Add owned files `v4_flash_tp_megamoe.py` and
  `include/v4_flash_tp_megamoe_sm90.cuh`; do not edit the DeepGEMM checkout.
- Parameterize the read-only `megamoe_nvfp4_dev_m` SM90 execution structure
  for H=4096, I-rank=512, E=256, top-k=6, 78 SMs and local expert ownership.
- Initially use graph-stable staged FP8 X/routes only as a bring-up seam.
  Prove exactly one compute launch, repeated graph replay and local output
  correctness.  This milestone is not eligible for a performance claim.

## Phase 2: move route preparation inside the launch

- Add in-kernel state reset, route counts, padded offsets and token/top-k pool
  construction; consume caller-provided FP8 X/X scales directly.
- Use a fully resident 78-CTA grid and local grid barriers.  Route mutation
  across replays must change device task bounds without recapture.
- Remove every input staging/copy/reset node from the timed graph.

## Phase 3: native-style interleaved MXFP4 compute

- Retain the reference 384-thread warp specialization and two-stage task
  mailbox, but remove EP rank ownership and remote token movement.
- Port the selected OCP MXFP4 group-32 dequant path and preserve BF16 numerical
  boundaries around W13, SwiGLU/requant and W2.
- Start with native full-K tasks.  If NCU confirms insufficient parallelism,
  add W13 K-partition tasks/readiness inside the same launch rather than
  spawning another kernel.

## Phase 4: inline route reduction and communication

- Adapt the validated fused k6+multicast push device protocol for TP4 M<=32.
- For M>=64, reduce local route rows to the multicast-bound pull slab and
  inline NVLS sharded two-shot reduction/broadcast after local/rank barriers.
- Add world-size-eight communication specialization and correctness coverage.

## Phase 5: verification and optimization

- Correctness: random/balanced/skew routes; M=8/16/32/64/128; graph mutation;
  repeated communicator phases; TP4 and TP8; independent NCCL/reference gate.
- Nsight: exactly one timed kernel node per candidate replay, with cold-L2
  clear outside events and outside the count.
- NCU: occupancy, tensor/DRAM utilization and barrier stalls after the first
  correct complete kernel.
- Fast screens: paired candidate/control at M8/M32/M128.  Tune BM/BN, expert
  waves, W13 K partition and AR CTA subset only within the one-launch design.
- Final: same-process TP4 10x200 cold-L2 at all five M.  Require geometric-mean
  control/candidate >=1.10x and report every point before declaring success.
