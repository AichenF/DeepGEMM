# SM90 NVFP4 Fused and FP8-Derived Split Follow-up

## Objective

Continue two independent paths toward optimized W8A8 latency. The fused path
keeps the current braided three-stage Pro winner. The split path is rebuilt
from `aichenf/megamoe_sm90_opt` at `be7c5a3`, not from the legacy NVFP4 split
bodies that were tuned primarily for the large-M BN128 deployment range.

Both paths retain standard ModelOpt-compatible E2M1 weights with one UE4M3
scale per 16 values. A lossless model-load prepack is allowed; persistent FP8
weights, request-time repacking, new runtime controls, commits, and pushes are
not.

## Split Alternatives

Three producer layouts have already been rejected: a 512-thread CTA with four
dequant warps, a compact two-warp producer that serialized TMA and dequant, and
a one-warp producer that decoded all 128 rows. Register-source WGMMA and
partial K32 shared-memory publication were also rejected by endpoint timing or
correctness.

The selected design keeps the optimized FP8 compact frontend:

- 64 dispatch threads, 64 TMA threads, and two 128-thread math warpgroups;
- BM64, BN128, BK128, dynamic swapAB N buckets, phase-specific L1/L2 scheduling,
  cleanup, combine, and epilogues from the FP8 implementation;
- one packed 80-byte standard-NVFP4 row and one transient 128-byte FP8 row in
  shared memory for each weight row;
- math-side dequantization after the packed-stage full barrier, so the TMA
  warps remain free to prefetch later stages.

BN128 gives each math warpgroup 64 weight rows but 128 threads. Threads are
paired per row: one thread decodes K0/K1 and the other K2/K3 from the lossless
braided deployment layout. Their stores are disjoint, followed by one
warpgroup barrier before WGMMA. This halves decoder work per thread without a
new producer role or dequant barrier. The expected pipeline depth is five
stages versus the FP8 kernel's seven.

## File Boundary

The candidate uses separate L1 and L2 implementation bodies plus a candidate
header/runtime/API. Existing fused and split bodies remain byte-for-byte
unchanged until a candidate wins:

- `sm90_nvfp4_mega_moe_fp8_split_l1_body.inl`
- `sm90_nvfp4_mega_moe_fp8_split_l2_body.inl`

The two bodies preserve the FP8 phase boundary. L1 owns dispatch, token pull,
SwiGLU quantization, and L1-output publication. L2 owns ready-state
consumption, final BF16 scatter/combine, statistics, and cleanup.

## Fused Follow-up

Fused experiments remain independent of the split port. The first low-risk
screen is a warp-distributed LUT cache for the braided decoder: each lane owns
one common scale entry and runtime scale indices use warp shuffles, with a
shared-LUT fallback. It is tested only in the isolated row decoder before any
real-kernel integration.

## Verification

1. Bit-exact half-row dequant against the full braided decoder for every E2M1
   code, all valid UE4M3 scales, and randomized packed rows.
2. Fresh-JIT MegaMoE correctness for Flash I=2048, middle I=2560, and Pro
   I=3072 at M=8/16/32/64, both global-scale modes; add M=128 after the first
   four pass.
3. Inspect each L1/L2 cubin for registers, stack, local traffic, dynamic shared
   memory, and the intended BN128 swapAB WGMMA shapes. Reject spills or a
   broken full/empty barrier protocol.
4. Seed-101 endpoint screen against the current best fused NVFP4 and optimized
   W8A8 under identical routing. Advance only winning points to three-seed
   A/B/B/A; do not add an exact-M policy gate.

No experiment is committed or pushed.

## Ownership Update

The user delegated the FP8-derived split implementation to another agent.
This workstream retains the audit and design as handoff material but does not
modify split kernels. Its active optimization work remains on the fused path.
