# Native split-weight/scale TMA design

## Context and target

The selected TP4 native Hopper MegaMoE is one CUDA business launch containing
device route preparation, MXFP4 W13, SwiGLU plus intermediate FP8 group-128
requantization, MXFP4 W2, ordered top-k6 reduction, TP all-reduce and replay
cleanup.  Iteration 472 measures a five-M independently cold-L2 geometric mean
of 0.229277 ms versus 0.178311 ms for the selected multi-kernel control, so the
native path remains 28.58% slower and the required 1.10x win is not achieved.

The selected normalized-scale transform limits every per-expert residual E8M0
code to 1..12, but the native transport still inherits an 80-byte record per
N-row/K128 tile: 64 bytes of packed MXFP4, eight bytes containing four scales
duplicated twice, and eight zero bytes.  The RS decoder consumes only one byte
per K32 group.  This design removes the duplicate and padding traffic without
changing the checkpoint format, caller input contract or numerical boundaries.

## Chosen layout

Keep packed weights and residual scales in separate model-load tensors:

- packed weights: one 64-byte K128 record per output row;
- residual scales: four bytes per output row, grouped as four rows in each
  16-byte record;
- one compute tile therefore loads `64 B x 256 = 16 KiB` of weights and
  `16 B x 64 = 1 KiB` of scales, versus `80 B x 256 = 20 KiB` today;
- transport reduction is 3 KiB per N256/K128 tile, or 15%;
- both TMA inner boxes and every global stride remain 16-byte aligned.

At model load, scales are transformed from `[E,N,K/32]` to contiguous
`[E,N/256,K/128,64,16]`.  Each 16-byte item contains four consecutive rows,
with four K32 scale bytes per row.  For tile-local output row `r` and K32 index
`k`, the RS consumer reads
`scale[(r >> 2) * 16 + (r & 3) * 4 + k]`.

## Kernel pipeline

The B producer keeps the existing four-stage mbarrier pipeline.  For each
N256/K128 task, its elected lane issues two asynchronous copies against the
same full barrier:

1. a `64 x 256` packed-weight TMA into the stage's weight region;
2. a `16 x 64` scale TMA into the stage's scale region.

It then calls `arrive_and_expect_tx(16 KiB + 1 KiB)`.  DeepGEMM's SM90 FP8
mainloop already uses multiple TMA copies with one transaction barrier, so this
does not introduce a new synchronization protocol.  Math warpgroups keep the
same WGMMA issue order, K128 batching, accumulator lifetimes, scheduler and TP
collective.  Only exponent addressing changes.

The post-GEMM combine alias remains the shared-memory high-water mark, so this
change is expected to reduce global/L2/TMA bytes but not CTA occupancy.  M128 is
the likely beneficiary; M8 may be neutral or regress slightly from the second
TMA issue, which must be measured rather than assumed.

## Compatibility and rollback

Add `V4_NATIVE_SPLIT_WEIGHT_SCALE_TMA`, initially default-off.  The existing
80-byte fused record remains compiled and runnable when the switch is off.
Native benchmark objects always carry optional scale tensors so the public
prequantized-X/checkpoint-facing contract remains unchanged.  The extension
cache key and benchmark metadata include the new switch.

TP4 is the first performance target.  TP8 remains on the existing runnable
fallback until the candidate passes TP4; before final selection, TP8 must be
revalidated at all required correctness entry points.

## Verification gates

1. Static/model-load checks: exact tensor sizes and byte mapping for W13/W2;
   descriptor box and stride alignment; no modification of the read-only main
   DeepGEMM checkout.
2. Local deterministic M8 and M128: finite outputs, accepted cosine/relative
   L2, and bitwise equality with the selected 80-byte native control.
3. TP4 graph smoke at M8 and M128: all ranks finite; embedded communication
   remains within the established NCCL envelope; exactly one business launch.
4. Independently cold-L2 same-process endpoint screen, balanced `2x10` or
   stronger.  Reject on correctness failure or a repeatable endpoint regression
   greater than 1%; retain only a measurable geometric-mean improvement.
5. If retained, run all five M values under the formal `10x200` rank-max
   protocol and compare against the selected 5/6-kernel control.  The project
   goal remains at least 1.10x five-M geometric-mean speedup, with every M
   reported.

## Follow-on if transport is insufficient

If the split layout is correct but contributes less than 3%, recapture M8 and
M128 NCU profiles before another source direction.  The next structural option
is a full CTA-role reorder that places both WGMMA groups at hardware-aligned
warps 0..7 and support roles afterward; simply deleting the inherited alignment
warp is forbidden by Iterations 467–468.

## Iteration-480 amendment: one contiguous tile transaction

The first implementation validated the 15% byte reduction and bitwise output,
but two TMA copies per K128 stage regressed the endpoint geometric mean by
0.26%.  Keep that result rejected.  The next bounded variant retains the same
17 KiB shared representation while removing the extra transaction.

At model load, lay out each complete N256/K128 tile contiguously as 16 KiB of
row-major packed weight followed by 1 KiB of four-row-grouped scales.  View the
17 KiB tile as a legal two-dimensional `128 B x 136` tensor-map box.  Both box
dimensions are at most 256, the inner dimension and global stride are 16-byte
aligned, and one TMA copy reproduces the same shared weight and scale planes.
The producer coordinate becomes a linear expert/N-tile/K-tile index multiplied
by 136; the RS consumer mapping is unchanged from the validated split layout.

Implement this as a separate default-off switch so Iteration 480 remains
reproducible.  Admission gates are the same: bitwise local M8/M128, TP4 graph
and communication correctness, then a matched endpoint cold-L2 screen against
the retained 80-byte control.  Do not combine it with scale-word caching or the
two-copy split flag.
