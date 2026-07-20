# SM90 NVFP4 BN128 Mode2 Braided Design

## Goal

Use one Mode2 Braided BK128 weight ABI for SM90 NVFP4 MegaMoE with both BN128
and BN256. BN128 remains the large-M split schedule, while BN256 remains the
small-M fused schedule. The schedule boundary must no longer imply a different
FP4 sign layout or a separate dequantization algorithm.

The migration intentionally invalidates cached BN128 weights packed with the
old standard-sign layout. Framework integrations must repack those weights.
The runtime will not carry a layout version or retain a legacy decoder.

## Architecture

The Python prepack path braids every accepted SM90 NVFP4 weight after producing
the existing 80-byte BK128 row: 64 bytes of packed FP4 values, 8 bytes of
scales, and 8 bytes of padding. `block_n` continues to determine only the N
tile shape and runtime schedule.

The existing Mode2 helper becomes the single owner of sign extraction and FP8
materialization. It exposes operations for both representations used by the
kernels:

- a packed-row source used by scratch-buffer dequantization;
- an in-place staged source used by the split L1/L2 pipeline.

Both operations reuse the same word-level Mode2 decoder. The split L1 and L2
bodies call these shared operations instead of the standard-sign LUT helpers.
Communication, barriers, scheduler state, launch geometry, and selector policy
remain unchanged.

## Data Flow

1. The framework chooses BN128 or BN256 from the existing load policy.
2. Prepack tiles scales and fuses each BK128 row, then applies Mode2 braiding.
3. Runtime dispatch still infers BN128 or BN256 from the scale tensor shape.
4. BN256 launches the fused body; BN128 launches split L1 followed by split L2.
5. Both paths decode braided signs through the common Mode2 word primitive.

There is no runtime layout flag, environment-variable switch, fallback, or
automatic conversion of old weights.

## Validation

Host tests must verify that BN128 and BN256 prepack both round-trip through the
exact inverse braid and produce the expected NVFP4 values and scales. CUDA
correctness must cover Flash, Pro, and MiMo, absent and per-expert global
scales, skewed routing, and representative large-M BN128 points.

AKO compares the new BN128 implementation against a frozen standard-sign
baseline with fixed seeds and cold-L2 flushing. H20 and H200 runs use the same
source and report per-shape latency. A stable regression above 3% is rejected;
large-M trial counts may be lower than the 50 samples required for small M.

## Failure Handling

The existing shape and scale-layout validation remains authoritative. No new
defensive runtime branch is added. Supplying an old standard-sign BN128 cache
after this migration is an ABI violation and must be prevented by framework
cache invalidation and repacking.
