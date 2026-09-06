# Iteration 711g — unbounded whole-W2 outline remains serialized

## Scope

Compile-only H20 GPU1 audit combining
`V4_SINGLE_LAUNCH_M128_UNBOUNDED=1` with
`V4_SINGLE_LAUNCH_W2_PHASE_NOINLINE=1`.  The selected compact-W13 schedule
was retained, bound9 and both 702-grid rotations were disabled, and no
business kernel was launched.

Extension: `v4tp_49f35bd1b995105c53bc_v178mspec`.

## Result

The TP4 M128/split-K2 main entry is
`REG85 STACK32 SHARED2048 LOCAL0`; the whole-W2 local callee is 20,544 bytes.
Its exact SASS contains 32 QGMMAs and 32 `WARPGROUP.DEPBAR` instructions.
Exact callee SASS SHA256:
`fcf7f5365971b40dda638d9daeac09af6a7d686dc48bdeb15b368a41dac3587f`.

## Decision and next direction

**Reject before correctness/timing.**  Combining natural register headroom
and phase-local compilation still does not reproduce standalone's 32/16
schedule.  The emitted fused sequence shows accumulator/source register
overlap around the second N64 QGMMA, while standalone bound8 assigns disjoint
source and destination groups.  Revisit the previously static-rejected
predecode/pair/spill/operand-fence chain only in the natural-register entry:
its earlier 56-register tests could not satisfy the intended disjointness.

