# Iteration 711i — natural-register W2 operand isolation remains serialized

## Scope

Compile-only H20 GPU1 audit of the full predecode + paired issue + one-value
shared spill + compiler operand-fence chain under the natural-register TP4
M128 entry.  The selected compact-W13 schedule was retained, bound9 and both
702-grid rotations were disabled, and no business kernel was launched.

Extension: `v4tp_3c4b796095f7e5cdca34_v178mspec`.

## Result

The TP4 M128/split-K2 main entry is
`REG80 STACK32 SHARED2048 LOCAL0`.  Symbol 205 is the 72,192-byte global main
entry; its exact `nvdisasm -fun 205 -sf` slice contains 32 W2 QGMMAs and 32
`WARPGROUP.DEPBAR` instructions.  Exact SASS SHA256:
`7b021bb546c86fc4bf21138595eb0f86be9fc75978a85587c76909fedab6a211`.

Source SHA256:
`30e6c402b4dc220d2a4eb03099d1f1cd0f82e59b0af38747b59906d84c3fdbdf`.

## Decision and next direction

**Reject before correctness/timing.**  Natural register headroom plus an
explicit shared spill and operand fences still does not reproduce the
standalone W2 schedule of 32 QGMMAs / 16 waits.  This closes the tested
source-lifetime/call-isolation family: its variants fail both under the
56-register residency constraint and under natural allocation.  Stop adding
more call-ABI or register-lifetime probes.  Re-audit the faster multi-kernel
path for reusable coarse W13/W2 task ownership, stage layout and overlap
structure that can be expressed inside the existing one-launch state
machine.

