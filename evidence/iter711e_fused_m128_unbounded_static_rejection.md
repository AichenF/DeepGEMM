# Iteration 711e — unbounded fused M128 remains serialized

## Scope

Compile-only H20 GPU1 audit of Iteration 711d.  The TP4 production compact
bundle was retained, M128 bound9 and both 702-grid rotations were disabled,
and `V4_SINGLE_LAUNCH_M128_UNBOUNDED=1` selected the effectively unconstrained
M128 entry.  No business kernel was launched.

Extension: `v4tp_c168d18c5c6bcac3eb17_v178mspec`.

## Resource result

TP4 M128/split-K2 compiles to:

```text
REG79 STACK32 SHARED2048 LOCAL0
```

The register footprint statically limits H20 residency to at most six
128-thread CTAs/SM, still enough for a complete 468-CTA co-resident grid if a
later runtime test were admitted.

## SASS result

The exact main entry contains 32 inline W2 QGMMAs and 32 dependency barriers,
not the required 32/16 pairing.  Exact SASS SHA256:
`2c9ac18ff447b9a19f6acecb081fc9c2694090cc9bbee40d67a42e46a176b1a1`.

## Decision

**Reject before correctness and timing.**  Removing the launch-bound register
ceiling is not sufficient while W2 remains embedded in the large fused entry.
Together with Iterations 710c and 711a-c, the remaining bounded combination is
an unbounded M128 entry plus a whole-W2 phase call: earlier outline tests were
still constrained to 56 registers, while this natural-register test remained
inline.  Require 32/16 statically before any launch.

