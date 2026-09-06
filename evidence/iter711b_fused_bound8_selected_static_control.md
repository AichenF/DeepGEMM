# Iteration 711b — selected fused M128 bound-8 static control

## Question

Is the selected M128 fused W2 serialization repaired by relaxing only its
nine-CTA/56-register constraint while retaining the compact-W13 outline,
dynamic route shared memory, release barrier and other production paths?

## Method

Compile unchanged current source on H20 GPU1 with the production TP4 bundle,
but set `V4_SINGLE_LAUNCH_M128_BOUND9=0`.  The two M128 wave rotations are
also set to zero because their mappings are defined only for the 702-CTA
bound-9 grid.  No business kernel was launched.

Extension: `v4tp_facd0af033b98b98562f_v178mspec`.

## Result

The TP4 M128/split-K2 main entry becomes:

```text
REG64 STACK48 SHARED2048 LOCAL0
```

The exact main-entry SASS contains only the inline W2 QGMMAs and has:

```text
QGMMA: 32
WARPGROUP.DEPBAR: 32
```

Exact SASS SHA256:
`f8264259210532eada47fb253d69192dbacff13a3c0621255d61d1f288802c38`.

## Decision

**Reject before correctness and timing.**  Relaxing the fused entry from 56
to 64 registers while preserving the selected compact outline is not
sufficient to recover standalone W2's paired issue.  Combined with Iteration
711a, register pressure is causal but not the only constraint.

The next static control applies explicit min-blocks 8 to standalone W2.  If
that also serializes despite a nominally wider register allocation, then the
compiler's launch-bound contract—not merely final register count—is the
remaining difference and an occupancy-derived, unbounded fused launch is the
next credible structural candidate.

