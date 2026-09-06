# Iteration 711a — standalone W2 bound-9 static control

## Question

Does standalone W2 retain its paired 32-QGMMA/16-wait schedule when ptxas is
given the same nine-CTA-per-SM register constraint as the selected fused M128
entry?

## Method

Compile the unchanged current source on H20 GPU1 with
`V4_W13_LAUNCH_BOUND_10=1`.  The existing launch-bounds macro applies
min-blocks 10 to standalone W13 and min-blocks 9 to standalone W2.  Importing
the extension performs JIT only; no MoE business kernel was launched.

Extension: `v4tp_660a3b674a50eda03d64_v178mspec`.

## Result

The exact standalone `route_gemm<512,4096,1,false>` specialization changes
from its natural 61-register contract to:

```text
REG56 STACK0 SHARED2048 LOCAL0
```

Its exact function SASS contains:

```text
QGMMA: 32
WARPGROUP.DEPBAR: 32
```

Exact SASS SHA256:
`417857b8a26f64a69b04727f5a6a746a7ebaafd92c9dcfad9f5f2d5edf304721`.

The unconstrained standalone control previously measured `REG61` and
`32 QGMMA / 16 DEPBAR`.

## Conclusion

The 56-register/nine-CTA contract is sufficient by itself to make even the
standalone W2 body serialize both N64 QGMMAs.  This establishes register
pressure as a causal prerequisite for the fused 1:1 schedule.  It does not
contradict the prior 64-register fused control: relaxing registers is
necessary but may not be sufficient in the large persistent entry.

The next isolation should retain every selected fused optimization while
disabling only M128 bound-9.  A candidate may advance only if the M128 fused
entry reaches at least the standalone-like register tier and recovers a
32/16 W2 interval.

