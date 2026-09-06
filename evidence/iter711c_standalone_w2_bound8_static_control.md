# Iteration 711c — standalone W2 bound-8 retains paired issue

## Question

Does any explicit minimum-block launch bound serialize standalone W2, or is
the cliff specifically caused by the 56-register/nine-CTA constraint?

## Method

Compile unchanged current source on H20 GPU1 with
`V4_MIN_BLOCKS_PER_SM=8`, which applies `__launch_bounds__(128,8)` to the
standalone route GEMMs.  No business kernel was launched.

Extension: `v4tp_30df862a87d9edcc2371_v178mspec`.

## Result

Standalone `route_gemm<512,4096,1,false>` uses:

```text
REG64 STACK0 SHARED2048 LOCAL0
```

Its exact SASS retains:

```text
QGMMA: 32
WARPGROUP.DEPBAR: 16
```

Exact SASS SHA256:
`991702fea4bfe729295bee7b01c3d008e0259a721539dcdb97ae4e6aacdd5ebc`.

## Conclusion

An explicit launch bound does not itself cause serialization.  The cliff is
between the 64-register/eight-CTA and 56-register/nine-CTA contracts for the
standalone body.  However, Iteration 711b proves that 64 registers alone are
not sufficient in the fused entry, whose cross-phase state/control still
prevents paired issue.

The next candidate should compile fused M128 without a minimum-residency
constraint, inspect its natural resource requirement and exact W2 schedule,
then—only if it restores 32/16—derive a fully co-resident grid from actual
occupancy for correctness and cold-L2 testing.

