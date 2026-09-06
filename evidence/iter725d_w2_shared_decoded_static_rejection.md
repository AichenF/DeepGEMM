# Iteration 725d: reject B32 shared-decoded W2 SS-WGMMA

## Fresh build

The repaired isolated candidate built as
`v4tp_6009ed5b408d872cf41c_v178mspec` with TP4, M128 bound eight,
schedule zero, compact-W13 production settings, and both M128 wave rotations
disabled.  The first JIT attempt was compile-only and failed because the
candidate's assertion also covered an unused eight-WG template
instantiation; commit `7888019` moved the flat-task requirements into the
compile-time branch and the same extension then built successfully.

The exact TP4 M128 split-K2 entry reports:

```text
REG64 STACK48 SHARED2048 LOCAL0
```

The candidate's dynamic shared memory is 22,528 bytes.  Together with the
2,048-byte static allocation this is 24,576 bytes/CTA; eight CTAs use 196,608
bytes, below the H20 SM shared-memory limit.  The 64-register launch bound is
therefore still the eight-CTA limiting resource.  No local-memory spill was
introduced.

## Exact SASS gate

The SS form is unambiguous in SASS because it has a second shared descriptor
GPR before the activation `gdesc`.  The isolated W2 interval contains:

```text
SS QGMMA instructions          32
WARPGROUP.DEPBAR instructions  32
```

Every SS QGMMA is still individually waited; the candidate did not approach
the declared `<=20` gate or standalone W2's 32/16 schedule.  The exact
287,430-byte interval has SHA256:

```text
b0348d5698e78f1a732e05ae314178aeb59730e7fc0fd62a569c04a722479923
```

## Decision

**Reject before correctness or CUDA business-kernel launch.**  Moving decoded
weights from RS operands into B32 shared memory removes the suspected source
register alias but does not change ptxas's fused-entry dependency schedule.
It also adds shared stores, async-proxy fences, and 16 CTA barriers per W2
task, so there is no static basis for a runtime win once the primary issue
depth objective fails.

Restore `v4_flash_tp_wgmma.py` byte-for-byte to the pre-probe production
source.  Per the user's stop condition, pause optimization after recording
this negative result.

## Artifacts

- `bench/results/iter725b_w2_shared_decoded_jit_20260907.log`
- `bench/results/iter725c_w2_shared_decoded_jit_20260907.log`
- `bench/results/iter725c_w2_shared_decoded_resources_20260907.log`
- `bench/results/iter725c_w2_shared_decoded_m128_split2_ss_interval.sass`
