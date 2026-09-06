# Iteration 723c — reject compact W2 bound-eight cross-composition

## Question

Can the existing compact per-task W2 device-call boundary recover standalone
W2's two-QGMMA issue depth when the fused M128 entry is raised from the
selected 56-register/nine-CTA contract to the standalone-like
64-register/eight-CTA contract?

This is the one missing cell between the prior observations:

- compact call + bound-nine: 32 QGMMAs / 32 dependency waits;
- inline fused W2 + bound-eight: 32 / 32;
- standalone W2 + bound-eight: 32 / 16.

## Exact configuration

Fresh extension `v4tp_2edb20864a9689f08b51_v178mspec` was built with:

```text
CUDA_VISIBLE_DEVICES=1
TORCH_CUDA_ARCH_LIST=9.0a
V4_SINGLE_LAUNCH_TP4=1
V4_SINGLE_LAUNCH_M128_BOUND9=0
V4_SINGLE_LAUNCH_MIN_BLOCKS=8
V4_SINGLE_LAUNCH_CTAS_PER_SM=8
V4_SINGLE_LAUNCH_W13_WAVE_ROTATE=0
V4_SINGLE_LAUNCH_W2_WAVE_ROTATE=0
V4_SINGLE_LAUNCH_W2_COMPACT_TASK_CALL=1
```

The TP4 M128 split-K2 entry reports:

```text
REG64 STACK48 SHARED2048 LOCAL0
```

## Exact SASS gate

The compact W2 split-K2 callee was extracted from the fresh cubin, isolated,
and counted directly:

```text
QGMMA instructions             32
WARPGROUP.DEPBAR instructions  32
```

The 89,531-byte exact interval has SHA256:

```text
56a22b8691d896235397a44522d864f2cb7af7c435e4e05e0118403dd73300a4
```

The predeclared admission gate required all 32 QGMMAs with at most 20 waits
(target 16), plus no resource cliff.  It fails decisively: every QGMMA is
still individually waited, and the entry also grows to STACK48.

## Decision

**Reject before CUDA business-kernel execution.**  No correctness or runtime
performance is claimed for this candidate.  Restore the production-only
bound-nine validator and remove the temporary compare-harness whitelist.

This closes the cross-composition rather than merely repeating one prior
probe: neither the compact call boundary nor the 64-register entry contract,
alone or together, gives the fused compiler frame standalone W2's 32/16
schedule.  The next candidate must change phase/dataflow ownership or the
code-object boundary, not another same-frame source spelling.

## Artifacts

- `evidence/iter723a_w2_compact_bound8_composition.md`
- `bench/results/iter723b_w2_compact_bound8_jit_20260906.log`
- `bench/results/iter723c_w2_compact_bound8_resources_20260906.log`
- `bench/results/iter723c_w2_compact_bound8_m128_split2_exact.sass`
