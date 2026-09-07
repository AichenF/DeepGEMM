# Iterations 729–731: dependency-ready W13 tail overlap rejection

## Goal and isolated design

Replace the production kernel's fixed W13-to-activation phase wait with a
generation-safe readiness edge while preserving its 128-thread single-WG
worker, static WGMMA task order, and 8/9 resident CTAs per H20 SM.

The final experiment is opt-in through
`V4_SINGLE_LAUNCH_DEP_READY_TAIL=1`.  At M128, the existing compact W13 phase
call executes every full grid wave, then every CTA contributes one release
RMW to a graph-stable packed generation word.  Residual W13 producers do not
wait.  Only suffix CTAs with no residual task acquire the publication and
execute at most one route-major activation N128 group whose inverse route map
proves that both gate/up tiles are in the complete prefix.  The existing W13
safety barrier remains before the remaining activation groups.  M<=64 compile
to the selected production phase flow.  The flag is disabled by default.

This deliberately does not add conventional loader/math warp specialization:
one 128-thread CTA is one homogeneous WGMMA warpgroup, and multiple resident
CTAs remain the independent persistent work providers.

## Fresh production anchor

Physical H20 GPUs 1–4, random routes with seed 20260902, TP4 CUDA Graph,
replay-level pairing, 4 warmups and 2 x 20 samples per implementation.  Every
replay has its own excluded 256 MiB L2 clear.

| M | same-source multi (ms) | production one-kernel (ms) | one / multi |
|---:|---:|---:|---:|
| 8 | 0.071248 | 0.075808 | 1.06400 |
| 16 | 0.114816 | 0.123584 | 1.07637 |
| 32 | 0.177744 | 0.198672 | 1.11774 |
| 64 | 0.249664 | 0.280480 | 1.12343 |
| 128 | 0.305600 | 0.349104 | 1.14236 |
| geometric mean | 0.161814 | 0.178704 | 1.10438 |

All correctness and all-reduce checks pass.  The first `iter729d` attempt is
a non-production legacy configuration because `V4_SINGLE_LAUNCH_TP4=1` was
omitted; it is retained only as a failure record and is not used above.

## Correctness, generation, and resources

- Local M8 and M128 candidate output is bitwise equal to the independently
  launched same-source multi-kernel output: cosine 1.0, rel-L2 0, finite.
- Seeding all four production barrier words at generation `2^22-1` wraps them
  to zero correctly.  W13 readiness owns a separate packed word, so it does
  not perturb the existing phase generations.
- TP4 end-to-end candidate/control correctness and embedded all-reduce pass
  at every M in the first screen.
- TP8 M128 executes through its one-node CUDA Graph on all eight H20s;
  all-reduce passes with minimum-rank cosine 0.999991959 and maximum rel-L2
  0.004010353.  TP8 uses its existing NVLS-pull kernel path; the new overlap
  is TP4-M128-only.
- Flag-off resources remain production-identical: M8/M16/M32 use 63
  registers/thread, M64 uses 64, and M128 uses 56; every entry has 32-byte
  stack, 2,048-byte static shared memory, and zero fixed local allocation.
- The candidate keeps 56 registers and zero fixed local allocation at M128,
  but expands its stack frame from 32 to 48 bytes.  M<=64 resources are
  production-identical in the repaired version.

## Performance falsification

The first implementation split the residual W13 tile into a second compact
device call and enabled the mechanism at every M.  It is correct but regresses
the production one-kernel anchor by approximately 9.3%, 5.8%, 3.8%, 2.7%, and
3.0% at M8/16/32/64/128; its all-M geometric mean is 4.9% slower.

The repaired version moves readiness and the residual producer task inside
the original compact phase call and limits overlap to M128.  M8 and M64 return
to production noise range (`0.075616` and `0.280400` ms).  M128 remains
negative:

| arm | paired multi (ms) | one-kernel (ms) | one / multi |
|---|---:|---:|---:|
| dependency-ready ON | 0.306560 | 0.356576 | 1.16315 |
| dependency-ready OFF | 0.304480 | 0.349664 | 1.14840 |

Direct ON/OFF is a 1.98% regression.  Normalizing each arm by its paired
multi-kernel control still gives a 1.29% regression, larger than the observed
run noise.

## Decision

Reject the readiness overlap as a production default and leave it explicitly
default-off.  The useful activation work hidden under one residual W13 wave
does not repay the extra packed arrival, inverse-route publication, divergent
control flow, and M128 stack growth.

Do not revive W2 chunk overlap in this round.  Earlier correct wait-only and
dedicated-helper experiments already found chunk enumeration overhead and
lost W2 producers, while returning a helper CTA to WGMMA caused corruption.
The newly required phase-readiness prerequisite is itself negative.  Per the
agreed stop condition, retain the fixed-phase production bundle and pause
performance optimization here.

## Raw artifacts

- `bench/results/iter729e_dep_ready_production_anchor_tp4_allm_cold_20260907.log`
- `bench/results/iter730a_dep_ready_tail_m8_jit_correctness_20260907.log`
- `bench/results/iter730b_dep_ready_tail_m8_generation_wrap_20260907.log`
- `bench/results/iter730c_dep_ready_tail_m128_correctness_20260907.log`
- `bench/results/iter730d_dep_ready_tail_tp4_allm_cold_screen_20260907.log`
- `bench/results/iter731a_dep_ready_compact_internal_m128_jit_correctness_20260907.log`
- `bench/results/iter731b_dep_ready_compact_internal_tp4_screen_20260907.log`
- `bench/results/iter731c_dep_ready_tp8_m128_graph_smoke_20260907.log`
- `bench/results/iter731d_dep_ready_flagoff_m128_jit_correctness_20260907.log`
- `bench/results/iter731e_dep_ready_flagoff_tp4_m128_cold_regression_20260907.log`
