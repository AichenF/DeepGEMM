# Iteration 660: M128 W2 complete-wave ownership rotation rejected

## Hypothesis and mapping

Iteration 659 found a repeatable benefit from rotating physical CTA ownership
between complete M128 W13 waves.  This experiment applied the same isolated
mapping to W2: for complete W2 grid wave `q`, physical CTA slot `cta` owned
logical slot `(cta + 13*q) mod 702`.  The residual wave, task count, arithmetic,
barriers, W13 and communication were unchanged.  The option was default-off
and compiled into an independent extension.

## Static and correctness gates

- Fresh SM90a extension: `v4tp_b304b945f4e18c33a206_v178mspec`.
- M128 split-K2/4 resources remained
  `REG56 STACK32 SHARED2048 LOCAL0`, retaining nine CTAs/SM.
- Random-route M128, seed 20260902, was bitwise equal to the independent
  same-source multi-kernel local output (`cosine=1`, `rel_l2=0`, finite), with
  1,992 padded rows.
- Packed barrier generations seeded at `2^22-1` returned exactly `[0,0,0,0]`.

## Local cold-L2 screen

Physical H20 GPU1, eight independent process samples ordered
OFF/ON/ON/OFF/OFF/ON/ON/OFF.  Every launch used a separate excluded 256 MiB
L2 clear.  Public X was prequantized FP8-E4M3 plus FP32 group-128 scales;
weights were MXFP4; TP communication was disabled only for phase isolation.

W2 microseconds:

- OFF: `104.960, 106.208, 106.560, 106.240`
- ON: `107.136, 105.504, 105.696, 105.888`

The W2 median moves from `106.224` to `105.792 us`, only 0.41%, below the
predeclared 1% gate.  More importantly, the mean moves from `105.992` to
`106.056 us`, so the direction reverses (ON is 0.06% slower).  The four-phase
sum appears 0.66% better by median and 0.45% by mean, but that aggregate is
contaminated by ordinary W13 process variation; the isolated target phase has
no robust gain.

## Decision

Reject before TP4.  Remove the W2 option, extension-identity field and
benchmark metadata.  Production source returns byte-identical to Iteration
659, preserving only the independently proven default-off W13 rotation.  W2's
n-tile/task locality does not support transferring this multi-kernel-inspired
ownership trick.

Raw artifacts:

- `bench/results/iter660_w2_wave_rotate13_{jit,resources}_20260906.log`
- `bench/results/iter660_w2_wave_rotate13_m128_correctness_20260906.log`
- `bench/results/iter660_w2_wave_rotate*_m128_phase*_20260906.log`
