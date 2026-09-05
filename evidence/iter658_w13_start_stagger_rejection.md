# Iteration 658: reject M128 W13 start staggering

## Hypothesis

The standalone multi-kernel W13 launch naturally ramps fresh CTAs onto each
SM, while the fused persistent grid releases all 702 CTAs from its route
barrier together.  If the resulting synchronized TMA/WGMMA bursts caused the
measured scheduler/throughput deficit, a small one-time phase offset across
the nine resident CTAs per H20 SM could retain the offset through the complete
grid-stride W13 loop.

The default-off prototype changed only the M128 compact-W13 callee.  Each CTA
executed `__nanosleep((blockIdx.x / 78) * 64)` once before its first task.
Task identity, task order, weights, arithmetic, barriers, W2 and communication
were unchanged.  Block-index groups of 78 correspond to the nine complete
residency waves measured by Iteration 653b.

## Gates

- Fresh SM90a JIT passed.  M128 split-K2/4 remained
  `REG56 STACK32 SHARED2048 LOCAL0`, preserving exactly nine CTAs/SM.
- Random-route M128, seed 20260902, matched the independent same-source
  multi-kernel local output bitwise (`cosine=1`, `rel_l2=0`, finite), with
  1,992 padded rows.
- Packed generations seeded at `2^22-1` returned exactly `[0,0,0,0]`.

## Cold-L2 phase result

Physical H20 GPU1, eight independent processes in
OFF/ON/ON/OFF/OFF/ON/ON/OFF order.  Every measured launch was preceded by a
separate excluded 256 MiB clear.  Public input was already FP8-E4M3 with
FP32 group-128 scales; weights were MXFP4.  TP communication was disabled
only for this local phase isolation.

W13 microseconds:

- OFF: `214.048, 216.992, 213.856, 214.624`
- ON: `215.552, 214.496, 214.272, 214.176`

OFF/ON medians are `214.336/214.384 us`: ON is 0.02% slower.  Means are
`214.880/214.624 us`: ON is 0.12% faster.  The median and mean disagree and
both deltas are noise-scale.

The corresponding sums of route, W13, requant and W2 phases have exactly the
same OFF/ON median, `332.048/332.048 us`.  Mean totals are
`332.592/332.208 us`, again only a contradictory 0.12% nominal difference.

## Decision

Reject before TP4 timing and do not sweep 16/32/128/256 ns.  The chosen 64 ns
already spans 0--512 ns across the resident slots; it creates no repeatable
W13 or complete-compute improvement.  The synchronized phase release is not
the missing standalone-W13 throughput mechanism.  Remove the experiment and
restore production source byte-identically.

Raw artifacts:

- `bench/results/iter658_w13_start_stagger64_resources_20260906.log`
- `bench/results/iter658_w13_start_stagger64_m128_correctness_20260906.log`
- `bench/results/iter658_w13_stagger_{off,64}_m128_phase_{1,2}_20260906.log`
- `bench/results/iter658_w13_stagger64_m128_phase_window2_20260906.log`
