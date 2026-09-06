# Iteration 692: M128 full-K one-launch W13 rejection

## Question

The multi-to-single audit showed that the persistent W13 callee adds a small
task-boundary instruction tail. Could M128 reduce that boundary cost by using
one full-K (`split_k=1`) N128 task instead of two split-K2 tasks, while still
keeping enough work for the 702 resident CTAs?

The candidate added only the missing TP4 M128 `SplitK=1` host dispatch. The
selected MXFP4/TMA/S2R/WGMMA body, 702-CTA grid, W13 wave rotation, activation,
W2, reduction, communication code, and public FP8-input contract were
unchanged.

## Gates

- Candidate SHA-256:
  `96115e35c4f8effe56cb2117fe199bb4d1a2212b0714a596e9e7b2e4b8d36fa6`.
- The first invocation before adding dispatch was a host validation failure
  (`single-launch TP4 split-K must be 2 or 4`) and launched no candidate
  kernel.
- Five valid physical-GPU0 runs used random M128 routes/seed 20260902 and a
  separate excluded 256 MiB L2 clear before each measured kernel.
- All five outputs were finite and exactly equal to the independent
  same-source full-K multi local reference (`cosine` effectively 1,
  `rel_l2=0`).
- The full-K M128 entry remains `REG56 SHARED2048 LOCAL0`, but its stack frame
  rises from production 32 bytes to 48 bytes.

## Result

Full-K W13 min/median/max is `224.384/225.664/226.880 us`, and its complete
route+W13+requant+W2 phase sum is `342.784/343.360/344.000 us`.

The adjacent exact-production split-K2 anchors from Iteration 691 are
`208.768/208.992/210.336 us` for W13 and
`325.472/325.952/327.488 us` for the four-phase sum. Full-K therefore
regresses median W13 by 7.98% and the local phase sum by 5.34%.

## Decision

Reject before distributed TP4 timing and restore exact production SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
Halving persistent task transitions does not offset the doubled K dependency
chain, reduced scheduling slack, and larger stack contract. Retain M128
split-K2.

Raw artifacts:

- `bench/results/iter692_fullk_single_m128_phase_repeat.log`
- `bench/results/iter692b_fullk_single_m128_jit_correctness.log`
- `bench/results/iter692c_fullk_single_m128_resources.log`
- `bench/results/iter692d_fullk_single_m128_phase_repeat.log`
- `bench/results/iter691f_production_m128_phase_repeat.log` (control)
