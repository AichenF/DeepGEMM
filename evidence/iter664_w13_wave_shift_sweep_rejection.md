# Iteration 664: alternative M128 W13 wave shifts rejected

## Candidate selection

The recovered Iteration-653 trace gives the exact physical SM for all 702
resident M128 CTAs.  For every shift from 1 through 175, the analysis mapped
each logical wave slot back to its physical owner across the five complete
W13 waves.  Shift 82 maximizes average cyclic SM-ID separation between
successive owners (`24.47` SM IDs); shift 5 combines high average separation
(`22.99`) with a larger minimum separation (`5`).  Both give five distinct
SM owners for every one of 702 logical slots.  The selected shift 13 gives
five distinct owners too, with average separation `22.03`.

These two candidates cover the strongest trace-derived alternatives without
turning an already small optimization into an exhaustive noisy sweep.  Each
preserves the exact global task set, arithmetic, residual wave and one-
subtraction modulo path.

## Local cold-L2 screen

Physical H20 GPU1, M128 random routes seed 20260902, split-K2, phase stamps,
communication disabled only for isolation, and a separate excluded 256 MiB
L2 clear before every launch.  Six independent processes were ordered
13/5/82/82/5/13.  Every complete routed W2 output remained bitwise equal to
the independent same-source multi local output (`cosine=1`, `rel_l2=0`,
finite), with 1,992 padded rows.

W13 microseconds:

- shift 13: `210.400, 210.336`, mean/median `210.368`
- shift 5: `211.136, 210.848`, mean/median `210.992` (0.30% slower)
- shift 82: `210.240, 212.320`, mean/median `211.280` (0.43% slower)

Neither candidate approaches the predeclared 1% improvement requirement;
shift 82 also reverses direction across its two samples.  Additional samples
cannot turn these observed means into a material gain.

## Decision

Reject without TP4 distributed timing and restore the public value set to
`{0,13}`.  The selected shift 13 remains the only reproduced winner.  Source
returns byte-identical to Iteration 663.

Raw artifact:

- `bench/results/iter664_w13_wave_shift5_82_m128_phase_screen_20260906.log`
