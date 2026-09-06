# Iteration 669: alternating W13 mbarrier banks is neutral/slower

## Question

Does the standalone W13 launch benefit because each fresh CTA receives fresh
TMA completion objects, while a resident single-launch CTA reinitializes and
immediately reuses the same shared-memory mbarriers for consecutive tasks?

## Implementation and gates

The temporary M128-only candidate alternated consecutive compact-W13 tasks
between two independent two-stage completion-barrier banks.  Weight and
activation buffers, task order, wave rotations, arithmetic, output, phase
barriers, and launch count were unchanged.  Each selected bank was freshly
initialized before use.

- Compute-only M128 passed bitwise against same-source multi (`cosine=1`,
  `rel_l2=0`, finite) in the initial gate.
- The complete TP4 M128 entry stayed at **56 registers/thread, 32-byte stack,
  2,048-byte static shared memory, and zero local bytes**.  Nine-CTA/SM
  admission was preserved.
- Every measured launch received its own separate excluded 256 MiB L2 clear.

## M128 cold-L2 ABBA phase result

Eight independent GPU1 processes used order `0/1/1/0/0/1/1/0`, random routes
and seed 20260902.  W13 times in microseconds were:

- Control: `210.048, 210.592, 211.392, 209.536`; median **210.320**, mean
  **210.392**.
- Alternating banks: `210.176, 210.784, 211.200, 211.456`; median **210.992**,
  mean **210.904**.

The candidate regressed by **0.32% by median** and **0.24% by mean**, below
the 1% phase gate and in the wrong direction.  One accepted control replay
showed a small nondeterministic route-output difference
(`rel_l2=3.45e-4`); all other replays were bitwise and all were finite.  This
does not create a performance win for the candidate.

## Decision

Reject before distributed timing and remove the diagnostic.  Reusing the
same mbarrier addresses is not the measured standalone advantage.  Restore
production source byte-for-byte.

Raw evidence:

- `bench/results/iter669_w13_alt_barrier_banks_m128_correctness_20260906.log`
- `bench/results/iter669b_w13_alt_barrier_banks_m128_phase_abba_20260906.log`
