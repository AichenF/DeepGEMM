# Iteration 668: reject 32-CTA expert wavefront

## Question

Can the multi-kernel path's one-tile-per-fresh-CTA behavior be approximated
inside the existing schedule-2 expert wavefront by widening each cohort from
16 to 32 CTAs?

For TP4 M128, W13 has 16 N128/split-K2 tasks per padded expert block while
activation/requant and W2 each have 32 tasks.  A 32-CTA cohort therefore
leaves half of its CTAs idle during W13, but changes W2 from two serial tasks
per CTA to one.  The experiment keeps one business kernel per rank.

## Implementation and gates

- Temporarily admitted `V4_SINGLE_LAUNCH_GROUP_CTAS=32`.
- Rounded the resident grid to complete 32-CTA groups.
- Generalized the W13 cohort loop so the upper 16 CTAs legally skip W13 for
  split-K2.
- The first compile attempt correctly failed at the old divisibility
  assertion; it executed no CUDA work.
- The repaired compute-only M128 run passed bitwise against the independent
  same-source multi-kernel `down` tensor (`cosine=1`, `rel_l2=0`, finite) and
  completed the expected phase words `[2048,0,0,2048]`.

## Cold-L2 TP4 result

Physical GPUs were 0/5/6/7.  The comparison used random routes with seed
20260902, replay-granularity paired CUDA Graphs, two outer batches by 20
samples, four warmups, and a distinct excluded 256 MiB L2 clear immediately
before every implementation replay.

- Multi control median: **0.305296 ms**.
- 32-CTA wavefront median: **0.531744 ms**.
- Candidate/control: **1.741733**; control/candidate: **0.574141x**.
- All-rank output and all-reduce correctness passed.

The schedule-2 candidate used its existing NVLS-pull tail while the control
used stock CARv2.  This prevents attributing a tiny difference solely to the
cohort, but the 226.448-us / 74.17% loss is far beyond any collective-tail
ambiguity.  Halving active W13 CTAs is decisively worse than removing one W2
task per CTA.

## Decision

Reject group 32, do not expand to other M values, and restore production
source byte-for-byte.  The transferable multi-kernel property is not simply
"one W2 tile per CTA"; preserving full W13 cold-weight concurrency is more
important.

Raw evidence:

- `bench/results/iter668_schedule2_group32_m128_correctness_20260906.log`
  (expected compile failure)
- `bench/results/iter668b_schedule2_group32_m128_correctness_20260906.log`
- `bench/results/iter668c_schedule2_group32_tp4_m128_cold_screen_20260906.log`
