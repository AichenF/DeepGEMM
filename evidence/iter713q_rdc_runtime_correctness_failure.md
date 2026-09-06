# Iteration 713q: RDC runtime reaches but fails the shared correctness gate

Date: 2026-09-06

The TP4 M128 smoke now passes all reconstructed Python/JIT/CAR initialization
and reaches the benchmark's shared correctness check.  It exits before timing.

Observed rank-aggregated result:

```text
candidate: finite=false, cosine=NaN, rel_l2=NaN, allreduce_ok=false
control:   finite=true, cosine=0.9210312561, rel_l2=0.3975496704,
           max_abs=120832, allreduce_ok=false
```

Because the same prebuilt module's frozen multi-kernel control also fails, this
run cannot isolate or score the RDC W2 candidate.  The failure instead shows
that the reconstructed SGLang/route-helper runtime does not yet reproduce the
previously accepted benchmark environment.  No latency number is reported and
the candidate is rejected from performance comparison until the control passes
unchanged.

Run contract: TP4 on devices 0--3, M128 random precomputed routes, CUDA Graph
benchmark entry, separate 256 MiB cold-L2 clear configured before each replay,
two outer repetitions, ten replays per implementation, and two warmups.  The
timing loop was not reached.

Raw log:
`bench/results/iter713q_rdc_tp4_m128_cold_smoke_20260906.log`.
