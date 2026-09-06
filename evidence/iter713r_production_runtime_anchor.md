# Iteration 713r: ordinary production runtime anchor

Date: 2026-09-06

To attribute Iteration 713q, the unchanged production source was run through
its ordinary non-RDC JIT on the same physical GPUs 0--3, same SGLang/CAR
runtime, same TP4 M128 random-route case, and same cold-L2 graph harness.

Both paths pass the independent correctness and all-reduce gates:

```text
candidate: cosine=0.9999955977, rel_l2=0.0029672640,
           max_abs=1024, finite=true, allreduce_ok=true
control:   cosine=0.9999955977, rel_l2=0.0029672640,
           max_abs=1024, finite=true, allreduce_ok=true
```

Short-run medians over 20 cold samples per path are 0.304096 ms for the
multi-kernel control and 0.351680 ms for the default single kernel.  These are
diagnostic anchors, not formal 10x200 results and not the selected optimized
flag set.

This same-environment pass isolates Iteration 713q's broken control and NaN
candidate to the whole-extension RDC composition rather than GPU topology,
route generation, CARv2, or the benchmark's reference path.  Whole-extension
RDC is rejected as a runtime implementation.  Any further device-link probe
must isolate the candidate translation unit and retain a non-RDC control.

Raw log:
`bench/results/iter713r_production_tp4_m128_cold_anchor_20260906.log`.
