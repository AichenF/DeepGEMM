# Iteration 714d: uniquely named RDC runtime rejection

Date: 2026-09-06

The uniquely named RDC host/CUDA entry is selectively dispatched while the
ordinary same-ABI library owns every helper and the multi-kernel control.
This removes ELF/CUDA symbol-preemption ambiguity.  TP4 M128 random-route
correctness passes for both paths with cosine `0.9999955977`, relative L2
`0.0029672640`, max absolute error 1024, finite outputs and
`allreduce_ok=true`.

Short cold-L2 result over 20 replay-paired samples/path:

```text
ordinary multi:      0.303504005 ms
unique RDC single:   0.363023996 ms
single / multi:      1.196109410  (single slower 19.611%)
```

The matched ordinary-unbounded run in Iteration 713y measured 0.362560004 ms
single and 0.303455994 ms multi.  Relative to that exact configuration, unique
RDC is 0.000463992 ms or 0.1280% slower, while the multi anchors differ by only
0.0158%.  Therefore the recovered 32-QGMMA/16-DEPBAR W2 schedule yields no
end-to-end gain and slightly regresses this screen.

Decision: close the RDC/device-call compiler-isolation path.  Do not run a
long benchmark or integrate it into production.  The likely costs are the
measured 195-register entry, 112-byte/thread stack, device-call overhead and
reduced residency.  Production source remains unchanged; the formal all-M gap
continues to be the prior 10x200 result, not this rejection screen.

Raw log:
`bench/results/iter714d_unique_rdc_tp4_m128_cold_smoke_20260906.log`.
