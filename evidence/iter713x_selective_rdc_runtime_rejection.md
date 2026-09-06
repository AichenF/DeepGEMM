# Iteration 713x: selective RDC single entry is correct but slower

Date: 2026-09-06

The method-level composition successfully loads both same-ABI libraries and
dispatches only `run_tp4_megamoe_single_launch` to the RDC artifact.  All
route/multi/helper methods remain in the ordinary non-RDC artifact.  Rank zero
prints both exact paths and the selected symbol before distributed setup.

TP4 M128 random-route correctness passes independently for both paths:

```text
candidate RDC single: cosine=0.9999955977, rel_l2=0.0029672640,
                      max_abs=1024, finite=true, allreduce_ok=true
ordinary multi:       cosine=0.9999955977, rel_l2=0.0029672640,
                      max_abs=1024, finite=true, allreduce_ok=true
```

Short cold-L2 result over two outer batches and ten replays per implementation
(20 samples/path, replay-granularity pairing):

```text
ordinary multi median: 0.303535998 ms
RDC single median:      0.362591997 ms
single / multi:         1.194560116  (single slower 19.456%)
```

This validates that Iteration 713q's NaNs came from executing the RDC-built
multi/helper path first, not from the selectively isolated single entry.  It
also rejects the candidate on performance: the recovered 32-QGMMA/16-wait W2
schedule does not repay the 195-register entry, 112-byte/thread stack frame,
device-call and reduced-residency costs.  A matched ordinary-unbounded run is
the final attribution control; no longer run is justified if it confirms RDC
is slower.

Raw log:
`bench/results/iter713x_selective_rdc_tp4_m128_cold_smoke_20260906.log`.
