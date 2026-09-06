# Iteration 713y: matched ordinary unbounded runtime anchor

Date: 2026-09-06

The same validated unbounded/whole-W2-phase configuration was run entirely
from the ordinary non-RDC miniforge library on the same TP4 M128 random-route
cold-L2 harness.  Both single and multi paths pass with cosine
`0.9999955977`, relative L2 `0.0029672640`, max absolute error 1024, finite
outputs and `allreduce_ok=true`.

Short result over 20 replay-paired cold samples/path:

```text
ordinary multi:             0.303455994 ms
ordinary unbounded single:  0.362560004 ms
single / multi:             1.194769626
```

The selective RDC run from Iteration 713x measured 0.362591997 ms single and
0.303535998 ms multi.  The single difference is only 0.000031993 ms
(ordinary is 0.0088% faster), far below a meaningful gain and within short-run
noise.

This is strong evidence that the recovered RDC W2 static schedule does not
improve the end-to-end kernel.  One attribution ambiguity remains: both
libraries export the same C++ host and CUDA entry symbols, so ELF symbol
preemption could make a method object from the second library resolve a
definition from the first.  Before closing RDC conclusively, rebuild the
temporary candidate with uniquely renamed host and CUDA entry symbols and
dispatch that distinct binding.

Raw log:
`bench/results/iter713y_nonrdc_unbounded_tp4_m128_cold_smoke_20260906.log`.
