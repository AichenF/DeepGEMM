# Iteration 651a: standalone W13 NCU filter matched no kernel

## Intent

Collect the same NCU sections as Iteration 650b for the current standalone
TP4 M128 W13 `route_gemm<4096,1024,split2>` on physical H20 GPU 1.  The
application retained its excluded 256 MiB cold-L2 clear before the local
pipeline.

## Result

The application ran, but NCU's demangled kernel-name view exposed only the
short name `route_gemm`; the requested regex `route_gemm.*4096` matched
nothing.  NCU reported:

```text
==WARNING== No kernels were profiled.
Available Kernels:
1. moe_fused_mul_sum_kernel
2. reduce_swiglu_quant_kernel
3. route_align_kernel
4. route_gemm
5. vectorized_elementwise_kernel
```

No W13 replay passes or performance metrics were collected.  The ordinary
application pipeline did execute, but this invocation emitted no tensor
correctness metric and therefore supplies neither performance nor correctness
evidence for the intended comparison.

## Decision

Retry with exact short-name filter `regex:^route_gemm$` and
`--launch-count 1`.  Inside the profiler range, route alignment precedes W13
but does not match; the first `route_gemm` is the selected split-K2 W13, so the
launch-count gate isolates it from the later W2 `route_gemm`.
