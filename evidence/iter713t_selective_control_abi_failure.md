# Iteration 713t: selective dispatch control library has the wrong ABI

Date: 2026-09-06

The first selective-dispatch TP4 launch exits while loading the ordinary
control library, before SGLang initialization or any CUDA work.  The cached
non-RDC `v4tp_49f35bd1b995105c53bc_v178mspec.so` was built against the older
`/usr` PyTorch ABI and cannot load under the fixed miniforge PyTorch 2.11
runtime used by the RDC library:

```text
undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

This is a control-artifact ABI mismatch, not a selective-dispatch, kernel,
correctness, or performance result.  Rebuild the exact non-RDC source against
the same miniforge headers/libraries in a distinct extension directory, then
retry with both libraries on one ABI.

Raw log:
`bench/results/iter713t_selective_rdc_tp4_m128_cold_smoke_20260906.log`.
