# Iteration 709g: profiler cache-clear API failure

The repaired custom-only environment successfully constructed the TP4 M128
case and completed two warm local W13/requant/W2 pipelines.  Before the
profiled replay, the legacy fixture attempted
`triton_runtime.driver.active.clear_cache(...)`; the container's Triton
driver has no such method and raised `AttributeError`.

No separately cold target replay or NCU metric was produced.  The compatibility
repair will use a streaming `zero_()` of the already validated >=256 MiB cache
buffer only when the old driver API is absent.  That operation remains
immediately before the target and is excluded by the `route_gemm` NCU filter.
