# Iteration 627: 156 CTA × 4 WG resource gate

## Purpose

Reject the 156-CTA/4-WG topology before runtime unless its fresh SM90a cubin compiles and can be checked for two-CTA-per-SM resource feasibility.

## Isolated configuration

- `CUDA_VISIBLE_DEVICES=1`
- `TORCH_CUDA_ARCH_LIST=9.0a`
- `V4_SINGLE_LAUNCH_TP4=1`
- `V4_SINGLE_LAUNCH_COMPACT_W13_BUNDLE=0`
- `V4_SINGLE_LAUNCH_156CTA_4WG=1`
- `V4_SINGLE_LAUNCH_ROUTE_DYNAMIC_SMEM=0`
- `V4_SINGLE_LAUNCH_M128_BOUND9=0`
- `V4_SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL=1`
- `V4_SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS=1`

## Observed result

The fresh extension `v4tp_63a318a5fa109f26b014_v178mspec` did not compile. NVCC reached both relevant TP4 instantiations and rejected the same stale invariant:

```text
static assertion failed
static_assert(IndependentTaskWGs == 1 || IndependentTaskWGs == 8);
reduce_swiglu_quant_task<..., IndependentTaskWGs=4>
SplitK=2, Tokens=8
SplitK=4, Tokens=8
```

Because compilation failed, no `.so`/cubin existed and no resource, correctness, or timing result can be claimed. The later empty-extension shell diagnostics are secondary fallout after the primary NVCC failure.

## Disposition

This iteration is a failed gate, not a performance result. Permit `IndependentTaskWGs == 4` in the activation reduction helper, rebuild under a new iteration, and leave the selected production path untouched.
