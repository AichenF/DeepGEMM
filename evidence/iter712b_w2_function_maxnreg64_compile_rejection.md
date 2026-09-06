# Iteration 712b — CUDA rejects device-function `__maxnreg__`

## Scope

Compile the committed Iteration-712a TP4 M128 candidate on H20 GPU1 with the
unbounded main entry, selected compact W13, whole-W2 phase outline and
`V4_SINGLE_LAUNCH_W2_FUNC_MAXNREG64=1`.  No business kernel launch was
requested.

Extension: `v4tp_44d9aceefaea5659a5ef_v178mspec`.

## Result

NVCC stops while compiling `cuda.cu` and reports:

```text
error: __maxnreg__ is only allowed on a __global__ function
```

The error names `single_launch_w2_gemm_phase`; linking, import, cubin
inspection, correctness and timing never occur.

## Decision

**Reject this language-level attribute approach.**  CUDA exposes the macro in
its headers, but its supported placement cannot create a function-specific
resource contract inside one global entry.  Preserve the failure and remove
the non-buildable probe before any default path can be affected.  A stronger
separable-device-code/device-link experiment is distinct: it must first prove
that an external W2 device callee retains standalone-like SASS without adding
a child kernel or hidden business launch.

