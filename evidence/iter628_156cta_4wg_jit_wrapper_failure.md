# Iteration 628: 4-WG helper repair, JIT-wrapper failure

## Source repair

`reduce_swiglu_quant_task` now permits `IndependentTaskWGs` in `{1,4,8}`. Its indexing, shared arrays, and named-barrier helper are already parameterized by that value; the stale assertion was the compile blocker found in Iteration 627.

## Test configuration

The isolated 156-CTA/4-WG extension was rebuilt from an empty exact JIT directory on H20 GPU 1 with SM90a and the same flags recorded in Iteration 627.

## Result

No NVCC/PyTorch build exception appeared after the repair. The command nevertheless failed in its post-import extension-name print:

```text
TypeError: bad operand type for unary +: 'str'
```

The cause is shell quote removal around the nested Python string literal, not CUDA compilation. Since the wrapper stopped before checking the `.so` and running `cuobjdump`, this iteration does not pass the resource gate. The next gate will address the deterministic path `v4tp_63a318a5fa109f26b014_v178mspec` directly and record the cubin resources.
