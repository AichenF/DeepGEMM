# Iteration 714b: unique RDC static composition

Date: 2026-09-06

The committed composer generated `/tmp/iter714b_unique_rdc` from the exact
Iteration-713f RDC source/build.  All asserted replacement counts pass.

Unique symbols:

```text
host: run_tp4_megamoe_single_launch_iter714b
CUDA: tp4_megamoe_single_launch_kernel_iter714b
```

The pybind surface remains `run_tp4_megamoe_single_launch` and points to the
unique host identifier.  Grep confirms one host declaration, one unique CUDA
definition, four unique launch references, and one unique CUDA host wrapper;
no original internal TP4 host/kernel identifier remains outside the two
intentionally canonical pybind strings.

Generated hashes:

```text
cuda.cu     bc5623fa0370169c71f19c722a0b2ff8b9c0d4aa34b71d3510733d98169747ae
main.cpp    e4c4af592a8283d2c473af80ce4f16c506376e4fc50c0a679fad7456c02588b5
build.ninja c4fb43ad0fc9c64d4ac9832d376f444cd72b4790ab3d092cd25269a951442843
```

This is a source-composition result only.  No compilation, module load or CUDA
work has occurred in the new directory yet.
