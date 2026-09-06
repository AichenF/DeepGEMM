# Iteration 715d: remove the redundant compile-side `-dlto`

Iteration 715c correctly changed both CUDA compile targets to `code=lto_90a`, but NVCC still rejected the compile command because it also retained an explicit `-dlto`. With explicit `-gencode`, `code=lto_90a` is itself the compile-side request to emit LTO IR; `-dlto` is required only on the device-link command that emits final `sm_90a` code.

The composer now leaves `-rdc=true` unchanged on the compile command, changes its two code targets to `lto_90a`, and adds `-dlto` only to the final `sm_90a` device link. The partial Iteration-715c directory is retained and the next attempt uses a fresh Iteration-715d directory.

The failed attempt did not reach ptxas, device link, a CUDA business-kernel launch, correctness, or timing.
