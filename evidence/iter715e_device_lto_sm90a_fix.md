# Iteration 715e: preserve the architecture-specific SM90a LTO target

Iteration 715d reached device LTO, but nvlink generated PTX with `.target sm_90`; ptxas then correctly rejected every architecture-specific WGMMA/FP8 instruction. Thus the explicit `code=lto_90a` spelling accepted by the compile driver did not preserve the `a` feature set through this CUDA 12.8 device-link path.

An isolated NVCC driver probe establishes the supported recipe: `-arch=sm_90a -dlto` at compile time and device link expands to NVVM images tagged `sm=90a`, `nvlink --arch=sm_90a -dlto`, and a final cubin image tagged `sm=90a`. The composer now removes both inherited explicit `-gencode` spellings from `cuda_cflags`, adds that driver form, and replaces the device-link gencode with the same architecture form.

The full Iteration-715d attempt never produced a cubin or launched a CUDA business kernel. The partial directory remains untouched; Iteration 715e uses a new output directory.
