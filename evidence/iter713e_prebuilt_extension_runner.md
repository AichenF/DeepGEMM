# Iteration 713e: isolated prebuilt-extension runner

Date: 2026-09-06

## Purpose

The Iteration-713d RDC library must be tested without overwriting the normal
PyTorch JIT cache or editing the benchmark's kernel module.  A small generic
runner loads the library from `V4_PREBUILT_EXTENSION`, replaces only the
matching `torch.utils.cpp_extension.load_inline` call, and executes the
unchanged benchmark through `runpy`.

The interception is name-exact.  Any other JIT extension continues through
the original PyTorch loader.  This keeps the frozen multi-kernel control,
inputs, CUDA Graph capture, cold-L2 clearing, timing, and correctness logic in
the existing paired benchmark unchanged.

## Gate

Run Python bytecode compilation before use.  Then test only TP4 M128 because
the temporary RDC CUDA source relaxed non-target launch bounds solely to make
device link possible.  Start with correctness and a short cold-L2 paired run;
do not spend a formal 10x200 run unless the candidate is competitive.

No extension load or CUDA kernel launch is claimed by this composition note.
