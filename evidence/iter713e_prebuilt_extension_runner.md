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

## Runtime-environment preflight

The runner itself works, but the first linked library cannot be used for the
formal paired benchmark because it was built against the container's `/usr`
PyTorch.  The unchanged benchmark imports Humming helpers, whose dtype table
requires `torch.float8_e8m0fnu`; that dtype is absent from `/usr` PyTorch.
Conversely, loading the `/usr`-linked library under the established miniforge
benchmark environment fails on an undefined C10 CUDA symbol, confirming a
PyTorch-library ABI mismatch.

The established miniforge environment reports PyTorch `2.11.0+cu129`, C++11
ABI enabled, and E8M0 dtype present.  The probe library must therefore be
rebuilt against that same environment before the runner can perform a valid
test.  These were import-only failures; no CUDA business kernel launched.
