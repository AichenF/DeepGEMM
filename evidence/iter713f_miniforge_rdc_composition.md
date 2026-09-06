# Iteration 713f: baseline-ABI RDC composition

Date: 2026-09-06

## Why another link is required

The initial RDC object was compiled with the container's `/usr` PyTorch
headers and libraries.  The existing paired TP benchmark uses
`/home/xutingz/workspace/miniforge3` because its PyTorch exposes
`float8_e8m0fnu`, which the unchanged Humming utility import requires.
Cross-loading the `/usr`-linked extension under miniforge fails on an undefined
C10 CUDA symbol, so timing that mixture would be invalid even if it loaded.

Verified target environment:

- PyTorch `2.11.0+cu129`;
- `_GLIBCXX_USE_CXX11_ABI=True`; and
- `torch.float8_e8m0fnu` present.

## Isolated build change

Retain the exact patched generated CUDA source from Iteration 713b, its
`-rdc=true` compilation, PIC device link, extension name, CUDA toolkit, and all
kernel defines.  Change only:

- PyTorch include roots to the miniforge `site-packages/torch/include` tree;
- Python include root to `miniforge3/include/python3.12`; and
- PyTorch library root to the miniforge `site-packages/torch/lib` tree.

No production source or benchmark logic changes.

## Build and static result

The miniforge-targeted relocatable compile, PIC device link, and shared-library
link all succeed.

- Raw build log SHA256:
  `3a9bf7525edba320c56c05e8a7601ed15a07c939e3ef6972ea8f695b8ecb1d9c`.
- Candidate library SHA256:
  `8664907620654c14b0e158614253db6c5caafdd1c3dc76370226315888f9c488`.
- Exact W2 callee SASS SHA256:
  `adf4eb6fd60c7954ca7c2bec966d6dc005c160d0f78d66fbc27804daca7f418e`.
- Exact W2 counts: 32 QGMMAs, 16 dependency barriers, 32 arrives.
- TP4 M128 split-K2 resources:
  `REG195 STACK112 SHARED1616 LOCAL0`.

The exact W2 SASS and resource report are byte-identical to the `/usr`-header
Iteration-713b artifacts, showing that the ABI rebuild did not change device
scheduling.  The static gate passes.  Proceed to the isolated TP4 M128
correctness and short cold-L2 screen.  No CUDA business kernel has launched
for this build/static result.
