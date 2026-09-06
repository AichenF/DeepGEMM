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

## Gate

After build, inspect the exact linked W2 callee again.  Continue to a TP4 M128
runtime smoke only if it remains 32 QGMMAs / 16 dependency barriers with no
fixed local spill.  No CUDA compilation or business-kernel launch is claimed
by this composition record.
