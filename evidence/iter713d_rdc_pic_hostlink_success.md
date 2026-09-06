# Iteration 713d: PIC RDC host-link success

Date: 2026-09-06

The temporary device-link rule was corrected to use:

```text
nvcc cuda.cuda.o -o dlink.o -dlink \
  -gencode=arch=compute_90a,code=sm_90a --compiler-options '-fPIC'
```

The unchanged Iteration-713b relocatable CUDA object then device-linked and
host-linked into the generated PyTorch extension successfully.

- Shared library SHA256:
  `d6253af2c851ad1e6f735b10557434a654b8901a245c9443611ecc03e7635058`.
- PIC device-link object SHA256:
  `6efbc396ae13432d341e7c1f2675e50faf49c4418840087c253394fb100d0039`.

The library still contains the temporary non-target launch-bound relaxations
documented in Iteration 713b and is therefore only a TP4-M128 experiment, not
a production artifact.  It has not yet been loaded, and no CUDA business
kernel was launched for this build result.
