# Iteration 713c: RDC host-link PIC rejection

Date: 2026-09-06

The Iteration-713b relocatable CUDA object and device-link object were passed
to the generated PyTorch extension host link without changing production
source.

`main.o` compiled successfully.  The final `c++ -shared` step failed with:

```text
dlink.o: relocation R_X86_64_PC32 against symbol ... can not be used when
making a shared object; recompile with -fPIC
```

The temporary `cuda_devlink` Ninja rule omitted the host PIC option.  This
failure occurs after the already validated linked device SASS was produced;
it does not alter the 32-QGMMA/16-DEPBAR result.  No extension was loaded and
no CUDA business kernel was launched.

Decision: add `-Xcompiler -fPIC` only to the temporary `nvcc -dlink` command,
rebuild `dlink.o`, and retry the shared-library link.
