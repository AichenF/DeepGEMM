# Iteration 713w: same-ABI non-RDC control build

Date: 2026-09-06

The complete validated Iteration-711g configuration was rebuilt through the
ordinary non-RDC `load_inline` path under the same miniforge PyTorch 2.11
process used by the RDC candidate.  Import and compilation succeed and the
computed extension identity is exactly:

```text
v4tp_49f35bd1b995105c53bc_v178mspec
```

Artifact:

```text
/tmp/iter713w_miniforge_nonrdc/v4tp_49f35bd1b995105c53bc_v178mspec/
  v4tp_49f35bd1b995105c53bc_v178mspec.so
```

The library is 9,964,120 bytes with SHA256
`6fb4add32d4f4458dbf338cb5a8b0ae1b18425012b3e97026ffab7ca52944b1b`.
Its generated ordinary source SHA256 is
`26d10ebbc86de2ec3440e2949066034c73242d23d277fcd1a1982dc3ccf16fca`.
The Ninja graph has a direct `cuda_compile` object and no RDC/device-link
stage.  The successful in-process import is the ABI gate; no business kernel
was launched by this build-only command.

The RDC generated-source copy intentionally differs only in the temporary
launch-bound relaxations documented in Iteration 713b, while retaining the
same extension configuration identity.  The new ordinary artifact is now the
control owner for selective dispatch.
