# Iteration 715f: combine a single SM90a LTO image with SM90a nvlink selection

Iteration 715e used the architecture-driver form at both stages. CUDA 12.8 then embedded both generic `sm_90` and architecture-specific `sm_90a` PTX/NVVM images in the compile object; device LTO attempted the generic image and ptxas again rejected WGMMA. This explains why a tiny dry-run showed a valid final `sm_90a` command yet the real multi-image object failed.

The two individually demonstrated properties are now composed without the invalid extra image: the CUDA compile line emits only explicit `compute_90a,code=lto_90a` IR, while the device-link line uses `-dlto -arch=sm_90a` so nvlink selects the architecture-specific IR and emits final SM90a code. The host/CUDA entry remains uniquely tagged and production sources remain untouched.

Iteration 715e produced no cubin and launched no CUDA business kernel. Its directory is preserved; the next build uses a fresh Iteration-715f directory.
