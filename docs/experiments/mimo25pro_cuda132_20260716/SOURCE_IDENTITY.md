# MiMo-V2.5-Pro CUDA 13.2 release identity

This directory documents the exact kernel source used by the CUDA 13.2 H200 performance run.

## Branch boundary

- Upstream repository: `AichenF/DeepGEMM`
- Original branch (unchanged): `megamoe_nvfp4` at `ba7ee0944c1fe31874b049ae354657ff62dae20b`
- New sibling branch: `megamoe_nvfp4_mimo25pro_cuda132_final`
- Final pre-compatibility source: `75186dde9dac140c053c9007ace0ce7cce41150c`
- CUDA 13.2 code commit: `a8f17ad58f97b49b33ed63a4ed7a7304c897d4af`
- CUDA 13.2 code tree: `80d517297b932911f785b6ffaf1b6adf608f73c8`

`75186dd` is a strict, linear descendant of `ba7ee094`: 13 commits, no merges and no history rewrite. The new branch preserves those commits. It does not move or force-push the original branch.

## CUDA 13.2 compatibility boundary

Raw `75186dd` does not compile in the pinned CUDA 13.2 environment. Commit `a8f17ad` adds the required dependent-template disambiguator at exactly 16 call sites in three files:

- `sm90_nvfp4_mega_moe_nibble_group_fused_body.inl`: 6 sites
- `sm90_nvfp4_mega_moe_fused_body.inl`: 6 sites
- `sm90_nvfp4_mega_moe_split_l1_body.inl`: 4 sites

The sealed patch is [available here](evidence/perf/final_deepgemm-75186dd-cuda132-dependent-template-compat.patch), with SHA-256:

```text
333154d5254467c5e4a399d5286b414c405dca7566f927f93cca76ea30fcb07d
```

It changes `.get_base_ptr<float>()` to `.template get_base_ptr<float>()`; it does not alter the math, memory layout, block-N policy or dispatch policy. The post-patch hashes in [final_cuda132_patch_file_hashes.tsv](evidence/perf/final_cuda132_patch_file_hashes.tsv) match the source at code commit `a8f17ad`.

Do not substitute the older baseline compatibility patch. That patch covers only two files and ten sites, so it omits the six sites in the final grouped-nibble path.

## Performance execution identity

- Hardware: one 8×H200 NVSwitch node; 56/56 directed mapped-P2P pairs passed
- Driver: `595.58.03`
- Image: `nvcr.io/nvidia/pytorch@sha256:f572dd504a3fef02277c21f228977f100f7831576ac73140a250c473f74d3ad3`
- PyTorch: `2.11.0a0+a6c236b`
- CUDA runtime/toolkit: 13.2
- DeepGEMM JIT nvcc: CUDA 13.2 V13.2.51
- Shape: H=6144, I=2048, E=384, top-k=8, ranks=8
- Protocol: one excluded full warmup, then 30 fresh full processes; each process ran all 11 M values and `--num-tests 20`

### Supplemental Flash/Pro execution

The same sealed final source and CUDA 13.2 image were later measured on two additional ImagePerf geometries:

- Flash: H=4096, I=2048, E=256, top-k=6, ranks=8
- Pro: H=7168, I=3072, E=384, top-k=6, ranks=8
- Final protocol: one excluded full warmup per model, then 30 fresh full processes per model, alternating Flash/Pro order; every process ran all 11 M values with `--num-tests 20`
- Baseline identity: original `ba7ee0944c1fe31874b049ae354657ff62dae20b` plus its sealed two-file/ten-site CUDA 13.2 syntax-only compatibility patch
- Hardware relation: baseline and final used the same physical H200 node, exact GPU UUID order, driver and container, but in separate allocations and time windows; this was not an interleaved baseline/final A/B run
- Health gates before the final run: 56/56 mapped-P2P pairs, baseline Pro M=8192 `--num-tests 500`, and final Pro M=8192 `--num-tests 500` all passed
- Result boundary: the run is structurally complete, but Flash and Pro are not overall performance wins; see the [supplemental comparison](evidence/perf/final_comparison_flash_pro_baseline_vs_final_30.md)

The ImagePerf values in that comparison are frozen reference constants transcribed from the supplied chart. They were not produced by rerunning ImagePerf code in the supplemental allocation.

The SGLang integration reviewed with this release is `b6a68c9acb6590b2849febe2b66807553923fc71`; it lives in the separate SGLang repository and is not part of this branch.

## Source history after `ba7ee094`

```text
bd4cee1  race fix
e5944f9  single-active-dispatch-warp 3-stage pipeline and N24 policy
d67b701  candidate-body race fix
6a070f1  defensive math-side dequant fence
d1c00cb  validation record
e65435f  grouped-nibble production path
b784815  grouped-nibble cutoff alignment
eb8bb43  historical validation record
f55dfef  MiMo prepack lifecycle
211f6f0  lifecycle tests
816c7d2  fail-closed hardening
e316323  fixed MiMo BN256 grouped policy
75186dd  full-row expert-move contract
a8f17ad  CUDA 13.2 dependent-template compatibility
```
