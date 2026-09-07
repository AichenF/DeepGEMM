# H20 Exact Outer-Pipeline TP MegaMoE Experiment

## Goal

Build an isolated experimental TP MegaMoE specialization that reuses the
current MXFP4 implementation but follows the H20 small-M outer pipeline from
`origin/megamoe_nvfp4_dev` as closely as the TP contract permits.  The
production flat and existing native paths remain unchanged.

The timed operator accepts caller-provided FP8-E4M3 activation and group-128
scales, precomputed routing metadata, and canonical MXFP4 weights.  It performs
route preparation, W13, SwiGLU and intermediate FP8 quantization, W2, ordered
weighted route reduction, one TP all-reduce, and replay cleanup in exactly one
CUDA kernel launch per rank.

## Reused implementation

Reuse `v4_flash_tp_native_megamoe.py` for weight preprocessing, workspace
layout, TMA descriptors, graph-stable communication pointers, host validation,
and TP collective helpers.  Reuse the current MXFP4 Mode2 decoder and numerical
semantics.  Do not modify the dirty main DeepGEMM checkout.

Add a separately named body and launch specialization so its generated cubin,
flags, and benchmark metadata cannot be confused with the selected flat path
or the earlier H200-derived native experiment.

## Outer pipeline

Launch exactly 78 cooperative CTAs of 384 threads on H20, with one persistent
CTA initially mapped to each SM.  Preserve the reference role partition:

- threads 0..63: TP-local route/pool formation, readiness publication, and
  cleanup coordination; no EP remote pull or ownership logic;
- threads 64..127: independent activation and packed-weight TMA producers;
- threads 128..383: two 128-thread WGMMA consumer/epilogue warpgroups.

Use the H20 common-body expert-wave scheduler, BN256/BK128 task geometry,
swap-AB small-M WGMMA policy, and resource-derived three/four-stage pipeline.
The runtime SM count is 78 rather than a model-specific constant.

For this first experiment, preserve the reference dependency boundary: W13
and SwiGLU quantize into the global FP8 intermediate pool plus group-128 scale
pool; readiness is published per pool/N tile; W2 reloads that pool.  This
isolates the outer scheduling experiment.  A shared/register-only handoff is a
separate follow-up and is not part of this implementation.

## TP adaptation

All 256 experts are local to every TP rank.  The two former dispatch warps only
form the local expert-major pool from replicated X/routes and maintain replay
state.  There is no EP dispatch, remote expert ownership, return scatter, or EP
combine barrier.

W2 emits the same BF16 per-route values as the current implementation.  The
existing ordered weighted k=6 reduction forms the rank-local `[M,4096]`
partial, followed by the existing embedded TP4 collective.  TP8 uses the same
one-launch outer specialization and a supported embedded/fallback collective
path without adding a second timed launch.

## Isolation and safety

Expose the experiment behind a separately named Python entry and JIT identity.
No unset environment default changes.  Workspace capacity and cooperative
launch occupancy are checked before launch.  Route inputs may change on every
CUDA Graph replay; no host route inspection is permitted.

## Validation

1. Compile and inspect resources/SASS before distributed execution.
2. Single-GPU local correctness at M=8 and M=128 against the canonical MXFP4
   reference, including W13 intermediate and W2 route output checks.
3. TP4 correctness for M=8,16,32,64,128, including embedded-collective versus
   NCCL oracle checks.
4. Nsight Systems verification of one business-kernel node per graph replay.
5. Same-process paired TP4 benchmark versus the selected multi-kernel baseline,
   with a separate excluded 256 MiB L2 clear before every replay.  Report every
   M and equal-weight geometric mean; do not infer a win from stage timings.
6. TP8 compile, correctness, and one-launch run-through.

The experiment is retained only if it is correct and offers a credible path
toward the current approximately 10% single-kernel deficit.  Reference EP8
latencies are architectural context, not a TP performance claim.
