# SM90 NVFP4 Per-128 E4M3 Intermediate Candidate

## Goal

Evaluate per-128-channel dynamic E4M3 scaling for the L1-to-L2 intermediate
activation across every BN256 fused path. Keep production kernels, BN128 split,
the standard NVFP4 weight format, and the framework's per-64 physical scale
buffer allocation unchanged during the experiment.

## Isolation

Implement a separate candidate API, JIT wrapper, kernel header, and fused body.
The first candidate uses the standard 80-byte NVFP4 deployment row and current
production decoder so the experiment measures only the activation scale change.
If it passes, port the same change into a separate Pro braided three-stage
candidate and compare it with the retained Pro winner.

Do not add a production environment variable or runtime argument. Do not commit
or push the experiment until explicitly requested.

## Physical And Logical SF Layout

Keep the allocated SM90 scale tensor at `intermediate_hidden / 64` float groups
per token. The candidate stores logical per-128 groups in the first half:

```text
physical group = intermediate channel / 128
used groups     = [0, intermediate_hidden / 128)
unused groups   = [intermediate_hidden / 128, intermediate_hidden / 64)
```

Each BN256 L1 tile produces 128 post-SwiGLU channels, so `n_block_idx` maps to
one scale group. Each L2 BK128 tile consumes the matching `k_block_idx` group.

## L1 Producer

For non-swap, each epilogue warpgroup reduces its local 64-column amax and
writes one float per row into a 2 x BLOCK_M shared scratch. A 256-thread barrier
separates the writes from reads. Both warpgroups take the max of the two values,
derive the same scale and inverse scale in registers, and quantize their own
64-column half. Only warpgroup zero writes the scale to global memory.

For swapAB, retain the existing per-warp amax scratch and barriers. Warpgroup
zero reduces all eight epilogue-warp contributions and publishes one inverse
scale. Both warpgroups consume it after the existing second barrier. This adds
no swapAB barrier.

## L2 Consumer

First change the tensor map to granularity 128 and replace the two SFA loads
with one. Per-stage SFA shared memory falls from 512 to 256 bytes for BM64.
Initially retain the two K64 WGMMA groups with the same scale to isolate the
data-path change.

Then combine the four K32 WGMMA instructions for one BK128 tile into one commit
group and one wait, followed by one scaled accumulator promotion. WGMMA count
does not change. Apply the same structure independently to each swapAB weight
half.

## Iteration Order

1. Candidate control matching the production standard-layout BN256 fused path.
2. Per-128 L1 scale duplicated into both old per-64 slots; L2 unchanged.
3. Compact first-half scale storage, one SFA TMA, and 256-byte SFA stages.
4. One K128 WGMMA commit/wait and one accumulator promotion.
5. Port the winning change to the Pro braided three-stage candidate.

Each step must pass correctness before performance measurement. Temporary
experiment stages are source revisions, not runtime switches.

## Gates

- Exact-NVFP4 correctness for Flash I=2048, middle I=2560, and Pro I=3072.
- M=8/16/32/64/128, forced BN256, multiple routing seeds, and
  `global_scale=none/expert`.
- Use the repository's existing finite, cosine, norm-ratio, and small-signal
  exact-NVFP4 checks. Do not use the Cupra elementwise gate for this experiment.
- Fresh JIT compile; no local-memory spill; register count no higher than the
  corresponding control unless end-to-end performance proves otherwise.
- SASS confirms one L2 SFA TMA and, in the final stage, one WGMMA commit/wait
  per BK128 group while preserving four WGMMA instructions.
- Multi-seed process-level ABBA reports rank-zero and max-rank latency. Retain
  only repeatable end-to-end wins with no material point regression.

## Outcome

The standard-layout candidate is not suitable as an all-BN256 policy. Its Flash
results were neutral to negative and changed direction across seeds. The
standard-layout Pro M=128 result was consistently positive but only about 0.7%,
which is below the acceptance margin.

The separately ported Pro braided three-stage candidate is the useful result.
Three-seed ABBA improved rank-zero/max-rank time by 2.03%/1.61% at M=8,
2.53%/2.52% at M=32, and 3.25%/3.19% at M=64. M=16 improved 0.53%/0.47% in
the first process order but changed direction across seeds. The reversed order
improved 2.19%/2.40%; combining both orders gives 1.36%/1.44%, with a positive
combined comparison for every seed. A seed-101 sweep at
M=12/20/24/28/40/48/56 found no regression and supported a continuous
expected-tokens-per-local-expert interval rather than point gates.

Keep the Pro braided implementation isolated as the winner for expected tokens
per local expert from one through eight. Production dispatch remains unchanged
for this experiment, and the rejected generic candidate is removed.
