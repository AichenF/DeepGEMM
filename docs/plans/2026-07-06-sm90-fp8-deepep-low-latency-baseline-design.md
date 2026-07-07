# SM90 FP8 DeepEP low-latency masked baseline design

## Scope

Add a standalone benchmark-only path on
`opt/megamoe-sm90-fp8-h200-retune`.  It measures Flash, Pro, and
MiMo-V2.5-Pro at `M={8,16,32,64,128}` on eight H200 GPUs.  It imports the
current workspace's DeepGEMM implementation and never calls, initializes, or
warms up a PR323 or local fused MegaMoE entry point.

## Data flow

The timed path is exactly:

1. DeepEP `Buffer.low_latency_dispatch` with `use_fp8=True`, accepting a BF16
   MoE input and producing expert-major E4M3 tokens, FP32 K128 scales, and
   per-local-expert `masked_m` counts.
2. Current DeepGEMM masked grouped-FP8 L1 GEMM.
3. A standalone Triton masked SwiGLU plus E4M3/K128-scale quantization kernel.
   It does not apply top-k weights because low-latency combine owns the
   weighted reduction.
4. Current DeepGEMM masked grouped-FP8 L2 GEMM.
5. DeepEP `Buffer.low_latency_combine`, applying the original top-k weights
   and returning one BF16 output per source token.

The buffer uses `low_latency_mode=True`, one QP per local expert, and
`allow_nvlink_for_low_latency_mode=True`.  For each point, its token capacity
matches M.  Masked GEMM storage uses
`M_max = M * num_ranks`; `expected_m` is the balanced mean
`ceil(M * num_ranks * top_k / experts)`.

## Timing and validation

Use seed 101, five warmups, three observations, and 20 samples per observation.
Before every timed call, zero an 8,000,000,000-byte per-rank flush buffer,
synchronize CUDA, and synchronize all ranks.  The flush remains outside CUDA
events; the events enclose the complete five-stage path.  Report each
observation's maximum rank and use the median of the three maxima.

Before timing, validate output shape, BF16 dtype, and finiteness.  Independently
gather the global top-k routes and require each rank's `masked_m.sum()` to equal
the number of expert assignments owned by that rank.  Require all eight ranks,
all raw samples, stable route signatures, a full 8 GB flush, and successful
leaf exits in the strict parser.

## Integration and reporting

Keep the implementation separate from the existing ElasticBuffer
high-throughput benchmark so their layouts and input boundaries cannot be
mixed accidentally.  Add a dedicated matrix runner and strict parser.  The
source audit rejects fused entry-point references.

Compare accepted low-latency centers against the existing cold-L2
high-throughput baseline and current cold-L2 MegaMoE at the same 15 points.
Reuse the established logical compute-SOL and whole-MoE memory-SOL formulas,
while noting that low-latency dispatch includes BF16-to-FP8 input quantization
inside the measured path whereas the high-throughput baseline receives a
prequantized FP8 input.

No commit or push is made unless the user explicitly requests it.
