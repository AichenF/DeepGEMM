# SM90 NVFP4 Pro Braided LUT-Window Candidate

## Goal

Transfer the isolated decoder win from a lossless braided E2M1 prepack plus a
next-quad LUT window into the real Pro M <= 64 MegaMoE kernel, while preserving
the current N24 winner and all production files.

## Deployment Layout

The external model remains standard NVFP4. At model load, each group of eight
E2M1 codes is bijectively rearranged inside the same four bytes. The low three
bits of the first four nibbles form one direct PRMT selector and the next four
form the second selector. Their eight sign bits are braided into the unused
selector high bits so the original two FP8 sign masks remain direct.

The cached row remains exactly 80 bytes: 64 bytes of E2M1 payload, eight
original UE4M3 scale bytes, and eight padding bytes. No FP8 value, expanded
table, or request-time repack is introduced.

## Decoder Schedule

Load the first two shared LUT entries, then recursively keep one next K32
quad's two LUT entries live before decoding the current quad. The braided
selector removes the DP4A chains that previously made this wider LUT window
use 60 registers. In the isolated SM90 kernel the combined schedule uses the
same 42 registers as the current decoder, with no stack or local memory.

The real implementation is a new Pro-braided API, JIT wrapper, kernel header,
and fused body. It does not modify the production fused/split bodies or the
retained standard-layout Pro candidate.

## Gates

1. Bit-exact row decoding over model-like, uniform, and exhaustive scale
   distributions.
2. Exact-NVFP4 M=8/64 correctness with `global_scale=none/expert`.
3. Real cubin at 168 registers or fewer, no additional stack/local traffic,
   and SASS showing the intended early LUT loads without DP4A selector pack.
4. Multi-seed process-level ABBA against the retained N24 candidate. Only a
   repeatable end-to-end gain is retained, followed by a fresh comparison with
   optimized W8A8 under identical routing.

No commit or push is allowed during the experiment.

## Three-Stage Follow-up

Phase profiling after the braided decoder showed `math_dequant` falling by
13.4%, but `math_full_wait` rising by 16%. The fused Pro configuration uses
two 61,968-byte stages and misses a third stage by only 5,824 bytes.

The selected follow-up keeps the 64 dispatch threads for all CTA-wide
barriers but makes only one dispatch warp perform routing and token pulls. Its
single 7,168-byte send buffer frees enough shared memory for a third pipeline
stage. A separate 3-stage API and JIT wrapper instantiate this compile-time
choice and recompute the final SMEM size; the accepted two-stage braided
candidate remains available as the direct A/B baseline.

Alternatives are deferred: a padded 128-byte deployment row permits in-place
decode but raises weight traffic by 60%, while overlapping dispatch storage
with GEMM stages is unsafe because dispatch and compute run concurrently.

The three-stage variant must pass the same correctness/resource gates and a
multi-seed ABBA against the two-stage braided candidate. Phase profiling must
show that reduced full-barrier waiting outweighs the lower dispatch parallelism.

## Dual-Warp Chunked-Pull Follow-up

The single-warp three-stage variant wins broadly, but raises dispatch-pull
time. A second three-stage experiment retains both active dispatch warps and
gives each a 4 KB shared pull buffer. Each 7,168-byte Pro token is transferred
as 4,096 plus 3,072 bytes through the same buffer. The two buffers consume
8,192 bytes instead of 14,336, putting three stages at 232,128 bytes, 320 bytes
below the SM90 limit.

This trades one extra TMA load/store pair per token for two-way dispatch
parallelism. It is implemented as a separate compile-time API and compared
directly with the accepted single-warp three-stage candidate. It is retained
only if correctness is unchanged and M=8/16/32/64 ABBA improves; a dispatch
phase win alone is insufficient.

Result: rejected. Correctness passed and M=16 improved by about 1.06% rank-0
and 0.95% max-rank over three-seed ABBA, but M=8 and M=64 regressed and M=32
was neutral. The extra TMA load/store pair made the benefit too narrow for an
interval policy, so the experimental API and compile-time branch were removed.
