# SM90 NVFP4 Pro Dispatch/Compute Split Experiment

## Goal

Measure whether removing dispatch from the Pro compute CTA creates enough
thread and register budget for a dedicated 128-thread NVFP4 dequant producer.
The experiment targets the retained BN256 Pro N24 candidate and does not alter
the production fused or split kernel bodies.

## Candidate Boundary

The candidate launches two kernels on the same CUDA stream:

1. A 64-thread dispatch-only kernel counts routes, publishes cross-rank
   receive counts, and pulls FP8 tokens, activation scales, top-k weights, and
   source metadata into the existing symmetric-buffer pool.
2. A 384-thread compute-only kernel contains four non-epilogue warps and eight
   math/epilogue warps. It runs the existing fused L1/L2 scheduler, epilogues,
   combine, and final workspace cleanup with no dispatch warps.

The dispatch kernel does not clean workspace state. Kernel completion on the
same stream is the publication boundary for all pulled data. The compute
kernel performs cleanup only after combine and finishes with the existing
cross-rank cleanup barrier.

## Attribution Steps

The first control specialization preserves the retained candidate's two-stage
pipeline and math-side dequantization. Its comparison with the one-kernel N24
candidate measures the cost of the extra launch and loss of dispatch/compute
overlap without changing the decoder.

Only if that control cost leaves a plausible margin, the second specialization
enables all four non-epilogue warps as an in-place dequant producer. Math waits
on the per-stage dequant barrier instead of decoding weights itself. Pipeline
depth may be tuned only after a same-stage-count comparison has isolated the
producer effect.

## Invariants

- External and cached weights remain standard lossless NVFP4 E2M1 plus one
  nonnegative UE4M3 scale per 16 values in the retained 80-byte row layout.
- No persistent FP8 weight copy, request-time prepack, environment variable,
  or production runtime argument is introduced.
- Dispatch and compute kernels live in separate candidate implementation
  files; existing fused and split bodies remain unchanged.
- Dispatch completes all TMA stores before returning. Compute starts later on
  the same stream and reads finalized receive counts and arrival counters.
- Compute owns cumulative receive statistics, workspace zeroing, and the final
  cross-rank barrier exactly once.

## Acceptance

Use fresh JIT artifacts. First require exact-NVFP4 correctness at M=8 and M=64
for `global_scale=none` and `global_scale=expert`, then inspect registers,
stack, spills, dynamic shared memory, and both kernel symbols. Screen the
control and producer variants against retained Pro N24 with identical routing.
Only a stable win advances to three-seed A/B/B/A; otherwise remove the
candidate wiring and record the rejection. Do not commit or push.

## Outcome

The architecture was rejected and all candidate source/API wiring was removed.
Both the two-stage math-dequant control and the 128-thread producer passed the
real Pro `H=7168, I=3072, E/rank=48, topk=6` exact-NVFP4 matrix at M=8/64
with `global_scale=none/expert`; minimum per-token cosine was 0.9988. The
producer compute cubin retained 168 registers, a 56-byte stack, and zero local
memory; dispatch used 48 registers, a 32-byte stack, and zero local memory.

At eight ranks, M64, seed offset 101, and 30 logical-call samples, the control
measured 1377.3 us median: 66.9 us dispatch plus 1309.3 us compute. This was
already about 2.3% slower than the retained N24 candidate's approximately
1346.9 us max-rank result. The dedicated producer measured 2325.3 us median,
with 2292.9 us in compute, regressing 68.8% versus the control and 72.6% versus
N24.

The producer must join the full-stage barrier before in-place expansion, so
the two TMA warps cannot continue prefetching while the four-warp decoder runs.
Separating dispatch creates the thread budget but does not create an
independent load/dequant pipeline; it serializes TMA progress behind decode.
No multi-seed run or NCU replay was warranted. Raw logs remain under
`/root/fac/scripts/megamoe/nvfp4_pro_dispatch_compute_20260703/`.
