# SM90 NVFP4 Dequant/WGMMA Overlap

## Goal

Reduce the small-M latency gap between the SM90 NVFP4 MegaMoE kernel and the
FP8 kernel without changing numerical results or the deployment-time NVFP4
weight layout.

## Selected Sequence

Try a single-kernel software pipeline first. Keep the BN256 fused layout and
launch, but decode stage `K+1` with the math warpgroups after committing WGMMA
for stage `K` and before waiting for that WGMMA group. If this cannot produce a
stable end-to-end win without register spills or correctness regressions, use
a BN256 two-kernel L1/L2 implementation as the fallback.

The alternatives are intentionally ordered this way:

- A dedicated 64-thread loader-dequant path already made dequant the critical
  path, so repeating that producer design is not the first experiment.
- A two-kernel split isolates L1 and L2 configuration, but adds a launch and a
  global intermediate round trip. It is a fallback rather than the default.
- Keeping the current immediate `commit -> wait -> dequant next` sequence is
  the correctness baseline, but leaves the measured FP4 decode work exposed.

## Single-Kernel Data Flow

For math-side dequant, decode the first stage as a prologue. For each K block:

1. Read the activation scale factors for the current decoded stage.
2. Issue and commit the current stage WGMMA group.
3. If another K block exists, wait for its TMA `full` barrier and decode its
   packed FP4 weights into the next stage's FP8 shared-memory buffer.
4. Fence and wait for the current WGMMA group, consume its accumulators, and
   release only the current stage's `empty` barrier.
5. Advance the stage index and parity. The next iteration consumes the stage
   decoded in step 3 without decoding it again.

The lookahead operation must run exactly once per K block on every math thread.
For L2 paths with two WGMMA groups per K block, it is placed after the first
commit so it overlaps the longest available tensor-core window. Swap-AB paths
use the first committed subgroup. Loader-dequant configurations retain their
existing producer/consumer protocol and do not use this lookahead.

No new persistent environment variable, runtime argument, weight copy, or
request-time layout choice is introduced.

## Synchronization Invariants

- The next stage is decoded only after its `full` barrier reaches the expected
  phase, so TMA has completed both activation and packed-weight transactions.
- The current stage's `empty` barrier is released only after all WGMMA reads
  from that stage have completed.
- Decoding the next stage never changes the current stage's shared-memory
  descriptors or barrier phase.
- All 256 math threads follow the same lookahead control flow; each thread
  decodes its existing BN256 row and therefore preserves the current FP8 byte
  layout exactly.
- WGMMA accumulator registers are not read until `warpgroup_wait<0>()`.

## Resource Risk

The decode temporaries become live while the asynchronous WGMMA accumulator is
live. This can raise the math-thread register count even though CTA occupancy is
already limited to one CTA per SM by shared memory. The experiment is rejected
if ptxas reports local-memory spills or if the added register pressure causes a
stable latency regression. SASS inspection must also confirm that decode
instructions are scheduled between `WGMMA.COMMIT_GROUP` and
`WGMMA.WAIT_GROUP`.

## Verification

Use a fresh JIT cache for every generated specialization.

- Correctness: Flash `I=2048`, Pro `I=3072`, and middle `I=2560`; include
  swap-AB on/off points and both `global_scale=none` and `expert`. Require the
  same numerical thresholds as the existing policy matrix, with no relaxed
  tolerance.
- Resource check: compare threads, dynamic shared memory, registers, local
  memory, and spills with the current commit.
- Focused performance: run matching-routing ABBA comparisons at
  `M={8,16,32,64,128}` for Flash and Pro, using multiple routing seeds and the
  existing robust statistic. Phase profiling must show reduced exposed
  math-side decode time rather than only run-to-run noise.
- Guardrails: include representative `M=256` and a non-swap middle-I point so
  a small-M improvement does not silently regress the generic fused path.

Keep the single-kernel implementation only when correctness is unchanged, no
spills appear, and the aggregate small-M result improves stably without a
material guardrail regression. Otherwise revert the experiment cleanly and
move to the BN256 two-kernel fallback under the same acceptance criteria.

## Outcome

No overlap variant met the acceptance criteria. All experimental source and
benchmark changes were removed; the production kernel remains byte-for-byte
identical to commit `a9c6e0d`.

| experiment | correctness/resource result | representative latency result |
|---|---|---|
| Math-WG K+1 lookahead | Full policy matrix passed, but SASS moved `WARPGROUP.DEPBAR` before the decode stores and ptxas introduced spills | Flash M64 regressed to 872.8 us from the approximately 470--480 us baseline used for that run |
| 512-thread dedicated producer | Numerically correct, but the 128-register launch cap caused up to 440-byte stores and 320-byte loads of spill traffic | Rejected before a broad benchmark sweep |
| 384-thread dispatch-assisted producer | Flash/Pro focused correctness passed; 168 registers, 56-byte stack, no local spills | Flash M64 890.7 -> 1230.0 us (+38.1%); Pro M64 1234.0 -> 1582.0 us (+28.2%) |
| BN256 split L1/L2 | Flash/Pro M64 passed for both global-scale modes, minimum cosine 0.9988 | Flash M64 871.5 -> 905.7 us (+3.9%); Pro M64 1145.5 -> 1357.7 us (+18.5%) |
| Two existing TMA warps decode | Flash/Pro correctness passed; 168 registers, 56-byte stack, no local spills | Flash M64 868.0 -> 1607.0 us; Pro M64 1162.0 -> 2147.0 us; Flash M128 844.2 -> 1591.0 us |
| One wide-N math WG plus dedicated producer | Flash M64 correctness passed, but ptxas used a 920-byte stack and SASS contained about 1003 `LDL`/`STL` instructions | Flash M64 regressed to 2834.0 us from 868.0 us |

The failures have distinct causes. Same-warpgroup stores cannot be scheduled
inside the useful WGMMA window by ptxas for this code shape. Reusing dispatch
warps serializes token movement with GEMM. Reusing the two TMA warps makes each
producer wait for a completed stage and destroys TMA lookahead. A second
kernel adds more launch and intermediate traffic than the isolated L2 overlap
saves. Reducing to one wide-N math warpgroup avoids extra CTA threads but makes
the N256 accumulator spill heavily.

The next viable direction requires a different resource balance, not another
placement of the same decode loop. In particular, it needs a dedicated
producer warpgroup without reducing two N128 math warpgroups and without
raising the CTA above the 384-thread register regime. Hopper does not provide
that budget in the current fused CTA layout.
