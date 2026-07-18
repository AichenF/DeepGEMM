# Shawn two-seed NVFP4 screening on H200

This note preserves the two Shawn-inspired experiments without selecting them
for production.  The two-seed register decoder is derived from the E91 Humming
overlay in `grouped_shawn.py`; that overlay attributes Humming and its custom
code under Apache-2.0.

## Environment

- GPU: H200, SM90, 132 SMs
- RS mainloop launch: 132 CTAs, 384 threads/CTA
- Work per sample: 28 BK128 stages
- Stable comparison: nine alternating-order rounds
- Correctness: every N8/N16/N24/N32/N64 RS result was checked bit-for-bit
  against the retained LUT-based RS accumulator output

The benchmark is `bench_split_rs_mainloop.py`.  It models the two math
warpgroups of the split kernel.  Both RS variants use the same 1280-byte
fragment footprint and identical packed FP4 payload.  The control stores four
UE4M3 scale bytes plus padding and reads the shared LUT.  The Shawn variant
uses the padding for eight E4M3 seed bytes and performs exponent-shift
expansion in registers.

## Mainloop results

| RS WGMMA N | W8 SS cycles/stage | LUT RS cycles/stage | Shawn two-seed RS cycles/stage | Shawn vs LUT |
|---:|---:|---:|---:|---:|
| 8 | 134.4 | 471.5 | 527.6 | +11.9% |
| 16 | 145.0 | 487.4 | 551.1 | +13.1% |
| 24 | 150.1 | 514.9 | 568.0 | +10.3% |
| 32 | 155.4 | 529.8 | 584.8 | +10.4% |
| 40 | not rerun | 554.4 | 552.6 | -0.3% |
| 48 | not rerun | 580.7 | 567.9 | -2.2% |
| 56 | not rerun | 608.9 | 575.3 | -5.5% |
| 64 | 191.1 | 643.5 | 589.0 | -8.5% |

N32 was a one-round boundary screen; its correctness check passed.  The
N40/N48/N56 extension used the same 132-CTA, 28-stage, nine-round protocol,
and every result passed the same bit-exact accumulator comparison.  N128 is
not reported because the pre-existing microbenchmark's N128 RS schedules
produced inconsistent upper accumulator registers for both transports.  That
is a harness-legality issue, not evidence for either decoder.

## MiMo M512 bucket distribution and hybrid threshold

The exact seed-0 eight-rank MiMo M512 routing used by the production benchmark
contains 384 local experts and 767 swap-AB token tiles.  The measured bucket
distribution is:

| RS WGMMA N | tile count |
|---:|---:|
| 8 | 29 |
| 16 | 85 |
| 24 | 132 |
| 32 | 97 |
| 40 | 31 |
| 48 | 9 |
| 64 | 384 |

N64 is therefore 50.1% of the tiles, while the other half is mainly the tail
left after an expert's first 64-token tile.  Enabling two-seed decode for every
bucket nearly cancels its N64 gain against its N8--N32 losses: the weighted
decoder improvement is only 0.4%.  Keeping the LUT decoder for N <= 40 and
using two-seed decode only for N48/N64 improves the weighted decoder mainloop
by 4.7% (578.8 to 551.4 cycles per BK128 stage).

This decoder-only delta is about 27.4 cycles per tile and stage.  Across the 48
L1 plus 16 L2 BK128 stages and roughly six tiles on the critical persistent CTA,
it projects to only about 5.3 us at 1.98 GHz.  A full-kernel win must therefore
also come from the smaller RS shared-memory footprint and the resulting deeper
pipeline; decoder selection alone cannot close the roughly 57 us M512 gap to
the FP8 target.

## N64 SASS attribution

Both N64 predecode kernels use 74 registers, zero stack, and zero local memory.
Each issues four QGMMA instructions and two WARPGROUP instructions per
unrolled stage body.  The relevant static instruction counts are:

| Instruction | LUT RS | Shawn two-seed RS |
|---|---:|---:|
| LDS | 24 | 8 |
| PRMT | 16 | 48 |
| LOP3 | 64 | 47 |
| QGMMA | 4 | 4 |

The N64 gain comes from removing shared-LUT dependency loads under higher
accumulator pressure.  At N32 and below, the longer dependent PRMT chain costs
more than the removed shared loads.

## Shared-B transplant control

The separate two-seed shared-B transplant retained the current SS-WGMMA
mainloop and changed only the packed transport and decoder.  Cold-L2,
all-132-SM, three-call H200 measurements were:

| M | Current NVFP4 mean | two-seed shared-B mean | Regression |
|---:|---:|---:|---:|
| 512 | 927.9 us | 1520.9 us | +63.9% |
| 1024 | 1515.7 us | 2101.6 us | +38.7% |

This control is intentionally retained because it demonstrates that Shawn's
benefit depends on the register-source mainloop; copying only the transport
into a shared-B kernel is counterproductive.

## Forced-RS fused-kernel validation

The first M512 stage sweep was not an RS measurement.  Although the candidate
transform emitted the RS prepack, the host selector compiled
`kSwapABRequested=false`; those roughly 1.01 ms stage-4 through stage-7
numbers used the standard shared decoder on an incompatible payload and must
not be used as performance evidence.

After rebuilding the host extension and explicitly forcing the RS selector,
the generated kernel was checked to contain
`kLoaderDequantRequested=false`, `kSwapABRequested=true`, and the requested
stage count.  An exact-NVFP4 M128/NE16/top-k8 test covered 53--75 routed tokens
per expert, including N56/N64.  Both no-global-scale and per-expert-scale
cases passed with finite output, per-token cosine minimum 0.9986 and mean
0.9990.

Cold-L2, all-132-SM, three-call MiMo M512 results were:

| Runtime N dispatch | Stages | ptxas local stack | Mean rank | Max rank |
|---|---:|---:|---:|---:|
| 8/16/24/32/40/48/56/64 | 6 | 944 B/thread | 3955.4 us | 3964.4 us |
| 8/16/64 | 4 | 56 B/thread, no spills | 1813.9 us | 1822.1 us |
| 8/16/32/64 | 4 | 56 B/thread, no spills | 1680.5 us | 1686.1 us |

The eight-way dispatch exposes a compiler integration problem: ptxas
materializes the variant state in local memory.  Removing that spill recovers
more than half of the loss, but the best forced-RS result is still 81% slower
than the retained current-NVFP4 mean of 927.9 us.  Therefore the remaining
gap is in the full RS fused mainloop, not in two-seed numerical correctness or
the selector plumbing.

## Selector conclusion

MiMo M64 currently uses BM24, so its common swap-AB expert buckets are
N8/N16/N24, exactly the region where two-seed RS loses 10--13%.  The retained
H200 M64 NVFP4 result is already 505.8 us versus the 536.8 us FP8 target.
Restoring a full RS kernel for that shape would therefore move the deployment
away from the parity objective.

Keep both implementations and this harness on the experimental branch.  Do
not add the shared-B path or an all-bucket two-seed path to the default
selector.  The hybrid decoder remains the right decoder policy for any future
RS retry--LUT for N <= 40 and two-seed for N48/N64--but the forced full-kernel
result rules out selecting the current RS integration for MiMo M512.
