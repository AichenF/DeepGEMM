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
| 64 | 191.1 | 643.5 | 589.0 | -8.5% |

N32 was a one-round boundary screen; its correctness check passed.  N128 is
not reported because the pre-existing microbenchmark's N128 RS schedules
produced inconsistent upper accumulator registers for both transports.  That
is a harness-legality issue, not evidence for either decoder.

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

## Selector conclusion

MiMo M64 currently uses BM24, so its common swap-AB expert buckets are
N8/N16/N24, exactly the region where two-seed RS loses 10--13%.  The retained
H200 M64 NVFP4 result is already 505.8 us versus the 536.8 us FP8 target.
Restoring a full RS kernel for that shape would therefore move the deployment
away from the parity objective.

Keep both implementations and this harness on the experimental branch.  Do
not add either path to the default selector.  Reconsider the register-source
route only for a future workload whose dominant legal WGMMA bucket is N64 or
wider.
