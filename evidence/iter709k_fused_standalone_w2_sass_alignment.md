# Iteration 709k — fused versus standalone W2 SASS alignment

This is a read-only comparison of the two application-cold-L2 reports from
Iterations 650b and 709j.  It launches no CUDA work and makes no new endpoint
latency claim.

## Exact intervals

- Fused W2: addresses `[0x7ffb9fe4d950, 0x7ffb9fe527c0)`.  The interval
  starts when the persistent M128 W2 task body acquires its 31,872 task-warps,
  includes the static grid-stride backedge at `0x7ffb9fe52710`, and stops
  before terminal whole-grid polling.
- Standalone W2 task: addresses
  `[0x7ffee1a91750, 0x7ffee1a96a00)`, after the standalone launch's
  grid-bound/valid-task exit prefix and through its route-task epilogue.

The compact helper `/home/xutingz/fac/analyze_ncu_sass.py` imported only the
saved reports with NCU's SASS SourceCounters page and aggregated rows in
these address ranges.

## Comparison

| metric | fused persistent W2 | standalone W2 task |
|---|---:|---:|
| static SASS rows | 1,255 | 1,323 |
| executed instructions | 36,758,906 | 38,225,312 |
| QGMMA static / executed | 32 / 1,019,904 | 32 / 1,019,904 |
| `WARPGROUP.DEPBAR.LE` static / executed | 32 / 1,019,904 | 16 / 509,952 |
| registers/thread, complete entry | 56 | 61 |
| previously measured cold W2 phase median | 107.200 us | 98.240 us |

The fused body executes 3.84% fewer total instructions and exactly the same
MMA work, so omitted standalone math or generic instruction bloat is ruled
out.  The concrete code-generation loss is that fused W2 serializes every
N64 QGMMA result behind its own dependency barrier, while standalone W2
keeps both N64 halves of an N128 task in flight and emits one dependency
barrier per pair.  In the saved SASS, for example, fused QGMMAs at
`0x...4e740` and `0x...4e8a0` each have an adjacent dependency barrier;
standalone QGMMAs at `0x...a92470` and `0x...a926c0` share the barrier at
`0x...a92880`.

Raw PC-sample totals differ across separately replayed kernels and are not
used as a cycle conversion.  The doubled executed dependency barriers and
the 56-versus-61-register contracts support, but do not alone prove, the
inference that the nine-CTA launch cap forces ptxas to serialize accumulator
groups.

## Bounded next experiment

Current W2 S2R lookahead carries, per N64 output group, two packed-weight
words plus two decoded `uint2` LUTs across the current QGMMA: six 32-bit
registers.  Predecoding the next pair before the current QGMMA and carrying
only two FP8 `uint2` values uses four registers, saving two registers per
group/four per N128 task while preserving the selected ahead-of-QGMMA packed
loads, LUT synthesis and integer decode.

Implement this only as a default-off, fused-TP4-M128 W2 template option.  It
is distinct from the rejected dual-K32 accumulator chains and from
Iterations 687--690, which removed/deferred selected lookahead or added a
shared broadcast.  The first gate is generated SASS: retain 56 registers,
nine CTAs/SM, no fixed spill and reduce dependency barriers from 32 toward
16.  Correctness and cold-L2 timing are forbidden unless that static gate
passes.
