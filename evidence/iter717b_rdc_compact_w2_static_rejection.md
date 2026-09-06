# Iteration 717b: reject compact whole-W2 RDC at the register gate

## Static result

The committed composer generated and device-linked the unique entry
`tp4_megamoe_single_launch_kernel_iter717a`.  The exact M128 compact W2
callee preserves the standalone-like issue schedule:

- 32 QGMMAs;
- 16 dependency barriers;
- 32 `WARPGROUP.ARRIVE` instructions;
- zero explicit `STL`/`LDL` instructions.

The ABI compaction does remove the high-argument stack cost, but it does not
reduce the linked register window:

| candidate | registers | stack/thread | shared | local |
| --- | ---: | ---: | ---: | ---: |
| high-argument RDC, M128 split-K2 | 195 | 112 B | 1616 B | 0 B |
| compact RDC, M128 split-K2 | 198 | 32 B | 1616 B | 0 B |
| compact RDC, M128 split-K4 | 198 | 32 B | 1616 B | 0 B |

The compact callee has no stack-frame prologue.  Its first uniform pointer
load is `LDC.64 R192, c[0x0][0x208]`, immediately showing that ptxas still
assigns the callable WGMMA body a register namespace reaching at least R193.
Thus the 195-register result was not primarily the cost of passing seventeen
arguments; it is the RDC callable-function allocation itself.

## Decision

**Reject before CUDA launch.**  REG198 admits only a small fraction of the
selected nine-CTA/SM persistent grid and fails the committed REG64 gate.  The
favorable 32/16 issue grouping cannot compensate for that structural
residency loss, as the earlier REG195 runtime already demonstrated.  No host
link, extension load, correctness run, cold-L2 timing, or speedup is claimed.

Production `v4_flash_tp_wgmma.py` and
`bench/v4_flash_tp_single_vs_multi_graph.py` are unchanged.  Artifact hashes
and exact static counts are in
`bench/results/iter717a_rdc_compact_w2_static_20260906.log`.
