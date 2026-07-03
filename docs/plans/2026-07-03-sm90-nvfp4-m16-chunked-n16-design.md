# SM90 NVFP4 M16 Chunked-N16 Fused Experiment

## Goal

Recover the code-footprint benefit measured by the rejected M16 narrow
candidate without assuming that one destination expert receives at most the
source rank's `num_tokens`. The candidate remains a single fused MegaMoE
kernel, preserves the standard NVFP4 deployment layout, and targets requests
with `num_tokens <= 16`.

## Alternatives

1. Keep only N8/N16 and assume `valid_m <= 16`. This is smallest but invalid:
   `valid_m` is the receive-count sum across all ranks.
2. Keep only N8/N16 WGMMA shapes and process `valid_m` in 16-token chunks.
   This preserves the common small-count path and remains correct through the
   full BM64 tile. This is the selected experiment.
3. Split dispatch/count from compute, return the maximum expert count to the
   host, and select a narrow or general compute kernel. This gives exact host
   dispatch but adds a launch and synchronization and violates the current
   fused-only direction.

## Data Flow

For each K128 stage and each 64-row weight half, the selected candidate loops
over token bases `0, 16, 32, 48`. A full chunk uses N16 WGMMA; a final chunk
of at most eight tokens uses N8. The activation descriptor advances by
`token_base * BLOCK_K`. Promotion advances both the logical token index and
the destination accumulator offset so the existing L1/L2 epilogues remain
unchanged.

The stage empty barrier is released only after every active token chunk and
both weight halves have completed. Generic shared stores from NVFP4 decode
are published to the WGMMA async proxy before the first WGMMA reads the tile.

## Scope

- Use separate candidate kernel, JIT runtime, API, and Python harness files.
- Do not add an environment variable or runtime argument.
- Do not change the retained general three-stage body or any split kernel.
- Do not commit or push the experiment.

## Verification Gates

1. Fresh-JIT compile with no local-memory traffic or spills and no register
   increase over the 168-register general three-stage kernel.
2. Eight-rank adversarial routing with M16 and all ranks targeting one local
   expert. Rank 0 must observe 128 received tokens, exercising two BM64 tiles
   and all four N16 chunks.
3. Exact-NVFP4 correctness for M8/M16 with `global_scale=none/expert` and
   multiple routing seeds.
4. SASS must contain only the intended N8/N16 swap-AB QGMMA shapes and show
   token-base descriptor offsets without stack/local traffic.
5. Same-machine max-rank A/B/B/A at M8 and M16 across seeds 101/202/303.
   Advance only if every seed is non-regressive and the geometric gain is at
   least 1% versus the general braided three-stage kernel.
6. Compare the accepted candidate with optimized W8A8 using identical routing.
