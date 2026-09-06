# Iteration 685 — reject the 160-thread lean TMA producer transfer

## Hypothesis

The selected multi-kernel W13 and W2 entries let their 128-thread math warp
group consume tiles while an independent warp issues weight/scale TMA.  Test
whether that producer/consumer split is the useful part of the multi-kernel
implementation that can be transferred into the one-business-kernel path.

For M=128 only, the candidate therefore used a 160-thread CTA:

- warps 0--3 retained the existing 128-thread WGMMA math group;
- warp 4 issued the compact two-stage weight/scale TMA pipeline for both W13
  and W2;
- activation and TP communication kept only the first 128 threads active;
- the grid changed from 702 CTAs at 9 CTA/SM to 546 CTAs at 7 CTA/SM, keeping
  the resident math-warp budget close to the selected implementation;
- per-stage empty barriers were taken from shared barrier slots unused by the
  corresponding GEMM phase.

Candidate source SHA-256:
`3d3a3edf1d234df6cda2d25a1fad203ce7759faa16393305bfa875a62d6acbfc`.
Selected production SHA-256:
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.

## Gates

- Python and CUDA JIT compilation passed for the M128 split-K2/split-K4
  specializations.
- The complete M128 entry compiled at `REG56 STACK64 SHARED2048 LOCAL0`.
  Production is `REG56 STACK32 SHARED2048 LOCAL0`, so the transfer doubled
  the stack frame even though it introduced no local-memory spill.
- Single-GPU random-route full-output validation passed the repository's
  tolerance: cosine `0.9999981035`, relative L2 `0.0019475397`, finite output,
  1,992 padded rows, and packed generations `[0,0,0,0]`.
- TP4 validation also passed, including the embedded allreduce:
  `allreduce_ok=true`, minimum-rank cosine `0.9999936035`, maximum-rank
  relative L2 `0.0035767262`, and finite output on every rank.

## Benchmark

- Hardware: one node, four H20 GPUs, physical devices 0/5/6/7, 78 SM/GPU.
- Shape: DeepSeek-V4-Flash TP4, M=128, random routing, top-k6, 1,992 padded
  expert rows, W13 split-K2.
- Input contract: caller-provided FP8-E4M3 activation plus FP32 group-128
  activation scales and MXFP4 weights; input quantization is outside both
  timed graphs.
- Timing: replay-paired CUDA Graphs, two batches x eight samples after two
  warm-up replays.  A separate 256 MiB Triton cache-clear kernel ran directly
  before every implementation replay and was excluded from event timing.
- Control: the selected multi-kernel path from the same source and process.
- Driver note: the tracked `bench/v4_flash_tp_single_vs_multi_graph.py` had
  drifted ahead of its tracked `CapturedCase` constructor and failed before
  graph capture with an unexpected native-only argument.  The successful run
  used `bench_stage_upload/v4_flash_tp_single_vs_multi_graph.py`, whose
  constructor exactly matches the current graph helper.  No kernel or timing
  behavior differs for the TP candidate/control paths.

## Result

| Path | min (ms) | median (ms) | max (ms) |
|---|---:|---:|---:|
| Multi-kernel control | 0.302144 | 0.305280 | 0.331968 |
| 160-thread producer candidate | 0.378080 | 0.381872 | 0.396608 |

The candidate/control median ratio is `1.250891`: the candidate is **25.09%
slower**.  The adjacent exact-production single-kernel anchor was 0.349616 ms
against 0.305008 ms (`1.146252`), so this transfer also regresses the existing
single kernel by about 9.23% and widens the multi-kernel gap by about 10.46
percentage points.

## Decision

Reject the dedicated producer-warp transfer and restore the selected source
byte-for-byte.  The multi-kernel advantage is not explained by merely adding
its producer/consumer warp split inside the persistent CTA: the extra warp,
extra empty-barrier handshakes, lower CTA residency, and larger stack frame
cost substantially more than any TMA/WGMMA overlap they recover.

Do not transfer multi-kernel mechanisms that require another permanent warp or
cross-phase asynchronous state.  Continue the audit with transfers that keep
the selected 128-thread/9-CTA residency contract, especially instruction-level
W13/W2 scheduling, compact parameter passing, and phase-local address/setup
hoisting.

Raw logs:

- `bench/results/iter685_lean_tma_producer_tp4_m128_cold_screen_20260906.log`
  (invalid harness-interface attempt; no graph/timing result)
- `bench/results/iter685b_lean_tma_producer_tp4_m128_cold_screen_20260906.log`
  (valid TP4 correctness and cold-L2 timing)
