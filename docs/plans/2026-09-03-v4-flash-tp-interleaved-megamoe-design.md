# DeepSeek-V4-Flash TP-local interleaved MXFP4 MegaMoE design

Date: 2026-09-03

## Goal and evidence

The required verdict is at least `1.20x` lower full TP4 latency than the exact Humming MXFP4 plus SGLang `CustomAllReduceV2` baseline, using equal-weight geometric mean over `M={8,16,32,64,128}`. Every replay is independently cold-L2: a separate 256 MiB clear immediately precedes the graph replay and is excluded from CUDA-event timing.

The selected separate-kernel implementation currently reaches `1.136755x`. Iteration 73 establishes a bitwise-equivalent multicast fused-k6/one-shot-push win of `1.0067x–1.0103x` at M8–M32. Long-window M128 NVLS-pull experiments are neutral. Iteration 74 shows W13 plus W2 consume 81–93% of instrumented custom local time and that active-expert cold weight traffic dominates balanced routes. Closing the remaining gap therefore requires a W13→W2 scheduling change, not another all-reduce launch-geometry tweak.

## Considered approaches

1. **TP-local interleaved persistent MegaMoE (selected).** Adapt DeepGEMM's existing SM90 interleaved scheduler and producer/math pipeline. Keep route preparation local, publish W13 intermediate tiles as soon as they are ready, and allow W2 tasks to consume completed expert blocks before the entire W13 grid drains. This is the only candidate with a plausible multi-microsecond gain while retaining dynamic routes.
2. **Move top-k weights into W2 activation scales.** This removes multiplications from the local k6 tail but changes the BF16 rounding boundary and can only save roughly a microsecond. Rejected by design.
3. **Shrink launches from the captured route distribution.** Host-specializing grid size to one fixed route can remove empty CTAs but does not represent serving, where route IDs change between graph replays. Rejected as benchmark specialization.

## Public data and numerical contract

Every TP rank receives replicated BF16 `X[M,4096]`, INT32 `topk_idx[M,6]`, and FP32 `topk_weights[M,6]`. TP4 owns W13 `[256,1024,4096]` and W2 `[256,4096,512]`; TP8 owns `[256,512,4096]` and `[256,4096,256]`.

The new path must preserve these boundaries:

1. dynamic BF16→FP8-E4M3 group-128 quantization of X;
2. MXFP4 W13 accumulation followed by BF16 output semantics;
3. SwiGLU followed by BF16 output semantics;
4. dynamic FP8-E4M3 group-128 requantization;
5. MXFP4 W2 followed by BF16 route rows;
6. ordered FP32 k=6 accumulation using `topk_weight * 1.5`, then BF16 local output;
7. one TP sum all-reduce.

The implementation may change WGMMA reduction order and physical weight layout, but it must pass the established dequantized-MXFP4 reference gates. It must not move top-k weighting across the W2 BF16 boundary.

## Architecture

### Preparation

Retain the selected `fused_route_quant` graph node. Besides the current sorted route IDs, expert-block IDs, padded count, quantized X and scales, its CTA 0 resets graph-stable scheduler counters and per-block readiness words. Route alignment remains device-resident and recomputes on every replay.

Blocks remain sorted by expert with block-M initially fixed at eight. This gives contiguous expert weight access without requiring per-replay host synchronization. Later block-M tuning may use the known graph shape M, but never observed route IDs.

### Persistent compute kernel

Add an opt-in TP4 kernel derived from the read-only DeepGEMM SM90 MegaMoE implementation. Launch exactly 78 persistent CTAs so device-wide progress cannot depend on oversubscribed blocks. Each CTA contains the existing producer/consumer organization and a two-stage task mailbox; tasks are claimed from device counters rather than derived from a fixed host grid.

The scheduler operates on local padded route blocks and has two task classes:

- W13 task: one expert block and one paired gate/up intermediate tile;
- W2 task: one ready expert block and one output-hidden tile.

W13 weights are preprocessed outside timing so each physical tile pairs gate and up channels. A completed full-K W13 task can therefore apply the BF16 boundary, SwiGLU, the second BF16 boundary and group-128 FP8 quantization locally. It writes the route-block activation tile and scale, then release-publishes one readiness bit. No separate full-grid activation kernel is required.

A W2 task becomes claimable only after all intermediate tiles required by that route block are ready. It acquires the readiness word, consumes the quantized activation and MXFP4 W2 weights, and writes BF16 route output. W13 and W2 tasks are interleaved by expert waves so early W2 work overlaps the long W13 tail without fragmenting either GEMM into graph nodes.

The initial TP4 policy uses small expert waves and the native two-stage task mailbox. Block-M, block-N and wave size remain compile-time candidates selected only by M, never by route contents. TP8 initially dispatches to the already verified separate-kernel path; a TP8 persistent specialization is allowed only after TP4 shows a material win.

### Tail and communication

For TP4 M8/M16/M32, use the iteration-73 multicast fused-k6/one-shot-push tail, which is bitwise equal to stock output. For M64/M128, keep the selected SGLang local reducer and stock CARv2 two-shot pull. TP8 keeps stock CARv2.

No EP dispatch, combine, symmetric expert pool or EP barrier is introduced. Symmetric memory is used only by the final SGLang-compatible TP collective.

## Progress and deadlock invariants

- Scheduler counters and readiness words are reset before every persistent launch inside the captured graph.
- The persistent grid is at most one CTA per physical H20 SM; no global progress wait may require an unlaunched CTA.
- Readiness publication uses release semantics after activation/scale stores; W2 observes it with acquire semantics before loading.
- Every task is uniquely claimed and every route/output element has one writer before the local k6 reduction.
- Invalid padded routes produce zero/ignored activation and never index X or weights out of bounds.
- A graph can replay after X, route IDs and route weights are replaced in-place; no host-derived active-expert count is part of correctness.
- The opt-in flag falls back to the selected implementation for unsupported TP/shape combinations.

## Implementation sequence

1. Copy only the necessary scheduler/task-mailbox concepts into owned TP files and add an opt-in compile path; do not edit the dirty DeepGEMM checkout.
2. Build a scheduler-only diagnostic that enumerates W13/W2 task ownership and readiness for balanced, random and skew routes. Compare device results with a CPU oracle and replay with changed routes.
3. Integrate the current MXFP4 W13 core and fused BF16/SwiGLU/requant epilogue. Validate the intermediate activation before enabling W2 interleave.
4. Integrate W2 and BF16 route output, then validate the complete local tensor against the current selected path and the dequantized-MXFP4 reference.
5. Run graph-internal cold-L2 screens at M8/M32/M128. Reject the persistent path if it cannot show a material core reduction; otherwise tune block-N, expert-wave size and task warmup.
6. Enable the confirmed multicast tail at M8–M32 and run same-process custom/control A/B.
7. Run the formal paired TP4 10×200 five-point cold-L2 verdict and a route-mutation correctness audit. Only claim success at geometric-mean speedup `>=1.20x` with all point results reported.
8. Run TP8 correctness and end-to-end graph smoke, using fallback unless a TP8 specialization has independently passed.

## Verification artifacts

Each source modification followed by a benchmark is one AKO iteration: record correctness, exact cold-L2 protocol, per-M min/median/max, acceptance decision and log path in `ITERATIONS.md`, then commit. Failed compiles, hangs and regressions are also logged. Formal evidence remains under `bench/results/`; unrelated dirty files and the main DeepGEMM checkout are untouched.
