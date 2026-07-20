# SM90 NVFP4 Unified Small-M Kernel Design

## Goal

Replace the separate Flash, Pro, and MiMo small-M implementations with one
production SM90 NVFP4 fused kernel template. Shape families share the weight
ABI, numerical path, scheduler, communication protocol, and kernel body. The
selector may produce different compile-time tuning configurations from
workload properties.

The initial small-M validation set is `M={8,16,32,64,128}` on H20 and H200.
The existing generic BN128 split path remains the large-M fallback until it is
independently migrated and benchmarked.

## Common Contract

- Weight storage is Mode2 Braided BN256/BK128.
- Each BK128 row remains 80 bytes: 64 packed E2M1 bytes, eight per-K16 UE4M3
  scale bytes, and eight padding bytes.
- L1 gate/up rows are interleaved during weight-load-time prepack.
- L1 and L2 execute in one fused kernel.
- Packed FP4 values are decoded through one shared Mode2 implementation.
- Intermediate FP8 activation scaling is per 128 channels.
- Dispatch, expert scheduling, barriers, publication, combine, and cleanup use
  one cross-rank protocol.
- The public entry point and weight transform remain singular.

## Kernel Structure

The common JIT kernel receives model dimensions and routing topology as shape
parameters. It receives a small compile-time tuning policy containing only:

- block M;
- pipeline stage count;
- experts per wave;
- dispatch warp participation;
- swap-AB selection; and
- row versus LUT-window scheduling for the same Mode2 decode operation.

The kernel body must not branch on model names or recognize exact Flash, Pro,
or MiMo tuples. Shape-specific expressions used only to select a measured
tuning winner move into the host selector and are replaced by routed-load and
derived-wave ranges.

## Selector

The selector derives expected local expert load as
`num_tokens * num_topk / num_experts_per_rank`. Continuous load ranges select
block M and the decode/dispatch schedule; the expert wave is rounded to a
divisor of the local expert count, and pipeline depth is limited by the
derived shared-memory footprint. No isolated equality point or exact model,
expert-count, token-count, or GPU fingerprint is permitted.

As in the SM90 FP8 selector, the implementation separates heuristic input,
normalized routed load, schedule tuning, configuration materialization, and
legality validation. The final selector only orchestrates those layers. Tile
sizes may be discrete compile-time values; the request M is never matched to
an isolated value.

The physical SM count controls only the local launch grid, work partitioning,
and grid synchronization. It does not select a tuning policy. Tuning fields do
not change the NVLink barrier sequence, tags, or targets. Deployment-wide
shape and topology fields that define the symmetric-buffer and cross-rank
protocol remain common to all ranks; observed receive counts remain scheduler
inputs and never select a protocol-changing configuration.

## Migration

1. Establish the current `megamoe_nvfp4` correctness and cold-L2 performance
   baseline for Flash, Pro, and MiMo.
2. Extract the retained MiMo Mode2 decoder and fused-body improvements into a
   generically named common implementation while preserving its generated
   configuration.
3. Change BN256 prepack to the Mode2 Braided ABI and invalidate old cached
   BN256 layouts. Keep BN128 standard weights for the unchanged split path.
4. Route all three small-M shape families through the common body using
   conservative correct configurations.
5. Tune only the policy fields through AKO. Retain a policy after cold-L2 ABBA
   results show no material regression across its full load range.
6. Remove superseded candidate APIs and duplicate bodies after the complete
   correctness and performance matrix passes.

## Verification

- Build and fresh-JIT compilation for every retained configuration.
- Exact NVFP4 correctness at `M={8,16,32,64,128}` for Flash, Pro, and MiMo.
- Normal and adversarial routing, with absent and per-expert global scales.
- H20 and H200 cold-L2 benchmarks using fixed seeds, 50 samples for small M,
  and median max-rank latency as the primary metric.
- Guard benchmarks at and beyond the small/large boundary to prove the BN128
  split path is unchanged.
- Generated-code comparison for the initial MiMo extraction before accepting
  any tuning change.
