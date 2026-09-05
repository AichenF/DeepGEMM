# Iteration 651b: standalone W13 M128 cold-L2 NCU collection

## Protocol

- Host/container: H20-GPU-06 / `dpskv4_h20_weekly_gap_20260727`
- Physical GPU: 1 (H20, 78 SM)
- Source branch/head before collection: `tpmegamoe_single_launch` / `ea2759b`
- Path: current same-source multi-kernel local pipeline, TP4 M128,
  split-K auto (M128 resolves to split-K2), random routes, seed 20260902
- Input: prequantized FP8-E4M3 activation plus FP32 group-128 scale and MXFP4
  weights/scales
- Cache policy: application-managed excluded 256 MiB L2 clear immediately
  before the local pipeline; NCU cache control disabled
- NCU filter: short demangled name `regex:route_gemm`, launch count one.  The
  first matching launch inside the profiler-delimited pipeline is W13; the
  later W2 launch is not collected.
- Sections: LaunchStats, Occupancy, SpeedOfLight, SchedulerStats,
  WarpStateStats, SourceCounters; 21 kernel-replay passes

## Collection result

NCU profiled exactly one `route_gemm` launch, completed 21 passes, and wrote:

```text
results/iter651b_standalone_w13_m128_cold_sourcecounters.ncu-rep
```

The helper confirms M128/TP4/random routing, the 256 MiB cold-L2 clear,
automatic split policy, Mode-2 braid, W13 S2R prefetch, two weight stages,
compact scale storage, and caller-provided prequantized input.

This profiling helper does not emit a tensor comparison, so this iteration
claims successful collection but no new correctness result.  The kernel body
and input construction are unchanged from the independently bitwise-qualified
same-source multi path used in Iterations 646/650b.

## Status

Parse the saved report together with Iteration 650b.  Compare only NCU metrics
and PC-sample composition; use the separately timed cold-L2 medians from
Iterations 646/647 for latency because NCU replay duration is diagnostic.
No CUDA source behavior changed.
