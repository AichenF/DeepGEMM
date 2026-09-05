# Iteration 650b: production one-kernel M128 cold-L2 NCU collection

## Protocol

- Host/container: H20-GPU-06 / `dpskv4_h20_weekly_gap_20260727`
- Physical GPU: 1 (H20, 78 SM)
- Source branch/head before collection: `tpmegamoe_single_launch` / `ee00da6`
- Configuration: selected TP4 one-kernel defaults, M128, split-K2, random
  routes, seed 20260902, device phase stamps enabled
- Input: caller-provided FP8-E4M3 activation plus FP32 group-128 scale and
  MXFP4 weights/scales
- Collective: disabled only for local compute attribution
- Cache policy: application performs a separate excluded 256 MiB L2 clear
  immediately before the profiled launch; NCU cache control is disabled so it
  does not replace that protocol
- NCU: 2025.3.1, kernel replay, 21 passes, LaunchStats, Occupancy,
  SpeedOfLight, SchedulerStats, WarpStateStats, and SourceCounters

## Collection result

The corrected `/usr/bin/python3` target command completed all 21 replay
passes and wrote:

```text
results/iter650b_production_m128_cold_sourcecounters.ncu-rep
```

The complete routed W2 tensor is bitwise equal to the independent same-source
multi-kernel reference: cosine 1.0, relative L2 0.0, finite true.  The route
population has 1,992 padded rows and all four packed barrier generations end
at 2048.

Device-stamped phases for the collected replay are:

| phase | time (us) |
|---|---:|
| route | 3.456 |
| W13 | 223.328 |
| activation/requant | 6.528 |
| W2 | 109.600 |

NCU replay duration and phase stamps are diagnostic only and are not used as
formal endpoint latency.  Detailed metric/PC interpretation is intentionally
deferred to a separate audit of the saved report.

## Status

Collection and correctness pass.  Parse the report next, compare W13-region
barrier/scoreboard/issue samples with standalone W13, and use that evidence to
select or reject the next structural design.  No CUDA source behavior changed
in this iteration.
