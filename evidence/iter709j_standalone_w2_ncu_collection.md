# Iteration 709j — standalone W2 cold-L2 NCU collection

## Scope and protocol

- Host/container: H20-GPU-06 / `megamoe`, physical H20 GPU 1.
- Shape: DeepSeek-V4-Flash TP4 local W2, M=128, H=4096,
  I/rank=512, random top-k6 routes, seed 20260902.
- Input ABI: caller-provided group-128 E4M3 activation/scales and canonical
  packed MXFP4 weights.  Profiling fixture construction is outside the range.
- Cold-L2: 256 MiB `tensor_zero` stream write immediately before the local
  W13/requant/W2 pipeline; device L2 is 60 MiB.  NCU uses
  `--cache-control none`, so the application-owned clear is the cache policy.
- Target selection: filter `^route_gemm$`, skip the first matching launch
  (standalone W13), collect one matching launch (standalone W2), 21 replay
  passes.  The clear kernel is excluded by the kernel-name filter.
- Report:
  `results/iter709j_standalone_w2_m128_cold_sourcecounters.ncu-rep`.

## Result

- Duration: 97.70 us (diagnostic NCU replay value, not endpoint timing).
- Grid/block: 10272 CTAs x 128 threads.
- Registers/thread: 61; dynamic/static shared memory: 18.43/1.02 KiB.
- Theoretical/achieved occupancy: 50.00% / 48.18%; 16.46 waves/SM.
- DRAM/L2/SM throughput: 57.61% / 72.76% / 74.08%.
- Issue rate: 0.76 instructions/cycle; no-eligible: 24.02%; active and
  eligible warps/scheduler: 7.71 and 2.44.
- QGMMA execution count: 1,019,904.
- Source-counter aggregate: 39,014,432 executed instructions and 6,263 PC
  samples.  Principal sampled stalls are barrier 1,149, long scoreboard 578,
  wait 714, math-pipe throttle 744, not-selected 1,350, short scoreboard 332.
- Branch efficiency: 68.67%.  Excess global sectors: 163,584 / 973,632
  requested sectors (about 16.8%).

## Interpretation and next action

The saved fused M128 report contains the same 1,019,904 W2 QGMMA executions,
so the fused path is not winning or losing through a different MMA count.
This collection supplies the missing same-shape standalone W2 control.  The
next action is to isolate the fused W2 SASS interval, compare useful
instruction and stall distributions with this standalone control, and only
then select a non-duplicated structural experiment.

No endpoint latency or correctness claim is made from the NCU duration.
