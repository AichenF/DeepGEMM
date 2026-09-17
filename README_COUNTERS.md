# SM90 NVFP4 fused MegaMoE: counter-based synchronisation (8 x H20-3e)

Branch `megamoe_nvfp4_counters` (from `megamoe_nvfp4_dev` @ e038aa5). The
production H20 small-M kernel (`sm90_nvfp4_mega_moe_fused_body.inl`, dev-m
dynamic scheduler + RS mode5) gets three counter-based replacements of its
NVLink barriers, each behind a knob and an M-gated policy:

| Step | Knob (host env, read per call) | Replaces | Enabled by the policy |
|---|---|---|---|
| 1 push dispatch + DONE flags | `DG_NVFP4_PUSH_DISPATCH` | barrier #1 + count broadcast + TMA pull loop | yes (see policy) |
| 3 parity-rotated pool | `DG_NVFP4_NO_CLEAN_BARRIER` (needs step 1) | barrier #3 (workspace cleanup) | yes (see policy) |
| 2 fine-grained combine | `DG_NVFP4_FINE_COMBINE` | barrier #2 | no |

Push dispatch: during routing the source rank takes one remote `atom.sys`
ticket per routed row on the destination's per-expert count (lane-parallel
per warp), copies the FP8 row (TMA bulk copy over NVLink), its per-K128
scales, the top-k weight and the source metadata straight into the
destination's pool (fixed stride of pool blocks per local expert; the dynamic
scheduler keeps dense task indices and only remaps the physical pool block),
then the last CTA `red.release.sys`-adds 1 to every rank's DONE count. The task
producer and the arrival-count publisher wait for `ranks * (epoch + 1)` DONE
signals instead of barrier #1. Rotated pool: the strided pool and the count
words are double-buffered by launch parity so barrier #3 is dropped (a rank can
only push launch N+2 rows after every rank signalled DONE for N+1, i.e. after
every rank completed N including its cleanup).

Fine-grained combine is implemented (mailbox from the epilogue to dispatch warp
0, per-token `red.release.sys` arrival counters, per-launch combine ticket) and
bit-identical, but it is not enabled: merely compiling its code into the
kernel slows every math task by ~10 % (Flash M8: L1 task 23.7 -> 26.0 us, L2
10.9 -> 12.6 us, measured with the code compiled in but never executed), which
outweighs the ~8 us it removes from the combine tail. The same effect hit the
push code twice during development (a device `printf` in a spin loop and a
`__noinline__` helper each cost 15-19 % per math task through the kernel-wide
register allocation); the shipped code has neither.

## Policy (`csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe.hpp`, `kSM90NVFP4H20CounterBuckets`)

M = tokens per rank. A step is on where it won >= 1 % on the random router and
did not lose more than 1 % on the all-experts router.

| Model | M | Policy |
|---|---:|---|
| Flash (E256 H4096 I2048 top-6) | 1-16 | push |
| Flash | 17-64 | push + rotated pool |
| Flash | >= 65 | barrier path (OFF) |
| Pro (E384 H7168 I3072 top-6) | 1-16 | push + rotated pool |
| Pro | >= 17 | OFF |
| MiMo (E384 H6144 I2048 top-8) | 1-16 | push + rotated pool |
| MiMo | 17-32 | push |
| MiMo | >= 33 | OFF |

H200 is not measured and keeps the barrier path. The pool stride holds
`num_max_pool_tokens / (experts_per_rank * slots)` rows per expert (8448 rows at
capacity 8448, 4224 with the rotated pool); the host falls back to the pull
path when `M * ranks` exceeds it.

## Results

Microseconds, fused family, W4A8, capacity 8448, max over 8 ranks, median of 50 calls (25 A/B/B/A blocks, same process, same build 39f4468). `pristine PR#8` = the AichenF/DeepGEMM PR #8 table (their node viking-prod-583, their "balanced" router; their in-repo bench script reproduces 262.7 us for Flash M8 on this node with the random router). `baseline` = e038aa5 barrier path (OFF arm; equals the unmodified e038aa5 build within noise: 273.0 / 877.2 us at Flash / Pro M1 all-experts). `counters` = the policy default of this branch.

| Model | M | router | pristine PR#8 | baseline (OFF) | counters (policy) | delta | policy arm |
|---|---:|---|---:|---:|---:|---:|---|
| Flash | 1 | random | - | 124.2 | 114.9 | +7.5% | push |
| Flash | 1 | balanced | - | 279.3 | 274.3 | +1.8% | push |
| Flash | 2 | random | - | 155.1 | 144.9 | +6.6% | push |
| Flash | 2 | balanced | - | 277.9 | 271.4 | +2.3% | push |
| Flash | 4 | random | - | 218.2 | 210.7 | +3.4% | push |
| Flash | 4 | balanced | - | 280.1 | 272.9 | +2.6% | push |
| Flash | 8 | random | 263.0 | 258.9 | 250.3 | +3.3% | push |
| Flash | 8 | balanced | 263.0 | 282.7 | 277.1 | +2.0% | push |
| Flash | 16 | random | - | 284.3 | 277.2 | +2.5% | push |
| Flash | 16 | balanced | - | 285.6 | 278.3 | +2.6% | push |
| Flash | 32 | random | - | 306.4 | 290.4 | +5.2% | push+rotated |
| Flash | 32 | balanced | - | 311.8 | 290.2 | +6.9% | push+rotated |
| Flash | 64 | random | 295.1 | 314.7 | 309.4 | +1.7% | push+rotated |
| Flash | 64 | balanced | 295.1 | 317.1 | 311.7 | +1.7% | push+rotated |
| Flash | 128 | random | 320.7 | 462.0 | 459.5 | +0.5% | OFF |
| Flash | 128 | balanced | 320.7 | 334.6 | 332.5 | +0.6% | OFF |
| Flash | 256 | random | 447.9 | 490.5 | 487.9 | +0.5% | OFF |
| Flash | 256 | balanced | 447.9 | 475.3 | 472.3 | +0.6% | OFF |
| Pro | 1 | random | - | 221.2 | 202.0 | +8.7% | push+rotated |
| Pro | 1 | balanced | - | 885.1 | 866.0 | +2.2% | push+rotated |
| Pro | 2 | random | - | 300.0 | 282.9 | +5.7% | push+rotated |
| Pro | 2 | balanced | - | 877.6 | 858.6 | +2.2% | push+rotated |
| Pro | 4 | random | - | 464.9 | 445.4 | +4.2% | push+rotated |
| Pro | 4 | balanced | - | 880.6 | 860.2 | +2.3% | push+rotated |
| Pro | 8 | random | 965.3 | 679.2 | 657.0 | +3.3% | push+rotated |
| Pro | 8 | balanced | 965.3 | 883.8 | 862.1 | +2.5% | push+rotated |
| Pro | 16 | random | - | 847.6 | 823.3 | +2.9% | push+rotated |
| Pro | 16 | balanced | - | 893.4 | 872.0 | +2.4% | push+rotated |
| Pro | 32 | random | - | 887.2 | 885.7 | +0.2% | OFF |
| Pro | 32 | balanced | - | 902.9 | 901.3 | +0.2% | OFF |
| Pro | 64 | random | 999.1 | 942.6 | 939.2 | +0.4% | OFF |
| Pro | 64 | balanced | 999.1 | 917.3 | 913.1 | +0.5% | OFF |
| Pro | 128 | random | 1096.1 | 1058.9 | 1058.4 | +0.0% | OFF |
| Pro | 128 | balanced | 1096.1 | 1007.5 | 1005.8 | +0.2% | OFF |
| Pro | 256 | random | 1593.3 | 1578.0 | 1573.2 | +0.3% | OFF |
| Pro | 256 | balanced | 1593.3 | 1573.0 | 1571.8 | +0.1% | OFF |
| MiMo | 1 | random | - | 194.0 | 177.7 | +8.4% | push+rotated |
| MiMo | 1 | balanced | - | 534.1 | 515.5 | +3.5% | push+rotated |
| MiMo | 2 | random | - | 231.6 | 215.2 | +7.1% | push+rotated |
| MiMo | 2 | balanced | - | 537.5 | 520.1 | +3.2% | push+rotated |
| MiMo | 4 | random | - | 354.8 | 337.0 | +5.0% | push+rotated |
| MiMo | 4 | balanced | - | 538.9 | 519.9 | +3.5% | push+rotated |
| MiMo | 8 | random | 571.6 | 486.6 | 467.0 | +4.0% | push+rotated |
| MiMo | 8 | balanced | 571.6 | 541.5 | 522.3 | +3.5% | push+rotated |
| MiMo | 16 | random | - | 537.8 | 517.1 | +3.8% | push+rotated |
| MiMo | 16 | balanced | - | 544.1 | 524.9 | +3.5% | push+rotated |
| MiMo | 32 | random | - | 559.0 | 548.5 | +1.9% | push |
| MiMo | 32 | balanced | - | 560.7 | 552.4 | +1.5% | push |
| MiMo | 64 | random | 638.5 | 598.0 | 595.4 | +0.4% | OFF |
| MiMo | 64 | balanced | 638.5 | 614.0 | 610.7 | +0.5% | OFF |
| MiMo | 128 | random | 687.9 | 801.9 | 800.0 | +0.2% | OFF |
| MiMo | 128 | balanced | 687.9 | 661.5 | 660.8 | +0.1% | OFF |
| MiMo | 256 | random | 951.0 | 961.8 | 957.1 | +0.5% | OFF |
| MiMo | 256 | balanced | 951.0 | 963.8 | 960.7 | +0.3% | OFF |

Step-2 (fine-grained combine) A/B, Flash, random router, same protocol: M1 118.4 -> 116.4 us (+1.6 %), M2 151.6 -> 152.5 (-0.6 %), M8 255.5 -> 267.1 (-4.6 %) versus the barrier path, i.e. worse than push alone (109.6 / 143.2 / 247.8); not enabled.

Phase stamps (random router, max over ranks, us from kernel entry, barrier path -> push): pool ready Flash M1 31.0 -> 14.5, Flash M8 29.7 -> 15.4, Pro M1 34.9 -> 14.3, Pro M8 36.6 -> 17.6; per-task durations unchanged (Pro L1 task 40.2 vs 39.6 us, L2 16.0 vs 16.4). Above the policy boundary the kernel is weight-stream bound from its first task (Pro all-experts: 1.98 GB of NVFP4 weights per rank in ~830 us = 2.4 TB/s; the weight stream starts at the count broadcast (~16 us) in both protocols), so an earlier pool has no effect on the kernel end while the ticket + row pushes (up to 20 serial rows per warp at M = 256) cost 1-4 %.

## Measurement method

Host 10.6.131.8 (`H20-GPU-08`), 8 x H20-3e, SM clock 1830 MHz (verified in
kernel: 1.82 GHz over the L1 tasks), container `fe5c_build` (torch
2.11.0+cu130, CUDA 13.0), one build of this branch per table, W4A8 (NVFP4
weights, FP8 activations), capacity 8448 tokens per rank, top-k weights and
random inputs seeded per rank.

`tests/ab_nvfp4_counters.py --mode bench`: profile-off CUDA events around the
kernel on every rank, `dist.barrier()` and an 8 GB L2 flush before every call,
25 same-process A/B/B/A blocks = 50 calls per arm, per call the maximum latency
over the 8 ranks, median over the 50 calls. Arms alternate inside one process
by setting the knobs above (the JIT name carries the knob set). Routers:
`random` = their bench script's router (`torch.manual_seed(rank + 101)`,
random scores, top-k; reproduces their PR #8 Flash M8 figure, 262.7 us, with
their own script on this node); `balanced` = every expert receives the same
number of rows (row `(rank*M + t)*topk + s mod E`), i.e. every rank streams
all of its experts' weights.

Phase stamps (`--mode stamps`, `DG_NVFP4_PHASE_STAMPS_PTR`, globaltimer, compiled
in only for that run) give per-rank routing / DONE / pool-ready / first math /
last L1 / last L2 / combine times and per-task durations; they are not used
for any number in the results table.

## Correctness

`tests/ab_nvfp4_counters.py --mode correctness`: for Flash, Pro and MiMo at
M = 1, 2, 8, 16, 64, 256 (random and all-experts routers, two seeds at M <= 8)
the output `y` of every enabled arm (push, push + rotated pool, and the
POLICY default) is bit-identical to the barrier path of the same build and to
the unmodified e038aa5 build (dumped from a separate process), all values
finite, and a 100-200 call alternating stress loop across all arms stays
bit-identical. Their single-rank correctness gate
(`tests/test_nvfp4_mega_moe_sm90_correctness.py`) cannot run these shapes: the
fused heuristic requires 8 ranks.

Reproduce (inside the container, repo root):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/ab_nvfp4_counters.py --mode correctness --shapes flash pro mimo --m 1 8 32 64 --arms OFF POLICY --routers random balanced --stress 100
python tests/ab_nvfp4_counters.py --mode bench --shapes flash --m 1 2 4 8 16 32 64 128 256 --arms OFF POLICY --routers random balanced --blocks 25
```
