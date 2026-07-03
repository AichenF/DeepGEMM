# AKO Iterations: SM90 FP8 MegaMoE H200 Retuning

## Objective and fixed references

- Optimization branch: `opt/megamoe-sm90-fp8-h200-retune`.
- Current FP8 baseline: `3552b62545e3602d60bde6ed3542934f6dcf6232`.
- PR323 upstream head: `8ddf7f96cb3300011f69458e88c7651a1e305a8c`.
- PR323 CUDA 13.2 syntax-only fix: `184d74cb052a028ac9d960d65abd35ec231146df`.
- Hardware evidence accepted for new decisions: 8x NVIDIA H200 only.
- Architecture constraint: retain the existing split L1/L2 FP8 kernels; no
  PR323 fused-kernel implementation.
- Success gate: every Flash/Pro M >= 128 point faster than PR323; M < 128 no
  confirmed regression from the current FP8 baseline.

## Prior evidence motivating iteration 1

- The previous three-way H200 matrix showed Pro gaps versus PR323 of 1.26%,
  1.18%, 11.88%, 19.16%, 19.99%, 27.79%, and 21.81% at M 128 through 8192.
- The Pro selector enables direct L2 scatter when expected tokens per expert
  reach 64, which is exactly M=512 for top-k 6 and 48 local experts.
- The direct path emits 32-bit remote stores, while the non-direct path packs
  128-bit stores. This is the first H200-only parameter attribution to test.
- At Pro M >= 2048, ours and PR323 already use the same main tile, stage count,
  thread layout, SM count, and experts per wave. Therefore the first search
  focuses on existing epilogue/scheduler modes rather than only BM/BN.

## Iteration record template

Each measured source or promoted-selector iteration records:

1. hypothesis and exact source/config change;
2. hardware, source commits, command, routes, and cache identity;
3. correctness result;
4. rank-zero and max-rank latency versus current best and PR323;
5. decision: retain, reject, or gather more repetitions;
6. raw artifact paths and commit hash.

## Baseline 0: pinned split FP8 versus PR323 on H200

- Hypothesis: reproduce the pinned comparison on a fresh H200 allocation before
  changing any selector or kernel parameter.
- Hardware: Slurm job `2957858`, node `viking-prod-299`, 8x NVIDIA H200
  (143771 MiB), driver 595.58.03, CUDA 13.2.78, PyTorch 2.12.1+cu132.
- Sources: ours `3552b62545e3602d60bde6ed3542934f6dcf6232`; PR323
  `8ddf7f96cb3300011f69458e88c7651a1e305a8c` plus syntax-only CUDA 13.2
  fix `184d74cb052a028ac9d960d65abd35ec231146df`.
- Correctness: the existing split-FP8 L1-L4 suite passed 33/33 scenarios with
  maximum `calc_diff=0.0006` against tolerance 0.01. Coverage included masked
  and all-masked routes, activation clamp variants, both fast-math modes, and
  zero/max token boundaries.
- Performance protocol: Flash and Pro at M=8/16/32/64/128/256/512/1024/2048/
  4096/8192, seed 101, median of 10 timed samples, alternating ours/PR323
  process order, capacity 8192, no L2 flush. The split implementation emitted
  exactly two matched events per call and the harness summed L1+L2; PR323
  emitted one matched event per call. All eight rank records were retained.
- Result status: the driver completed all 44 leaf runs with `RUN_EXIT=0`.
  Representative Pro max-rank observations retained the historical trend:
  ours/PR323 were about 1186/1091 us at M=512, 3005/2463 us at M=2048, and
  10047/7843 us at M=8192. The strict route/rank parser produces the complete
  table after this audit commit.
- Decision: retain this as the H200 baseline. Begin the H200-only parameter
  search without modifying the existing H20/H100/generic selector paths.
- Raw artifacts:
  `/home/scratch.aichenf_wwfo/greencontext/results/sm90_fp8_h200_retune_job2957858/`
  (`environment.txt`, `logs/baseline_correctness_l1_l4.log`, 44 baseline leaf
  logs, and `logs/baseline_driver.log`).

## Parameter screen 1: disable Pro direct L2 scatter

- Hypothesis: the direct path's scalar remote stores are responsible for the
  sharp Pro regression beginning at M=512; forcing the existing non-direct
  vector/TMA path may close the PR323 gap without changing architecture.
- Exact configuration: `DG_SM90_MOE_DIRECT_L2_SCATTER=0`, all other selector
  controls at their defaults; Pro M=512/1024/2048/4096/8192, seed 101,
  median-10, 8x H200. No source or production-selector change was made.
- Results (max-rank):

  | M | baseline us | direct-off us | vs baseline | PR323 us | vs PR323 |
  |---:|---:|---:|---:|---:|---:|
  | 512 | 1185.491 | 1191.638 | +0.52% | 1090.546 | +9.27% |
  | 1024 | 1892.598 | 1858.309 | -1.81% | 1582.323 | +17.44% |
  | 2048 | 3004.873 | 2938.806 | -2.20% | 2463.095 | +19.31% |
  | 4096 | 5409.617 | 4956.729 | -8.37% | 4338.506 | +14.25% |
  | 8192 | 10046.682 | 8654.320 | -13.86% | 7842.460 | +10.35% |

- Decision: reject direct-off as a complete Pro large-M rule. Retain it as the
  parent for the M>=4096 beam, where it is a large repeatable-looking
  improvement, and combine it with H200-only stage/wave/block experiments.
  Keep the baseline direct path at M=512.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_direct0/`.

## Parameter screen 2: Pro representative-point beam expansion

- Hypothesis: combine the useful direct-off parent with one existing scheduler,
  pipeline, wave, or tile-axis change; retain a direct-on branch for M=512.
- Protocol: Pro M=512/4096/8192, seed 101, median-10, 8x H200, one axis per
  candidate, isolated JIT caches. Ten candidates and 30 points completed with
  eight rank records each.
- Best max-rank results by representative point:

  | M | best candidate | candidate us | vs baseline | PR323 us | vs PR323 |
  |---:|---|---:|---:|---:|---:|
  | 512 | direct0 + stage3 | 1132.499 | -4.47% | 1090.546 | +3.85% |
  | 4096 | direct0 + stage3 | 4500.249 | -16.81% | 4338.506 | +3.73% |
  | 8192 | direct0 + N-major | 8432.094 | -16.07% | 7842.460 | +7.52% |

- Other useful signals: direct0+EPW16 reached 4822.348 us at M=4096;
  direct0+EPW24 reached 8598.828 us at M=8192; direct-on+stage4 improved the
  baseline by 1.40%, 4.63%, and 7.54% but remained behind the direct-off beam.
- Rejected axes: BN128 and BM128 regressed all three points by 3.68% to
  17.55% versus baseline. Direct-on+EPW24 also failed to improve M=512/4096.
- Decision: retain `direct0+stage3` and `direct0+N-major` as beam parents.
  Test their combination and neighboring EPW/cleanup combinations, then fill
  M=1024/2048 only for survivors. No production selector change yet.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_d0_*`
  and `.../candidates/pro_d1_*`.

## Parameter screen 3: Pro combined beam

- Hypothesis: the direct-off stage3 and N-major gains are at least partly
  additive; wave count or cleanup may recover the remaining representative-
  point gaps.
- Protocol: Pro M=512/4096/8192, seed 101, median-10, 8x H200. Eleven
  combinations of stage2/3, N-major, EPW16/24, and cleanup completed with
  isolated caches and complete rank output.
- Best max-rank results:

  | M | best candidate | candidate us | vs baseline | PR323 us | vs PR323 |
  |---:|---|---:|---:|---:|---:|
  | 512 | direct0 + stage3 + N-major + EPW24 | 1111.138 | -6.27% | 1090.546 | +1.89% |
  | 4096 | direct0 + stage3 + N-major | 4419.630 | -18.30% | 4338.506 | +1.87% |
  | 8192 | direct0 + stage3 + N-major + EPW16 | 8325.422 | -17.13% | 7842.460 | +6.16% |

- Attribution: N-major improved the stage3 parent at M=4096 and M=8192 but
  hurt M=512 unless paired with a smaller expert wave. EPW24 was best at
  M=512, no forced wave was best at M=4096, and EPW16 was best at M=8192.
  Cleanup was neutral-to-negative. Stage2 regressed all points and is rejected.
- Decision: retain three load-specific beam winners. They are not yet
  promotable because no representative point beats PR323. Search neighboring
  legal waves/stages and fill M=1024/2048 for the best families; do not encode
  a production H200 selector yet.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_c_*`.

## Parameter screen 4: remaining Pro wave and direct-stage neighbors

- Hypothesis: smaller expert waves or direct-on stage3 variants may close the
  final representative-point gaps left by the combined beam.
- Protocol: Pro M=512/4096/8192, seed 101, median-10, 8x H200. Tested
  direct-off stage3+N-major with EPW12/8/6/4, stage3 without N-major at
  EPW12/8, and five direct-on stage3/N-major/EPW variants.
- Best updated max-rank results:

  | M | best candidate | candidate us | PR323 us | gap |
  |---:|---|---:|---:|---:|
  | 512 | direct0 + stage3 + N-major + EPW12 | 1099.110 | 1090.546 | +0.79% |
  | 4096 | direct0 + stage3 + N-major + EPW4 | 4400.872 | 4338.506 | +1.44% |
  | 8192 | direct0 + stage3 + N-major + EPW16 | 8325.422 | 7842.460 | +6.16% |

- Attribution: M=512 improves as the combined beam moves from EPW24 toward
  EPW12, then worsens at EPW8/6/4. M=4096 favors EPW4. M=8192 has a shallow
  minimum around EPW16. Every direct-on stage3 variant remained 2.99% to
  19.19% behind PR323 and is rejected.
- Decision: existing public force dimensions are exhausted for the Pro
  representative points and have not met the strict gate. M=512 is within the
  1% remeasurement band, but M=4096/8192 still require an additional
  parameter dimension or split-kernel implementation improvement. Preserve
  all H20 selector behavior; do not port PR323 fusion.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_w_*`.

## Iteration 1: expose the existing split-MN tile to Pro experiments

- Hypothesis: BM128xBN256 with four epilogue warpgroups is implemented and
  legal for the split kernel, but the experiment predicate previously allowed
  only the Flash shape. A Pro-only explicit experiment may improve large M
  without changing architecture.
- Source change: permit the existing `DG_SM90_MOE_SPLIT_MN=1` debug control to
  select the tile for `(experts/rank=48, top-k=6, IH=3072)`. The default value
  remains zero, so H20, H100, generic SM90, and production H200 behavior are
  unchanged.
- Protocol: Pro M=512/4096/8192, seed 101, median-10, 8x H200. Tested default
  split-MN, stage3, stage3+EPW24/16, and stage3+N-major.
- Best split-MN results versus PR323:

  | M | candidate | candidate us | PR323 us | gap |
  |---:|---|---:|---:|---:|
  | 512 | split-MN + stage3 | 1130.084 | 1090.546 | +3.63% |
  | 4096 | split-MN + stage3 + N-major | 4736.657 | 4338.506 | +9.18% |
  | 8192 | split-MN + stage3 + N-major | 8946.647 | 7842.460 | +14.08% |

- Decision: reject split-MN for the H200 Pro selector; the BM64 combined beam
  remains materially faster. Retain only the opt-in experiment capability,
  which has no default-device effect, for reproducibility.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_x_*`.

## Diagnostic 1: split L1/L2 timing attribution

- Method: enabled event-name breakdown in the audited timing harness and
  collected median-10 L1/L2 events for the baseline Pro M=8192 and the current
  load-specific beam winners. Logical-call timing remained the sum of the two
  ordered events.
- Worst-rank results:

  | Case | total us | L1 median us | L2 median us |
  |---|---:|---:|---:|
  | baseline M=8192 | 9682.652 | 5665.421 | 4098.737 |
  | winner M=512 | 1097.781 | 681.811 | 413.106 |
  | winner M=4096 | 4436.979 | 2792.716 | 1624.487 |
  | winner M=8192 | 8340.630 | 5360.889 | 2964.733 |

- Interpretation: the combined winner removes much more L2 time than L1 time
  at M=8192, but L1 still accounts for roughly 64% of the remaining latency.
  The residual PR323 gap cannot be closed by scatter scheduling alone.
- Decision: expose/test separate L1 and L2 tuning parameters while preserving
  the two-kernel architecture and existing shared defaults. Do not pursue
  PR323-style fusion.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_diag_*`.

## Iteration 2: phase-specific expert-wave experiment

- Hypothesis: the shared EPW16 winner at Pro M=8192 may be a compromise
  between L1 and L2; independently tuning each split kernel could reduce both
  phases without architectural changes.
- Source change: add opt-in `DG_SM90_MOE_L1_EXPERTS_PER_WAVE` and
  `DG_SM90_MOE_L2_EXPERTS_PER_WAVE` overrides after selecting the shared
  config. Both default to the existing shared value, so all production and
  non-H200 behavior is unchanged.
- Protocol: current M=8192 parent (`direct0, stage3, N-major, shared EPW16`),
  seed 101, median-10, 8x H200. Sweep L1 or L2 independently over
  EPW4/8/12/24/48 while holding the other phase at 16.
- Result: control was 8344.677 us. The best L1 point was EPW48 at
  8323.244 us and the best L2 point was EPW48 at 8323.972 us, improvements of
  only about 0.26% and still 6.13%-6.14% behind PR323. The response was shallow
  and within the confirmation band.
- Decision: phase-specific waves do not explain the residual large-M gap and
  are not promoted into the H200 selector. Retain default inheritance and move
  to phase-specific pipeline/tile investigation.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_p_*`.

## Iteration 3: phase-specific pipeline-stage experiment

- Hypothesis: L1 and L2 may prefer different pipeline depths even though the
  shared stage3 config is the best aggregate candidate.
- Source change: add opt-in L1/L2 stage overrides that independently recompute
  each phase's stage count, shared-memory size, and launch configuration.
  Defaults inherit the selected shared config exactly.
- Protocol: current Pro M=8192 parent, seed 101, median-10, 8x H200. Test L1
  or L2 at stage2/4 around the shared stage3 control, plus two crossed pairs.
- Result: control was 8348.518 us. The best point, L1-stage4 with L2-stage3,
  was 8329.762 us (about 0.22% faster) and still 6.21% behind PR323. Other
  combinations ranged from 8333.347 to 8351.656 us.
- Decision: phase-specific stage depth is not a material residual lever and is
  not promoted. Continue with phase-specific N tiles while keeping BM and the
  staging layout shared.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_s_*`.

## Iteration 4: phase-specific N-tile experiment

- Hypothesis: the shared BN128 regression may come from only one split phase;
  keeping the other phase at BN256 could recover a better phase-specific tile.
- Source change: add opt-in L1/L2 BN128/BN256 overrides for BM64. Rebuild each
  phase's dispatch/epilogue thread layout, pipeline shared memory, TMA weight
  descriptors, and L1 output/L2 input descriptors independently. Defaults do
  not apply an override.
- Protocol: current Pro M=8192 parent, seed 101, median-10, 8x H200. Compare
  BN256/256 control, BN128/256, BN256/128, and BN128/128.
- Result: control was 8350.336 us; L1-only BN128 was 8343.746 us, L2-only
  BN128 was 8334.010 us, and both BN128 was 8322.295 us. The best nominal
  change is only 0.34% and remains 6.12% behind PR323.
- Decision: phase-specific N tiles are not a material residual lever and are
  not promoted. Investigate per-phase persistent-grid/SM allocation next.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_n_*`.

## Iteration 5: phase-specific persistent-grid experiment

- Hypothesis: H200's 132-SM grid may create avoidable tail waves for the
  48-expert Pro shape; phase-specific 128/120/112/96 CTA grids may improve
  block-wave alignment.
- Source change: add opt-in L1/L2 grid-size overrides, each bounded by the
  physical/runtime SM count. Defaults remain the full runtime SM count.
- Protocol: current Pro M=8192 parent, seed 101, median-10, 8x H200. Compare
  full 132, L1-only 128, L2-only 128, both 128, and both 120/112/96.
- Result: control was 8368.946 us; the best result was both-112 at
  8323.770 us (about 0.54% faster) and still 6.14% behind PR323. All tested
  grids landed between 8323.770 and 8359.728 us aside from the control.
- Decision: grid alignment is not a material residual lever and is not
  promoted. Host-side legal parameter dimensions are exhausted; move to
  split-kernel internal optimization without introducing fusion.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_g_*`.

## Diagnostic 2: PTXAS resources and phase counters

- PTXAS verbose rebuild of the Pro M=8192 winner reported 168 registers for
  both L1 and L2, zero-byte stack frames, and zero spill loads/stores. L1 used
  three barriers and L2 used sixteen. Raising register redistribution limits
  therefore has no spill-removal justification.
- Built-in counters attributed the remaining work primarily to math/GEMM;
  combine barrier/reduce and per-block epilogues were smaller. Event timing
  remained roughly 5.3 ms L1 plus 3.0 ms L2 for the instrumented run.
- Artifacts: `.../candidates/pro_ptxas8192/` and
  `.../candidates/pro_phase8192/`.

## Iteration 6: asynchronous L1 TMA-store experiment

- Hypothesis: the existing double-buffered asynchronous L1 output-store path
  can hide synchronous TMA-store waits behind the next GEMM block.
- Source change: expose the already-implemented path through opt-in
  `DG_SM90_MOE_ASYNC_L1_TMA_STORE`; default remains false. The experiment is
  restricted to non-direct configs whose shared host SMEM allocation already
  covers the double buffer.
- Protocol: current Pro winners at M=512/4096/8192 with EPW12/4/16,
  respectively; seed 101, median-10, 8x H200.
- Results versus PR323 were +5.94%, +2.38%, and +7.30%, all worse than the
  synchronous winners (+0.79%, +1.44%, +6.16%).
- Decision: reject async L1 TMA stores and keep the default synchronous path.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_a_*`.

## Iteration 7: phase-local dual-accumulator experiment

- Hypothesis: the already-implemented L1 dual-K and L2 dual-half accumulator
  paths may expose more independent WGMMA work and reduce the long dependency
  chain that dominates Pro M=8192.
- Source change: expose the two existing paths through opt-in
  `DG_SM90_MOE_L1_DUAL_K_ACCUM` and `DG_SM90_MOE_L2_DUAL_ACCUM` switches.
  Both default to false, so the existing H20 and generic selector behavior is
  unchanged.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  seed 101, median-10, 8x H200. Compare control, L1-only, L2-only, and both;
  report the maximum returned time across ranks.
- Result: control was 8371.925 us (+6.751% versus PR323), L1-only was
  8369.825 us (+6.724%), L2-only was 8336.957 us (+6.305%), and both was
  8354.491 us (+6.529%). The best nominal gain was only 0.42% and remained
  well behind PR323.
- Decision: reject both dual-accumulator paths as H200 selector candidates;
  they do not materially close the large-M gap.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_dacc2_*`.

## Iteration 8: two-CTA weight-multicast experiment

- Hypothesis: H200 may benefit from pairing adjacent M tiles in a two-CTA
  cluster and multicasting each B/weight TMA load to both CTAs, reducing the
  dominant large-M weight traffic without changing the split architecture.
- Source change: restore an opt-in `DG_SM90_MOE_CLUSTER_SIZE` selector for the
  already-implemented cluster scheduler and multicast path. The default
  remains cluster size one, preserving the existing H20 and generic behavior.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  seed 101, median-10, 8x H200. Compare explicit cluster sizes one and two;
  report the maximum returned time across ranks.
- Result: cluster one was 8372.159 us (+6.754% versus PR323); cluster two was
  8348.770 us (+6.456%). The nominal 0.28% gain is inside the confirmation
  band and does not materially close the gap.
- Decision: reject two-CTA B multicast as an H200 selector candidate. H200 L2
  reuse is already effective enough that B traffic is not the residual
  bottleneck.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_cluster2_*`.

## Iteration 9: unscaled E5M2 combine contributions

- Hypothesis: storing each L2-to-combine contribution as unscaled FP8 E5M2
  instead of BF16 can halve NVLink scatter and combine-read bytes while the
  reduction and final output remain FP32/BF16.
- Source change: add opt-in `DG_SM90_MOE_FP8_COMBINE`, including E5M2 L2
  epilogue packing, byte-width-aware NVLink scatter, E5M2-to-FP32 combine
  reduction, and BF16 final output. The default remains the original BF16
  contribution layout, so no H20 selector or default precision changes.
- Protocol: current load-specific Pro parents at M=512/4096/8192 (EPW12/4/16,
  respectively), seed 101, median-10, 8x H200. Compare same-source BF16 and
  E5M2 modes and report maximum returned latency across ranks. A focused
  top-k6 correctness scenario passed with `calc_diff=0.0006 < 0.01`.
- Results:

  | M | BF16 us | E5M2 us | E5M2 vs BF16 | PR323 us | E5M2 vs PR323 |
  |---:|---:|---:|---:|---:|---:|
  | 512 | 1121.283 | 1100.999 | -1.81% | 1090.546 | +0.958% |
  | 4096 | 4432.329 | 4427.793 | -0.10% | 4338.506 | +2.058% |
  | 8192 | 8415.524 | 8346.194 | -0.82% | 7842.460 | +6.423% |

- Decision: do not promote E5M2 globally; it does not address the large-M
  GEMM/scheduling gap. Retain it temporarily as an M=512 sub-candidate because
  the result is inside the 1% remeasurement band. It must beat PR323 on repeat
  and pass the full precision suite before any H200-only selector use.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_fp8combine_v2_*`,
  `.../candidates/pro_fp8combine_v3_*`, and
  `.../candidates/pro_fp8combine_v2_correctness_smoke/`.

## Iteration 10: adjacent scale-domain accumulation

- Hypothesis: keeping the WGMMA destination in the preceding K segment's
  scale domain and multiplying by only the adjacent scale ratio can replace
  the in-place path's two full-accumulator scale passes with one per segment.
- Source change: add opt-in `DG_SM90_MOE_ADJACENT_SCALE_DOMAIN` for M64N128
  warpgroups. The default path and all H20 selector behavior remain unchanged.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  BF16 combine, seed 101, median-10, 8x H200; compare same-source control and
  adjacent-domain modes using maximum returned latency across ranks.
- Result: control was 8362.077 us (+6.626% versus PR323); adjacent-domain was
  8398.652 us (+7.092%), a 0.44% regression.
- Decision: reject adjacent scale-domain accumulation. Its reciprocal and
  accumulator dependency costs outweigh the removed scale pass on H200.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_adjscale_v1_*`.

## Iteration 11: B-loader weight-scale prefetch

- Hypothesis: moving block weight-SF loads from math warpgroups after the full
  barrier to the B-loader warp can hide their latency under the TMA pipeline.
- Source change: add opt-in `DG_SM90_MOE_PREFETCH_WEIGHT_SF`; reserve one
  aligned 128-byte shared line per stage, add a barrier producer, prefetch the
  L1 gate/up or L2 N-group scales in the B-loader, and consume them from shared
  memory. The default path and H20 behavior remain unchanged.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  BF16 combine, seed 101, median-10, 8x H200; report maximum returned latency.
- Result: control was 8413.468 us (+7.281% versus PR323); prefetch was
  8342.900 us (+6.381%), a 0.84% improvement but still far from the gate.
- Decision: do not promote the shared-prefetch form. The signal motivates a
  lower-overhead test that issues the same LDG from math warps before their
  full-barrier wait, avoiding the extra shared stage and producer arrival.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_prefetchwsf_v1_*`.

## Iteration 12: pre-barrier math-warp weight-scale loads

- Hypothesis: issuing each math warp's uniform weight-SF LDG before the stage
  full-barrier wait can hide its latency without the shared-memory producer
  overhead seen in iteration 11.
- Source change: add opt-in `DG_SM90_MOE_EARLY_WEIGHT_SF`; force the LDG value
  live before the full-barrier wait. The default and H20 behavior are unchanged.
- Protocol: current Pro M=8192 parent, BF16 combine, seed 101, median-10,
  8x H200; report maximum returned latency across ranks.
- Result: control was 8359.339 us (+6.591% versus PR323); early LDG was
  8387.056 us (+6.944%), a 0.33% regression.
- Decision: reject early math-warp LDG. Hopper's existing uniform read-only
  load scheduling is already effective, while extending scale live ranges is
  slightly harmful.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_earlywsf_v1_*`.

## Iteration 13: FP16x2 scaled accumulator

- Hypothesis: retaining the cross-K scaled sum in FP16x2 can replace 64 scalar
  FP32 promotions per thread and K segment with 32 packed half2 FMAs.
- Source change: add opt-in `DG_SM90_MOE_FP16_SCALED_ACCUM`; WGMMA still emits
  FP32 raw accumulators, which are converted to half2 for scaled accumulation
  and converted back to FP32 once before the unchanged epilogue. Defaults and
  H20 behavior remain unchanged.
- Protocol: current Pro M=8192 parent, BF16 combine, seed 101, median-10,
  8x H200; report maximum returned latency across ranks.
- Result: control was 8390.828 us (+6.992% versus PR323); FP16x2 accumulation
  was 8338.229 us (+6.322%), a 0.63% improvement but still far from the gate.
- Decision: do not promote this conversion-based form. CUTLASS exposes native
  F16-output WGMMA, so the next bounded experiment will remove the per-segment
  FP32-to-FP16 conversion rather than treating this small result as final.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_fp16acc_v1_*`.

## Iteration 14: native FP16 WGMMA initial launch

- Hypothesis: native `m64n128k32.f16.e4m3.e4m3` WGMMA can emit two packed
  FP16 accumulators per register and remove the FP32-to-FP16 conversion paid
  by iteration 13 after every scale domain.
- Source change: add an opt-in M64N128 native-FP16 WGMMA wrapper, packed
  register fencing, and a scaled half2 accumulation loop behind
  `DG_SM90_MOE_NATIVE_FP16_WGMMA=1`. The default remains disabled and no H20
  or generic selector behavior changes.
- Protocol: intended same-source Pro M=8192 control/candidate comparison on
  job 2957858, seed 101, median-10, 8x H200.
- Result: no kernel was built or timed because the initial command referenced
  `scripts/run_h200_fp8_candidate.sh` inside the remote worktree, while the
  campaign copy lives under the result root.
- Decision: this is a harness-path failure, not evidence about the candidate.
  Retry the identical source and protocol with the campaign runner path.
- Raw artifacts: none; execution stopped before candidate-directory creation.

## Iteration 15: native FP16-output FP8 WGMMA

- Hypothesis: replacing FP32-output WGMMA plus per-domain FP32-to-FP16
  conversion with native packed-FP16 WGMMA can materially reduce the dominant
  GEMM-loop instruction and register cost at Pro M=8192.
- Source change: use `MMA_64x128x32_F16E4M3E4M3_SS_TN` for the opt-in path,
  keep its 32 packed accumulator registers correctly fenced, scale-accumulate
  them with half2 FMA, and convert to FP32 only once for the unchanged
  epilogue. The flag remains default-off and no H20 selector is modified.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  BF16 combine, seed 101, median-10, 8x H200; report maximum returned latency
  across ranks.
- Result: control was 8404.653 us; native FP16 WGMMA was 8380.051 us, a
  nominal 0.29% improvement. It remains 6.85% slower than PR323 at
  7842.460 us.
- Decision: reject native FP16 WGMMA as an H200 selector candidate. The gain
  is below the 1% confirmation band and does not justify its additional
  accumulation rounding or a broad precision campaign.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_nativefp16_v1_*`.

## Iteration 16: wide-N single-consumer initial launch

- Hypothesis: one M64N256 consumer warpgroup with in-place scale-domain
  accumulation can replace the current pair of M64N128 consumers, reducing
  consumer threads and duplicated control work while retaining the same CTA
  tile and split L1/L2 architecture.
- Source change: allow an explicitly forced one-warpgroup BN256 candidate and
  extend the existing in-place FP32 accumulation path to N256, including the
  two independent L2 weight-scale regions. Defaults still select two
  warpgroups, so H20 and generic behavior are unchanged.
- Protocol: current Pro M=8192 parent, seed 101, median-10, 8x H200; compare
  explicit epilogue-WG counts two and one.
- Result: the two-WG control completed at 8378.121 us max-rank. The one-WG
  candidate stopped before JIT with `not candidates.empty()` at the final
  heuristic candidate check; no candidate timing was produced.
- Decision: treat this as an incomplete legality-plumbing iteration, not a
  performance result. Identify and fix the remaining candidate filter, then
  rerun the identical source-level design.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_widen_v1_*`.

## Diagnostic 3: stale host-extension audit

- The remote worktree's `_C.cpython-312-x86_64-linux-gnu.so` predated the
  source campaign. `strings` contained the original public SM90 knobs but none
  of the subsequently added `FP8_COMBINE`, `PREFETCH_WEIGHT_SF`,
  `FP16_SCALED_ACCUM`, `NATIVE_FP16_WGMMA`, phase-specific, or dual-accumulator
  environment names.
- Consequence: source-dependent iterations 1-15 did not have evidence that
  their new host flags reached the generated kernel template. Their nominal
  sub-1% deltas must be treated as same-configuration noise and are not valid
  selector evidence. The original baseline and parameter screens using
  pre-existing public knobs remain valid.
- Repair: force-rebuilt the in-place extension with CUDA 13.2 and the campaign
  venv. The rebuilt binary timestamp is 2026-07-03 13:05:40 PDT, and binary
  strings now contain every added host flag. All subsequent candidate runs use
  fresh candidate/JIT cache names and printed concrete configs.

## Iteration 17: wide-N single-consumer after host rebuild

- Hypothesis: after making the host configuration effective, one M64N256
  consumer with in-place FP32 scale-domain accumulation may outperform the
  standard pair of M64N128 consumers.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  seed 101, median-10, 8x H200. Fresh caches were used after rebuilding the
  host extension. Printed configs confirmed 256 versus 128 epilogue threads
  for the control and wide-N candidate, respectively.
- Result: the two-WG control was 8472.745 us max-rank; the one-WG N256 path was
  10011.301 us, an 18.16% regression and 27.66% slower than PR323.
- Decision: reject the wide-N single-consumer path. One consumer does not
  provide enough WGMMA latency hiding, and its in-place full-accumulator scale
  passes are substantially more expensive than the dual-consumer layout.
- Raw artifacts: `.../sm90_fp8_h200_retune_job2957858/candidates/pro_widen_v2_*`.

## Iteration 18: effective FP16 accumulation comparison

- Hypothesis: with the rebuilt host runtime actually forwarding the template
  flags, native F16-output FP8 WGMMA may remove enough accumulator conversion
  and register traffic to close a material part of the large-M gap.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`, two
  M64N128 consumers), BF16 combine, seed 101, median-10, 8x H200. Compare
  control, FP32-output WGMMA plus half2 scaled accumulation, and native
  F16-output WGMMA in fresh caches. Printed configs matched on all host tiles.
- Results (max-rank): control 8371.610 us; conversion-based half2 accumulation
  8367.652 us (-0.05%); native F16 WGMMA 8034.813 us (-4.02%). The native path
  is 2.45% slower than PR323 at 7842.460 us, versus roughly 6.75% for control.
- Decision: reject the conversion-based form. Retain native F16 WGMMA as the
  first material H200 candidate, but do not promote it until focused and broad
  numerical validation passes; native F16 accumulation changes rounding inside
  each sequence of K32 WGMMAs.
- Correctness gate: the focused eight-rank `L2.profile_topk6.t512` scenario
  failed with `calc_diff=nan`. Native F16 WGMMA accumulates the unscaled
  E4M3-by-E4M3 dot product before the external block scales are applied, so the
  packed FP16 destination can overflow. This is a hard rejection regardless of
  its 4.02% timing gain; no tolerance relaxation or selector promotion is valid.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_fp16_effective_v1_*`
  and `.../candidates/pro_nativefp16_correctness_smoke/`.

## Iteration 19: effective internal-candidate screen

- Hypothesis: after rebuilding the host extension, one of the previously
  unevaluated internal paths may provide a material H200 gain without changing
  the split L1/L2 architecture.
- Protocol: current Pro M=8192 parent (`direct0, stage3, N-major, EPW16`),
  seed 101, median-10, 8x H200. Each mode changed exactly one internal switch
  from the same-source control and used a fresh cache.
- Results (maximum returned latency across ranks):

  | Mode | us | vs control |
  |---|---:|---:|
  | control | 8358.573 | — |
  | async L1 store | 8345.627 | -0.15% |
  | L1 dual-K | 9975.751 | +19.35% |
  | L2 dual-half | 16026.832 | +91.74% |
  | both dual | 17459.549 | +108.88% |
  | E5M2 combine | 8196.752 | -1.94% |
  | adjacent scale domain | 8624.391 | +3.18% |
  | B-loader SF prefetch | 8375.747 | +0.21% |
  | early math-warp SF | 8347.869 | -0.13% |

- Decision: retain E5M2 combine as the only material candidate (-1.94%); it
  requires focused precision validation and multi-M confirmation. Async L1
  store and both weight-SF load variants are inside the 1% noise band. Reject
  adjacent-domain and all dual-accumulator forms, which regress materially.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_effective_screen1_*`.

## Iteration 20: effective E5M2 combine across Pro loads

- Hypothesis: halving the L2-contribution scatter and combine-read bytes with
  unscaled E5M2 is a repeatable gain across the load-specific Pro parents, not
  only a single M=8192 observation.
- Protocol: Pro M=512/4096/8192 with their retained EPW12/4/16 parents,
  respectively; `direct0, stage3, N-major`, seed 101, median-10, 8x H200.
  Compare BF16 and E5M2 contribution storage from the same rebuilt host runtime.
- Results (maximum returned latency across ranks):

  | M | BF16 us | E5M2 us | E5M2 vs BF16 | PR323 us | E5M2 vs PR323 |
  |---:|---:|---:|---:|---:|---:|
+  | 512 | 1119.474 | 1079.510 | -3.57% | 1090.546 | -1.01% |
  | 4096 | 4451.371 | 4341.600 | -2.47% | 4338.506 | 0.07% |
  | 8192 | 8332.095 | 8179.550 | -1.83% | 7842.460 | 4.30% |

- Correctness: the effective eight-rank `L2.profile_topk6.t512` smoke passed
  at `calc_diff=0.0020 < 0.01`. This confirms finite output and the E5M2
  pack/scatter/reduce mapping, but it is not the full 112-case precision gate.
- Decision: retain E5M2 combine. It beats PR323 at M=512 and provides a
  repeatable material gain at M=4096/8192, but still needs the 112-case
  precision campaign and does not alone close the two large-M gaps.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_fp8combine_effective_v2_*`
  and `.../candidates/pro_fp8combine_effective_correctness_smoke/`.

## Iteration 21: effective phase-specific expert waves with E5M2

- Hypothesis: L1 and L2 may prefer different expert-wave sizes once the E5M2
  combine path removes part of L2's scatter/reduce cost.
- Protocol: Pro M=8192 E5M2 parent (`direct0, stage3, N-major, shared EPW16`),
  seed 101, median-10, 8x H200. Sweep one phase over EPW4/8/12/24/48 while
  holding the other at 16, using the rebuilt host runtime.
- Results (maximum returned latency across ranks):

  | L1 EPW | L2 EPW | us | vs 16/16 |
  |---:|---:|---:|---:|
+  | 16 | 16 | 8211.883 | — |
  | 4 | 16 | 8185.451 | -0.32% |
  | 8 | 16 | 8196.595 | -0.19% |
  | 12 | 16 | 8289.427 | 0.94% |
  | 24 | 16 | 8197.039 | -0.18% |
  | 48 | 16 | 8167.025 | -0.55% |
  | 16 | 4 | 8226.518 | 0.18% |
  | 16 | 8 | 8191.213 | -0.25% |
  | 16 | 12 | 8199.580 | -0.15% |
  | 16 | 24 | 8186.790 | -0.31% |
  | 16 | 48 | 8205.725 | -0.07% |

- Decision: the best point is L1 EPW48 / L2 EPW16 at 8167.025 us,
  only 0.55% faster than the 8211.883 us control. This is below the
  1% confirmation band and remains 4.14% behind PR323, so do not
  promote a phase-specific wave rule.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_e5m2_phasewave_v1_*`.

## Iteration 22: effective phase-specific pipeline stages with E5M2

- Hypothesis: L1 and L2 may prefer different TMA pipeline depths after E5M2
  reduces the combine-side work.
- Protocol: Pro M=8192 E5M2 parent, shared EPW16, seed 101, median-10, 8x
  H200. Sweep legal L1/L2 stage counts two through four with fresh caches.
- Results (maximum returned latency across ranks):

  | L1 stages | L2 stages | us | vs 3/3 |
  |---:|---:|---:|---:|
+  | 3 | 3 | 8226.000 | — |
  | 2 | 3 | 8198.978 | -0.33% |
  | 4 | 3 | 8176.543 | -0.60% |
  | 3 | 2 | 8184.229 | -0.51% |
  | 3 | 4 | 8228.157 | 0.03% |
  | 4 | 2 | 8214.189 | -0.14% |
  | 2 | 4 | 8199.900 | -0.32% |
  | 4 | 4 | 8192.161 | -0.41% |

- Decision: the best point is 4/3 stages at 8176.543 us,
  0.60% faster than the 8226.000 us control and
  4.26% behind PR323. This remains inside the 1% confirmation
  band, so phase-specific pipeline depth is not a structural large-M lever.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_e5m2_phasestage_v1_*`.

## Iteration 23: two-plane BLOCK_K=256 mainloop

- Hypothesis: grouping two independently scaled K128 input planes into one
  K256 pipeline stage may reduce producer/barrier iterations enough to improve
  large-M throughput while preserving the existing K128 scale domains.
- Implementation: add an explicitly selected `DG_SM90_MOE_BLOCK_K=256` path.
  Each stage issues two K128 A/B TMA loads; L1 consumes two independent K128
  scale regions and L2 consumes four K64 activation-scale regions with the
  corresponding two K128 weight-scale regions. The existing K128 path remains
  the default, so H20 tuning and default selector behavior are unchanged.
- Protocol: Pro M=8192 E5M2 parent (`direct0, N-major, EPW16`, seed 101,
  median-10, 8x H200), changing only `BLOCK_K`. Printed configs confirmed
  K128/stage3/173760-byte shared memory versus K256/stage2/215216-byte shared
  memory.
- Results (maximum returned latency across ranks):

  | BLOCK_K | stages | shared memory | us | vs K128 | vs PR323 |
  |---:|---:|---:|---:|---:|---:|
  | 128 | 3 | 173760 B | 8164.657 | — | +4.11% |
  | 256 | 2 | 215216 B | 8361.760 | +2.41% | +6.62% |

- Decision: reject K256 for H200 selection. Halving the logical K-stage count
  did not compensate for losing a pipeline stage and increasing per-CTA shared
  memory. Because the performance gate failed materially, no correctness
  campaign was run for this optional path.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bk256_v1_bk*`.

## Iteration 24: chunked native-FP16 WGMMA accumulation

- Hypothesis: the 4.02% native-FP16 WGMMA speedup from iteration 18 may be
  recoverable safely by converting its packed FP16 accumulator to FP32 after
  K128, K64, or K32, then applying the unchanged block scale in FP32.
- Implementation: add an explicitly selected
  `DG_SM90_MOE_NATIVE_FP16_CHUNK_K={32,64,128}` mode. L1 promotes after the
  requested chunk; L2 additionally respects its K64 activation-scale boundary.
  The existing FP32 path and all default/H20 behavior remain unchanged.
- Correctness protocol: eight-rank `L2.profile_topk6.t512`, E5M2 combine,
  tolerance 0.01, testing K128, K64, and the minimum single-instruction K32
  chunk before any performance measurement.
- Result: all three chunk sizes returned `calc_diff=nan`. Because K32 is one
  `m64n128k32.f16.e4m3.e4m3` instruction, the unscaled E4M3 dot product can
  overflow the packed FP16 destination within a single WGMMA; neither earlier
  FP32 promotion nor a FP32 final accumulator can make this representation
  numerically safe.
- Decision: hard-reject every native-FP16 WGMMA variant for these inputs. Do
  not benchmark, relax tolerance, or promote it into an H200 selector.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_nativefp16_chunk_correctness_k*`.

## Iteration 25: four-way M64N64 split-N consumers

- Hypothesis: four independent M64N64 WGMMA consumer warpgroups may hide more
  math latency than the default pair of M64N128 consumers without changing
  the BM64/BN256/BK128 CTA tile or split L1/L2 architecture.
- Implementation: permit the already represented four-way split-N layout only
  for explicit `DG_SM90_MOE_FORCE_EPILOGUE_WG=4`. The first JIT exposed that
  each WG's 32 post-SwiGLU columns are half of the required 64-column L2 scale
  domain. The repaired path exchanges per-row amax values between adjacent WGs,
  computes one shared 64-column scale, then quantizes and TMA-stores the same
  output layout. The default remains two consumers, preserving H20 behavior.
- Protocol: Pro M=8192 E5M2 parent (`direct0, stage3, N-major, EPW16`), seed
  101, median-10, 8x H200; compare two versus four epilogue warpgroups.
- Results (maximum returned latency across ranks):

  | consumer layout | epilogue threads | us | vs 2 WG | vs PR323 |
  |---|---:|---:|---:|---:|
  | 2 x M64N128 | 256 | 8204.985 | — | +4.62% |
  | 4 x M64N64 | 512 | 8403.626 | +2.42% | +7.16% |

- Decision: reject four-way split-N. Additional WGMMA consumers do not offset
  the 512-thread CTA and cross-WG scale-domain synchronization overhead. The
  performance gate failed, so no broader correctness campaign is warranted.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_splitn4_v1_wg2/`,
  `.../candidates/pro_splitn4_v1_wg4/` (initial JIT legality failure), and
  `.../candidates/pro_splitn4_v2_wg4/`.

## Iteration 26: phase-specific four-way consumers

- Hypothesis: the aggregate four-WG regression may hide a gain in one split
  phase, especially L1, which accounts for about 64% of Pro M=8192 latency.
- Implementation: add explicit `DG_SM90_MOE_L1_EPILOGUE_WG` and
  `DG_SM90_MOE_L2_EPILOGUE_WG` experiment overrides. Each phase independently
  recomputes its stage/shared-memory launch configuration. Defaults inherit the
  shared configuration exactly, so H20 and generic behavior are unchanged.
- Protocol: Pro M=8192 E5M2 parent, seed 101, median-10, 8x H200; compare
  L1/L2 consumer counts 2/2, 4/2, and 2/4. Cached JIT sources confirmed that
  the 512-thread specialization appeared only in the requested phase.
- Results (maximum returned latency across ranks):

  | L1 WG | L2 WG | us | vs 2/2 | vs PR323 |
  |---:|---:|---:|---:|---:|
  | 2 | 2 | 8213.641 | — | +4.73% |
  | 4 | 2 | 8281.242 | +0.82% | +5.60% |
  | 2 | 4 | 8290.444 | +0.94% | +5.71% |

- Decision: reject four-way consumers for both phases. Neither phase contains
  a hidden gain, so do not promote a phase-specific consumer-count rule.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_phasewg_v1_*`.

## Iteration 27: packed-BF16 scaled accumulation

- Hypothesis: packed BF16x2 accumulation can retain FP32's exponent range while
  halving scale-accumulation instructions and registers. In particular, it may
  make one M64N256 consumer faster by replacing that path's two full FP32
  reciprocal/rescale passes per K segment.
- Implementation: add explicit `DG_SM90_MOE_BF16_SCALED_ACCUM`. WGMMA still
  produces FP32 raw dot products; each block-scaled contribution is converted
  to BF16x2 and accumulated with packed FMA, then converted to FP32 once for
  the unchanged epilogue. It supports M64N128 and M64N256 and defaults off.
- Correctness: the eight-rank `L2.profile_topk6.t512` smoke passed for both
  two-WG N128 and one-WG N256 at `calc_diff=0.0020 < 0.01`; unlike native FP16,
  BF16's FP32-sized exponent range produced no NaN/Inf.
- Performance protocol: Pro M=8192 E5M2 parent, seed 101, median-10, 8x H200;
  compare FP32/two-WG, BF16/two-WG, and BF16/one-WG.
- Results (maximum returned latency across ranks):

  | accumulator | consumers | us | vs FP32/2WG | vs PR323 |
  |---|---:|---:|---:|---:|
  | FP32 | 2 | 8203.120 | — | +4.60% |
  | BF16x2 | 2 | 8244.344 | +0.50% | +5.12% |
  | BF16x2 | 1 | 9107.493 | +11.03% | +16.13% |

- Decision: reject packed-BF16 accumulation for H200 selection. Conversion
  and packing erase the instruction-count benefit for two consumers, while a
  single consumer still lacks sufficient WGMMA latency hiding.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bf16acc_correctness_wg*`
  and `.../candidates/pro_bf16acc_v1_*`.

## Iteration 28: expanded dispatch/loader frontend

- Hypothesis: Pro M=8192 may benefit from four dispatch warps because built-in
  phase counters show a nontrivial dispatch interval before the dominant GEMM.
  The existing register budget exactly supports four dispatch plus four
  non-epilogue warps alongside the two M64N128 consumers.
- Implementation: permit the expanded 4+4 frontend only when explicitly
  requested with `DG_SM90_MOE_DISPATCH_WARPS=4` on a non-swap compact tile.
  The default BN256 frontend remains 2+2, preserving all H20 behavior.
- Correctness: eight-rank `L2.profile_topk6.t512` passed at
  `calc_diff=0.0020 < 0.01` with the expanded frontend.
- Performance protocol: Pro M=8192 E5M2 parent, seed 101, median-10, 8x H200;
  compare two versus four dispatch warps.
- Results (maximum returned latency across ranks):

  | dispatch/non-epilogue warps | shared memory | us | vs 2+2 | vs PR323 |
  |---|---:|---:|---:|---:|
  | 2+2 | 173760 B | 8196.514 | — | +4.52% |
  | 4+4 | 188112 B | 8891.979 | +8.49% | +13.38% |

- Decision: reject the expanded frontend as a shared L1/L2 configuration.
  Extra threads and shared memory materially outweigh any dispatch gain. Test
  it once as an L1-only phase override because L2 does not perform dispatch.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_dispatch4_correctness_smoke/`
  and `.../candidates/pro_dispatch_v1_*`.

## Iteration 29: phase-specific dispatch frontend

- Hypothesis: the shared 4+4 frontend may regress because L2 pays for extra
  threads despite not running dispatch; expanding only L1 could retain a
  dispatch gain while leaving L2 unchanged.
- Implementation: add explicit `DG_SM90_MOE_L1_DISPATCH_WARPS` and
  `DG_SM90_MOE_L2_DISPATCH_WARPS` overrides. Each phase independently selects
  the 2+2 or 4+4 frontend and recomputes its pipeline/shared-memory launch
  configuration. Defaults inherit the shared 2+2 configuration exactly.
- Protocol: Pro M=8192 E5M2 parent, seed 101, median-10, 8x H200; compare
  L1/L2 dispatch-warp counts 2/2, 4/2, and 2/4.
- Results (maximum returned latency across ranks):

  | L1 dispatch warps | L2 dispatch warps | us | vs 2/2 | vs PR323 |
  |---:|---:|---:|---:|---:|
  | 2 | 2 | 8232.724 | — | +4.98% |
  | 4 | 2 | 8185.800 | -0.57% | +4.38% |
  | 2 | 4 | 8864.429 | +7.67% | +13.03% |

- Decision: hard-reject an expanded L2 frontend. The L1-only result is inside
  the 1% confirmation band and remains well behind PR323, so do not promote it
  alone; test it once in combination with the best sub-1% L1 wave/stage signals.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_phasedispatch_v1_*`.

## Iteration 30: combined sub-1% L1 signals

- Hypothesis: the nominal L1-only gains from four dispatch warps, stage4, and
  EPW48 may be individually small but additive when applied together.
- Protocol: Pro M=8192 E5M2 parent, seed 101, median-10, 8x H200. Hold L2 at
  dispatch2/stage3/EPW16 and compare the L1 control against combinations of
  dispatch4 with stage4 and/or EPW48.
- Results (maximum returned latency across ranks):

  | L1 dispatch | L1 stages | L1 EPW | us | vs control | vs PR323 |
  |---:|---:|---:|---:|---:|---:|
  | 2 | 3 | 16 | 8187.545 | — | +4.40% |
  | 4 | 4 | 16 | 8186.583 | -0.01% | +4.39% |
  | 4 | 3 | 48 | 8210.998 | +0.29% | +4.70% |
  | 4 | 4 | 48 | 8191.486 | +0.05% | +4.45% |

- Decision: reject all combinations. The prior sub-1% observations do not add
  and are not selector-grade evidence. Return to a structural tile/mainloop or
  scheduling change rather than stacking shallow host parameters.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_l1combo_v1_*`.

## Iteration 31: BN512 four-consumer tile with BF16 accumulation

- Hypothesis: BM64/BN512 can halve CTA scheduling and A-tile reloads while
  retaining four independent M64N128 WGMMA consumers. Packed BF16 accumulation
  keeps each consumer within the 512-thread CTA register budget, and E5M2
  combine reduces the L2 shared-output footprint enough for a two-stage pipe.
- Implementation: add explicit `DG_SM90_MOE_FORCE_BLOCK_N=512` support with
  four consumers. Split each B load into two legal BN256 tensor-map planes,
  retain one full-barrier byte expectation, size host shared output using the
  selected E5M2/BF16 combine element width, and leave all defaults unchanged.
  The first launch exposed the tensor-map BN limit; the second JIT exposed the
  old BN<=256 static guard. Both were fixed before producing any result.
- Correctness: the repaired eight-rank `L2.profile_topk6.t512` smoke passed at
  `calc_diff=0.0020 < 0.01` with BN512, BF16 accumulation, and E5M2 combine.
- Performance protocol: Pro M=8192, seed 101, median-10, 8x H200; compare the
  same-source E5M2 BN256/FP32 parent, BN256/BF16 attribution point, and
  BN512/BF16 candidate.
- Results (maximum returned latency across ranks):

  | tile | accumulator | stages | us | vs BN256/FP32 | vs PR323 |
  |---|---|---:|---:|---:|---:|
  | BN256 | FP32 | 3 | 8246.554 | — | +5.15% |
  | BN256 | BF16x2 | 3 | 8260.263 | +0.17% | +5.33% |
  | BN512 | BF16x2 | 2 | 8017.227 | -2.78% | +2.23% |

- Decision: retain BN512/BF16 as the second material H200 candidate after
  E5M2 combine. It does not yet meet the PR323 gate and has only smoke-level
  correctness, so next test mixed L1/L2 tile choices and BN512 scheduler
  neighbors before broad precision validation or selector promotion.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_bf16_correctness_smoke*`
  and `.../candidates/pro_bn512_v1_*`.
