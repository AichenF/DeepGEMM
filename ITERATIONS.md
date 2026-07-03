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
