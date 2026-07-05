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

## Iteration 32: phase-specific BN512 L1 tile

- Hypothesis: BN512 reduces L1 CTA scheduling and A-tile reloads, while L2
  still benefits from the lower thread count and three-stage pipeline of
  BN256. Applying BN512 independently by phase can retain the L1 gain without
  paying the BN512 L2 regression.
- Implementation: extend the explicit `DG_SM90_MOE_L1_BLOCK_N` and
  `DG_SM90_MOE_L2_BLOCK_N` overrides to accept BN512 and derive the matching
  compact frontend and four-consumer epilogue. Both overrides remain zero by
  default, so the existing H20 selector and all H20 tuning results are
  unchanged.
- Correctness: the exact L1 BN512 / L2 BN256 eight-rank
  `L2.profile_topk6.t512` scenario passed at `calc_diff=0.0020 < 0.01` with
  BF16x2 accumulation and E5M2 combine.
- Performance protocol: Pro M=8192, seed 101, median-10, 8x H200; BF16x2
  accumulation and E5M2 combine for all four phase-tile combinations. BN256
  uses three stages and BN512 uses two stages.
- Results (maximum returned latency across ranks):

  | L1 BN | L2 BN | us | vs 256/256 | vs PR323 |
  |---:|---:|---:|---:|---:|
  | 256 | 256 | 8250.647 | — | +5.21% |
  | 512 | 256 | 7742.250 | -6.16% | -1.28% |
  | 256 | 512 | 8559.457 | +3.74% | +9.14% |
  | 512 | 512 | 8052.001 | -2.41% | +2.67% |

- Decision: retain L1 BN512 / L2 BN256 as the first M=8192 H200 candidate
  that beats PR323. Hard-reject BN512 for L2. Do not add an automatic selector
  yet: first confirm the 1.28% margin across seeds and tune the phase-specific
  scheduler neighbors, then validate all required M points and the full
  precision matrix.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_phase_v1_*`
  and `.../candidates/pro_bn512_phase_512_256_correctness_smoke/`.

## Iteration 33: interleaved M8192 confirmation

- Hypothesis: the 1.28% M8192 lead from iteration 32 is large enough to
  survive longer samples and interleaved execution with PR323, rather than
  being caused by a favorable single-run clock or load state.
- Protocol: Pro M=8192, seed 101, median-20, 8x H200. Interleave three runs of
  the retained L1 BN512 / L2 BN256 candidate with three unmodified PR323 runs.
  For each run, report the maximum returned latency across ranks.
- Results:

  | observation | ours us | PR323 us | ours vs PR323 |
  |---:|---:|---:|---:|
  | 1 | 7728.855 | 7844.184 | -1.47% |
  | 2 | 7711.712 | 7866.673 | -1.97% |
  | 3 | 7751.338 | 7867.560 | -1.48% |

- The median of the three run maxima is 7728.855 us for ours and
  7866.673 us for PR323, a 1.75% lead. The conservative cross-run comparison
  (slowest ours versus fastest PR323) still leads by 1.18%. Ours spans 0.51%
  across observations and PR323 spans 0.30%.
- Decision: confirm L1 BN512 / L2 BN256 as a stable H200 M8192 winner and use
  it as the next tuning parent. This is still an explicit, default-off H200
  candidate; no automatic selector or H20 tuning entry is changed.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_phase_interleaved_s101_n20/`.

## Iteration 34: BN512 L1 expert-wave sweep

- Hypothesis: halving L1 CTA count with BN512 may shift the preferred number
  of experts scheduled per wave away from the BN256 parent's EPW16.
- Protocol: Pro M=8192, seed 101, median-10, 8x H200. Hold L1 BN512 / L2
  BN256, BF16x2 accumulation, E5M2 combine, and L2 EPW16 fixed; sweep L1
  EPW4/8/12/16/24/48. Report maximum returned latency across ranks.
- Results:

  | L1 EPW | us | vs EPW16 |
  |---:|---:|---:|
  | 4 | 7725.876 | +0.72% |
  | 8 | 7733.768 | +0.82% |
  | 12 | 7809.523 | +1.81% |
  | 16 | 7670.768 | — |
  | 24 | 7680.280 | +0.12% |
  | 48 | 7748.728 | +1.02% |

- Decision: retain L1 EPW16. EPW24 is inside the noise band and all other
  neighbors regress. No selector change is justified, and the existing H20
  tuning remains untouched.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_l1wave_v1_*`.

## Iteration 35: BN256 L2 expert-wave sweep and confirmation

- Hypothesis: after L1 moves to BN512, the changed arrival pattern into L2
  may shift L2's preferred expert wave away from EPW16.
- Initial protocol: Pro M=8192, seed 101, median-10, 8x H200. Hold the retained
  candidate fixed and sweep L2 EPW4/8/12/16/24/48. The initial run suggested
  EPW12/24 could be about 1.3% faster than its EPW16 control, but that control
  was itself 1.4% slower than the preceding experiment.
- Confirmation protocol: interleave EPW12/16/24 in rotating order, three
  observations each, median-20. Report the maximum returned latency across
  ranks for every observation, then compare the median of those maxima.
- Confirmation results:

  | L2 EPW | observation maxima (us) | median us | vs EPW16 |
  |---:|---|---:|---:|
  | 12 | 7736.242, 7768.476, 7753.274 | 7753.274 | +0.49% |
  | 16 | 7705.099, 7776.444, 7715.600 | 7715.600 | — |
  | 24 | 7773.342, 7776.561, 7750.844 | 7773.342 | +0.75% |

- Decision: reject the apparent first-pass EPW12/24 gain and retain L2
  EPW16. Interleaving reverses the result and demonstrates that sub-percent
  sequential sweeps on this node need confirmation before promotion. No H20
  tuning or default selector is changed.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_l2wave_v1_*`
  and `.../candidates/pro_bn512_l2wave_confirm_s101_n20/`.

## Iteration 36: BN256 L2 stage and N-major sweep

- Hypothesis: BN512 L1 changes the producer cadence enough that L2 may prefer
  a different pipeline depth or no longer benefit from N-major scheduling.
- Initial protocol: Pro M=8192, seed 101, median-10, 8x H200. Sweep L2 stages
  2/3/4 with N-major enabled and disabled; keep all other retained parameters
  fixed.
- Initial results (maximum returned latency across ranks):

  | L2 N-major | L2 stages | us |
  |---:|---:|---:|
  | 1 | 2 | 7756.274 |
  | 1 | 3 | 7707.218 |
  | 1 | 4 | 7665.443 |
  | 0 | 2 | 7840.451 |
  | 0 | 3 | 7822.364 |
  | 0 | 4 | 7812.259 |

- N-major disabled regressed every stage by 1.4% or more and is rejected.
  The apparent 0.54% stage4 lead was then checked with three rotating,
  interleaved median-20 observations per stage.
- Confirmation results:

  | L2 stages | observation maxima (us) | median us | vs stage3 |
  |---:|---|---:|---:|
  | 3 | 7727.468, 7791.515, 7737.307 | 7737.307 | — |
  | 4 | 7813.226, 7752.464, 7712.187 | 7752.464 | +0.20% |

- Decision: retain L2 N-major with three stages. The sequential stage4 signal
  does not reproduce under interleaving. No H200 selector is promoted yet and
  the existing H20 tuning remains unchanged.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_l2sched_v1_*`
  and `.../candidates/pro_bn512_l2stage_confirm_s101_n20/`.

## Iteration 37: BN512 candidate across Pro M>=128

- Hypothesis: the L1 BN512 / L2 BN256 winner is useful across the required
  Pro load range, but may need an H200-only threshold where low-M CTA
  parallelism becomes more important than fewer L1 A-tile reloads.
- Protocol: Pro M=128/256/512/1024/2048/4096/8192, seed 101, median-10,
  8x H200. Interleave the retained split-FP8 candidate and unmodified PR323 at
  each M; report the maximum returned latency across ranks.
- Results:

  | M | ours us | PR323 us | ours vs PR323 |
  |---:|---:|---:|---:|
  | 128 | 870.482 | 894.226 | -2.66% |
  | 256 | 886.226 | 859.090 | +3.16% |
  | 512 | 1117.651 | 1086.498 | +2.87% |
  | 1024 | 1541.193 | 1580.518 | -2.49% |
  | 2048 | 2342.198 | 2446.090 | -4.25% |
  | 4096 | 4054.989 | 4305.722 | -5.82% |
  | 8192 | 7765.363 | 7867.802 | -1.30% |

- Decision: retain BN512 for M=128 and M>=1024 on the Pro shape, subject to
  broader seeds and precision validation. Do not use it at M=256/512; tune a
  BN256 H200 path for those points. Any eventual thresholds must live in an
  H200-specific selector branch and must not overwrite the existing H20
  tuning table.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_bn512_mge128_vs_pr_s101_n10/`.

## Iteration 38: BN256 small-M scheduler screen

- Hypothesis: M=256/512 recover their CTA parallelism with the pre-BN512
  FP32-accumulator BN256 path, while E5M2 combine plus load-specific stage and
  N-major choices can close the remaining PR323 gap.
- Protocol: Pro M=256/512, seed 101, median-10, 8x H200. Use BN256,
  FP32 accumulation, E5M2 combine, direct scatter disabled; sweep stage3/4 and
  N-major0/1. Use EPW16 at M=256 and the retained EPW12 parent at M=512.
- Results (maximum returned latency across ranks):

  | M | EPW | N-major | stages | ours us | current PR323 us | gap |
  |---:|---:|---:|---:|---:|---:|---:|
  | 256 | 16 | 0 | 3 | 872.451 | 859.090 | +1.56% |
  | 256 | 16 | 0 | 4 | 859.046 | 859.090 | -0.01% |
  | 256 | 16 | 1 | 3 | 857.941 | 859.090 | -0.13% |
  | 256 | 16 | 1 | 4 | 858.499 | 859.090 | -0.07% |
  | 512 | 12 | 0 | 3 | 1176.852 | 1086.498 | +8.32% |
  | 512 | 12 | 0 | 4 | 1157.667 | 1086.498 | +6.55% |
  | 512 | 12 | 1 | 3 | 1113.959 | 1086.498 | +2.53% |
  | 512 | 12 | 1 | 4 | 1152.261 | 1086.498 | +6.05% |

- A follow-up old-versus-new JIT-cache diagnostic at M=512 produced
  1078.710/1113.428 us for the older cache and 1096.323/1114.244 us for the
  current cache. The overlapping, roughly 3% cross-run movement does not
  establish a source regression; it reinforces the need for interleaved
  candidate/PR confirmation.
- Decision: use N-major1/stage3 as the small-M screening parent. M=256 is only
  a noise-level lead and M=512 still misses the gate; sweep EPW, then confirm
  any winner interleaved with PR323. Keep all settings H200-only and
  default-off.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_smallm_sched_v1_*`
  and `.../candidates/pro_m512_old_new_jit_compare_s101_n20/`.

## Iteration 39: BN256 small-M expert waves

- Hypothesis: M=256 and M=512 need different expert-wave sizes after E5M2
  combine; the wave can recover scheduler utilization without changing the
  kernel structure.
- Screen protocol: Pro M=256/512, seed 101, median-10, 8x H200. Hold BN256,
  FP32 accumulation, E5M2 combine, N-major1, and stage3; sweep
  EPW8/12/16/24/48.
- Screen results (maximum returned latency across ranks):

  | EPW | M256 us | M512 us |
  |---:|---:|---:|
  | 8 | 871.846 | 1117.557 |
  | 12 | 851.154 | 1119.348 |
  | 16 | 862.563 | 1109.572 |
  | 24 | 854.418 | 1088.051 |
  | 48 | 856.979 | 1163.685 |

- Confirmation protocol: interleave the M256/EPW12 and M512/EPW24 winners
  with PR323, three observations each, median-20. Compare the median of each
  run's maximum-rank latency.
- Confirmation results:

  | M | ours observation maxima (us) | ours median | PR323 maxima (us) | PR323 median | gap |
  |---:|---|---:|---|---:|---:|
  | 256 | 853.523, 880.067, 862.018 | 862.018 | 852.435, 863.330, 858.386 | 858.386 | +0.42% |
  | 512 | 1081.414, 1086.163, 1122.420 | 1086.163 | 1093.012, 1101.043, 1083.410 | 1093.012 | -0.63% |

- Decision: retain EPW24 as the provisional M512 H200 candidate, but require
  broader confirmation because one observation was noisy. EPW12 does not meet
  the M256 gate after interleaving; continue with tile/frontend structure at
  M256. No H20 selector or tuning result is changed.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_smallm_wave_v1_*`
  and `.../candidates/pro_smallm_wave_vs_pr_confirm_s101_n20/`.

## Iteration 40: M256 tile and frontend screen

- Hypothesis: M=256 needs more CTA or warp-level parallelism than the retained
  M64N256 two-consumer tile, and may benefit from BN128, a different consumer
  count, direct scatter, cleanup, or a wider frontend.
- Protocol: Pro M=256, seed 101, median-10, 8x H200. Hold FP32 accumulation,
  E5M2 combine, stage3, N-major1, and EPW12 unless the candidate changes the
  tile/frontend mode. Report maximum returned latency across ranks.
- Results:

  | candidate | us | outcome |
  |---|---:|---|
  | BM64 BN256, 2 consumers | 852.868 | retained control |
  | BM64 BN256, 1 consumer | 897.426 | reject |
  | BM64 BN256, 4 consumers | 864.661 | reject |
  | BM64 BN256, direct scatter | 854.210 | reject |
  | BM64 BN256, cleanup | 877.234 | reject |
  | BM64 BN256, 4-warp frontend | 1005.044 | reject |
  | BM64 BN128, 1 consumer | 945.667 | reject |
  | BM64 BN128, 1 consumer + direct | 926.630 | reject |
  | BM128 BN128, 2 consumers | 1316.292 | reject |
  | BM128 BN256, 4 consumers | 1007.859 | reject |

- BN128 with two consumers is not a legal existing layout: each L1 consumer
  would own only 32 post-SwiGLU columns and fails the compile-time 64-column
  scale-domain requirement. Cluster2 produced an unspecified launch failure
  and hung surviving ranks; the run was terminated and GPU processes were
  cleaned before continuing. Neither mode is retained.
- Phase timing of the retained control showed about 551 us in L1 versus
  304 us in L2 at M=256, so subsequent tuning focuses on phase-local L1
  scheduling rather than further global tile changes.
- Decision: retain BM64 BN256 with two M64N128 consumers and the compact
  frontend. All tested structural alternatives regress or are invalid. These
  were explicit H200 experiments; H20 defaults remain unchanged.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_m256_tile_v1_*`
  and `.../candidates/pro_smallm_phase_breakdown/`.

## Iteration 41: M256 phase-local wave and stage tuning

- Hypothesis: because L1 is about two thirds of M=256 latency, independent L1
  and L2 expert waves plus phase-local pipeline depth can remove the global
  scheduler compromise.
- First screen: with stage3/3, L1/L2 EPW12/24 reached 851.556 us versus
  856.818 us for 12/12 (-0.61%). The other useful neighbor was L1/L2 EPW8/12
  at 852.403 us; all remaining one-phase wave changes were 857.874 us or
  slower.
- Stage screen at L1/L2 EPW12/24:

  | L1 stages | L2 stages | us |
  |---:|---:|---:|
  | 3 | 3 | 861.715 |
  | 2 | 3 | 848.371 |
  | 4 | 3 | 861.411 |
  | 3 | 2 | 984.662 |
  | 3 | 4 | 861.363 |
  | 2 | 4 | 858.354 |
  | 4 | 4 | 858.643 |

- L1 stage2 is the only material stage signal. A follow-up wave screen with
  L1/L2 stages2/3 found 852.549 us for EPW12/16, while EPW12/24 varied from
  848.371 us in the stage screen to 886.451 us in the wave screen. This
  required an interleaved confirmation rather than promotion of the minimum.
- Confirmation protocol: compare stage2/3 with L1 EPW12 and L2 EPW16 or 24
  against PR323, three rotating observations each, median-20.
- Confirmation results:

  | candidate | observation maxima (us) | median us | vs PR323 median |
  |---|---|---:|---:|
  | L2 EPW16 | 860.706, 859.205, 851.956 | 859.205 | +0.50% |
  | L2 EPW24 | 857.235, 851.601, 906.563 | 857.235 | +0.27% |
  | PR323 | 854.914, 848.772, 857.058 | 854.914 | — |

- Decision: retain L1 stage2 / L2 stage3 as the next M256 parent, but neither
  phase-wave candidate meets the PR323 gate after interleaving. Do not encode
  an H200 selector yet; test bounded L1 accumulator/store scheduling changes.
  The H20 tuning table remains untouched.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_m256_phasewave_v1_*`,
  `.../candidates/pro_m256_phasestage_v1_*`,
  `.../candidates/pro_m256_phasewave_s2_v1_*`, and
  `.../candidates/pro_m256_phasewave_s2_confirm_s101_n20/`.

## Iteration 42: M256 accumulator and scale-scheduling screen

- Hypothesis: the stage2 L1 parent is now limited by register pressure or
  scale-factor issue placement; BF16x2 scaled accumulation and earlier weight
  scale loads may improve the L1-dominated path without changing its tile.
- Protocol: Pro M=256, seed 101, median-10, 8x H200. Hold L1/L2 stages2/3,
  BN256, N-major1, FP8 E5M2 combine, and L1/L2 EPW12/16 unless noted. Screen
  accumulator, store, prefetch, scale-domain, and dual-accumulator features.
- Screen results (maximum returned latency across ranks):

  | candidate | us | outcome |
  |---|---:|---|
  | control | 859.810 | retained parent |
  | BF16x2 scaled accumulator | 854.050 | useful signal |
  | FP16x2 scaled accumulator | 862.421 | reject |
  | asynchronous L1 TMA store | launch failure | reject |
  | prefetched weight scale | 867.202 | reject |
  | early weight scale | 851.924 | useful signal |
  | adjacent scale domain | 863.155 | reject |
  | L1 dual-K accumulator | 884.899 | reject |

- Combining early weight-scale scheduling with FP32 accumulation and L2
  EPW16/48 reached 854.003/848.101 us in the sequential screen. BF16x2 did
  not improve the combination. Interleaved median-20 confirmation did not
  reproduce the apparent lead:

  | candidate | observation maxima (us) | median us | vs PR323 median |
  |---|---|---:|---:|
  | early scale, L2 EPW16 | 859.411, 859.330, 853.569 | 859.330 | +1.10% |
  | early scale, L2 EPW48 | 856.355, 857.843, 859.075 | 857.843 | +0.92% |
  | PR323 | 849.235, 850.387, 849.986 | 849.986 | — |

- A follow-up L1 wave sweep with early scale (EPW4/6/8/12 paired with L2
  EPW16/48) found no candidate below the retained EPW12 screen result.
- Decision: do not promote a global accumulator or scale-scheduling flag.
  Because L1 accounts for about two thirds of M256 latency, isolate these
  knobs by phase before rejecting them. All knobs remain default-off and no
  H20 selector is changed.
- Raw artifacts:
  `.../candidates/pro_m256_feature_v1_*`,
  `.../candidates/pro_m256_early_combo_v1_*`,
  `.../candidates/pro_m256_early_confirm_s101_n20/`, and
  `.../candidates/pro_m256_early_l1wave_v1_*`.

## Iteration 43: M256 L2 BN128 screen

- Hypothesis: L1 must retain BN256, but the shorter L2 K dimension may benefit
  from BN128 and the additional output-tile parallelism.
- Protocol: Pro M=256, seed 101, median-10, 8x H200. Keep L1 BN256/stage2,
  early weight-scale scheduling, FP32 accumulation, and E5M2 combine. Set only
  L2 to BN128 and sweep stage3/4, EPW6/12/16/24/48, plus direct scatter at the
  stage3/EPW16 point.
- Results (maximum returned latency across ranks):

  | L2 stages | L2 EPW | direct scatter | us |
  |---:|---:|---:|---:|
  | 3 | 6 | 0 | 1016.196 |
  | 3 | 12 | 0 | 893.490 |
  | 3 | 16 | 0 | 898.882 |
  | 3 | 24 | 0 | 896.002 |
  | 3 | 48 | 0 | 895.219 |
  | 4 | 16 | 0 | 896.310 |
  | 3 | 16 | 1 | 877.891 |

- Decision: reject L2 BN128. Even its best direct-scatter result is about
  3.3% slower than the contemporaneous PR323 reference near 850 us. Continue
  with BN256 and phase-local accumulator/scale scheduling. The experiment is
  H200-only; H20 tuning and defaults remain untouched.
- Raw artifacts:
  `.../sm90_fp8_h200_retune_job2957858/candidates/pro_m256_l2bn128_v1_*`.

## Iteration 44: M256 phase-local feature and frontend isolation

- Hypothesis: the earlier global BF16/early-scale signals may belong only to
  L1, which contributes roughly two thirds of M256 latency; alternatively a
  phase-local frontend or BN512 L1 tile may remove scheduler overhead.
- Temporary implementation: allow L1/L2 to select early weight-SF and BF16x2
  accumulation independently. These experiment-only overrides stayed off by
  default and were removed after the screen.
- Phase-feature screen (Pro M=256, seed 101, median-10, 8x H200, maximum rank):

  | candidate | us |
  |---|---:|
  | control | 893.138 |
  | L1 early weight-SF | 882.596 |
  | L2 early weight-SF | 894.885 |
  | L1 BF16x2 | 880.563 |
  | L2 BF16x2 | 888.276 |

- Three-observation median-20 follow-up did not reproduce a useful feature
  gain: control was 889.379 us, L1 BF16x2 888.693 us, L1 early scale
  893.858 us, and PR323 851.526 us.
- BN512 L1 with BN256 L2 reached 870.531 us at its best L1/L2 wave setting,
  still well behind PR323. L1 four-consumer and four-dispatch-warp variants
  were 884.113 and 895.941 us, respectively; the other phase-local frontend
  changes were slower.
- Decision: reject all phase-local feature/frontend variants and remove their
  temporary host plumbing. Keep BN256 and the compact two-consumer frontend.
  H20 defaults and tuning are unchanged.
- Raw artifacts:
  `.../candidates/pro_m256_phasefeat_v1_*`,
  `.../candidates/pro_m256_phasefeat_confirm_s101_n20/`,
  `.../candidates/pro_m256_l1bn512_v1_*`, and
  `.../candidates/pro_m256_phasefrontend_v1_*`.

## Iteration 45: M256 phase-local L1 BLOCK_K=256

- Hypothesis: M256 L1 already runs only two pipeline stages at BK128, so
  grouping two independently scaled K128 planes into one BK256 stage can
  halve its 56 logical K-stage iterations without sacrificing pipeline depth.
  L2 remains BK128/stage3.
- Implementation: add independent `DG_SM90_MOE_L1_BLOCK_K` and
  `DG_SM90_MOE_L2_BLOCK_K` host overrides. They inherit the existing config
  and remain opt-in; the H20 selector is not modified.
- Initial same-source screen improved the maximum-rank latency from
  889.410 us at L1 BK128 to 860.628 us at L1 BK256. Focused eight-rank
  correctness on `L2.profile_topk6.t512` passed at
  `calc_diff=0.0020 < 0.01` with E5M2 combine.
- Wave screen found L1 EPW8 and L2 EPW48 as the best stable parent. The useful
  neighbors were L1 EPW8/12/48 at 852.146/853.124/852.837 us and L2
  EPW24/48 at 852.978/852.755 us.
- Interleaved median-20 confirmation:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | L1 BK256, L1/L2 EPW8/48 | 857.091, 859.493, 851.174 | 857.091 |
  | PR323 | 857.075, 855.890, 851.814 | 855.890 |

  The initial confirmed gap is +0.14%; later interleaved batches continued to
  alternate around parity rather than establish the required lead.
- PTXAS/SASS audit reported 168 registers, zero spill stores/loads, and three
  barriers for L1. Weight-scale reads already lower to `LDG.E.CONSTANT`, so
  warp shuffle or manual scale prefetch is not a remaining opportunity.
- Decision: retain phase-local L1 BK256 as the M256 H200 parent and keep its
  focused correctness evidence, but do not encode an H200 selector until it
  has a stable margin over PR323. H20 remains on its existing BK128 tuning.
- Raw artifacts:
  `.../candidates/pro_m256_l1bk256_*`,
  `.../candidates/pro_m256_l1bk256_confirm_s101_n20/`,
  `.../candidates/pro_m256_l1bk256_correctness_smoke/`, and
  `.../candidates/pro_m256_l1bk256_resource_v1/`.

## Iteration 46: BK256 bounded microarchitecture follow-ups

- Hypothesis: after BK256 reaches parity, one small producer, accumulator, or
  cache-policy change may provide the remaining stable margin without fusing
  L1/L2 or changing the H20 path.
- Results:

  | candidate | result | decision |
  |---|---|---|
  | packed L1 SFA TMA | 863.236 us vs 852.164 us control median | reject (+1.30%) |
  | packed L2 SFA TMA | 876.195 us screen | reject |
  | dual-plane accumulators | 863.780 us vs 859.237 us control median | reject (+0.53%) |
  | BK256 BF16x2, EPW8 | 857.701 us vs 858.101 us control; PR 853.027 us | reject (noise-level) |
  | BK256 BF16x2, EPW24 | 864.657 us vs 862.659 us control median | reject |
  | initialize from first plane | 862.981 us vs 856.932 us old path median | reject (+0.71%) |
  | B-TMA `EVICT_FIRST` | 871.955 us screen | reject |
  | move empty-barrier release earlier | 859.842 us vs 858.738 us | reject |
  | precompute scale products | 865.827 us vs 853.780 us | reject |
  | vectorized paired weight-SF load | 862.645 us vs 854.097 us | reject |

- The BK256+BF16 focused correctness smoke passed at
  `calc_diff=0.0020 < 0.01`; it was rejected for performance, not accuracy.
- Cross-load checks also rejected L1 BK256 at Pro M512
  (1144.579 us versus 1130.339 us BK128) and L2 BN512/BF16 at Pro M256
  (1001.123 us versus roughly 857 us for BN256).
- Decision: revert every microarchitecture candidate and keep only the clean
  L1-BK256/L2-BK128 parent. No PR323 fused code was copied or implemented;
  PR323 remained a black-box timing reference. H20 tuning and selectors remain
  untouched.
- Raw artifacts:
  `.../candidates/pro_m256_l1bk256_packsfa_*`,
  `.../candidates/pro_m256_l1bk256_dualplane_*`,
  `.../candidates/pro_m256_l1bk256_bf16_*`,
  `.../candidates/pro_m256_l1bk256_initfirst_*`,
  `.../candidates/pro_m256_l1bk256_bevictfirst_v1/`,
  `.../candidates/pro_m512_l1bk256_*`, and
  `.../candidates/pro_m256_l1bk256_l2bn512_bf16_v1/`.

## Iteration 47: M256 cache-policy and asynchronous-store closure

- Hypothesis: after BK256 reaches parity, a cache-policy adjustment or a
  corrected double-buffered L1 TMA store may provide the final few
  microseconds without changing the split architecture.
- Weight TensorMap L2-promotion screening retained the original 256-byte
  promotion policy: 256/128/64/NONE produced 853.843/859.105/858.389/
  867.411 us. B-TMA `EVICT_FIRST` and A-TMA `EVICT_LAST` were also rejected at
  871.955 and 863.011 us.
- A bounded joint-wave screen produced 858.354, 856.627, and 861.218 us for
  L1/L2 EPW 8/24, 12/24, and 48/48. The sub-2-us signal was not promoted.
- The prior asynchronous-store launch failure was traced to three
  experiment-only defects: host SMEM sizing did not reserve the double
  buffer, split-N over-allocated the buffer stride, and stage one stored from
  the stage-zero base. After correcting all three, the candidate ran without
  a CUDA fault, but reached 861.600 us versus 856.193 us for the synchronous
  control (+0.63%).
- Decision: reject and fully revert every cache/TMA experiment. The retained
  source remains the clean BK256 parent from iteration 46; H20 defaults and
  selectors are unchanged.
- Raw artifacts:
  `.../candidates/pro_m256_l1bk256_l2promotion_v1/`,
  `.../candidates/pro_m256_l1bk256_{aevictlast,bevictfirst}_v1/`,
  `.../candidates/pro_m256_l1bk256_jointwave_*`, and
  `.../candidates/pro_m256_l1bk256_asyncfix_*`.

## Iteration 48: H200 grid alignment and remaining small-M points

- M256 grid hypothesis: using 128 of H200's 132 SMs may align persistent CTA
  waves better than the full grid. The BK256 parent was screened at 124, 128,
  130, and 132 SMs; 128 SMs was the only useful point.
- Interleaved Pro M256 median-20 confirmation:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | BK256, 128 SM | 849.746, 867.493, 846.675 | 849.746 |
  | BK256, 132 SM | 868.465, 872.996, 863.394 | 868.465 |
  | PR323 | 855.331, 856.610, 858.755 | 856.610 |

  The 128-SM candidate leads PR323 by 0.80%. Exact eight-rank focused
  correctness passed at `calc_diff=0.0020 < 0.01`. This remains an explicit
  H200 parameter candidate; no H20 entry is overwritten.
- Pro M512 did not obtain a stable winner. The old BN256/EPW24 candidate
  reconfirmed at 1117.910 us versus 1090.839 us for PR323. Phase-local wave
  searches reached isolated low screens but failed median-20 confirmation.
  L1-BN512/BF16 plus L2-BN256/FP32 passed focused correctness and screened at
  1065.970 us, but confirmed at 1091.810 us versus 1085.427 us for PR323
  (+0.59%). Wave and 120/128-SM follow-ups did not reproduce a lead. The
  temporary phase-BF16 host override was therefore removed.
- Flash M128 improved to 262.785 us with E5M2, non-direct N-major scheduling,
  and EPW4, versus the pinned PR323 result near 295.0 us (-10.9%). Flash M512
  improved from the original 425.5-us baseline to a best screen of 363.265 us
  with E5M2, non-direct N-major scheduling, and EPW16, but remains about 1.1%
  behind the pinned PR323 result near 359.3 us. Phase waves, stages, SM-grid
  sizes, L1 BN512, L1 BK256, direct scatter, cleanup, and phase-local BF16
  attribution did not establish a better candidate.
- Decision: retain the confirmed Pro M256 128-SM result and the Flash M128
  parameter result. Pro M512 and Flash M512 remain the two unsolved M>=128
  points. Do not encode final H200 selector thresholds yet; M<128 and every
  H20 path remain unchanged.
- Raw artifacts:
  `.../candidates/pro_m256_l1bk256_sms*`,
  `.../candidates/pro_m512_*`,
  `.../candidates/flash_small_wave_v1_*`, and
  `.../candidates/flash_m512_*`.

## Iteration 49: Flash M512 cluster and tail-wave search

- Hypothesis: Flash M512 remains close to PR323 and may benefit from the
  existing two-CTA B multicast path plus load-specific expert waves, reducing
  repeated weight traffic without changing the split L1/L2 architecture.
- An asymmetric L1/L2 SM-grid experiment (128/132 and 132/128) was rejected
  as illegal in practice: the first run timed out before profiling and the
  reverse order failed after the stale rendezvous. Symmetric 128/120-SM grids
  also regressed, so phase grid sizes must remain equal.
- Cluster2 was legal for Flash M512. The initial global wave screen produced
  max-rank 350.704, 343.249, 346.337, and 344.386 us for EPW2/4/8/32.
  Exact eight-rank focused correctness passed at
  `calc_diff=0.0020 < 0.01`.
- Fresh interleaved median-20 confirmation showed that the pinned historical
  PR result near 359 us was no longer representative of the current node
  state. Global EPW4 confirmed at 350.193 us versus 347.330 us for PR323
  (+0.82%).
- Phase-local cluster waves reduced the screen minimum to 342.689 us at
  L1/L2 EPW16/4. Its three-observation confirmation was 365.362, 349.248,
  and 347.794 us (median 349.248), versus PR323 at 347.730, 346.962, and
  346.914 us (median 346.962), still +0.66%.
- A final stage screen selected L1/L2 stage4/4 at 345.056 us, but confirmation
  again failed: ours 347.378, 362.528, 351.777 us (median 351.777) versus
  PR323 346.609, 346.752, 355.664 us (median 346.752), or +1.45%. Combining
  cluster2 with 128/120 SMs regressed further.
- Pro M512 cluster2 was not legal for this implementation and failed on the
  first EPW4 launch with `CUDA_ERROR_LAUNCH_FAILED`; no Pro cluster result is
  retained.
- Decision: do not promote cluster2 or any new Flash M512 selector. It closes
  most of the gap in favorable screens but does not beat PR323 under the
  required interleaved confirmation. No source change, H20 selector change,
  or M<128 change is retained.
- Raw artifacts:
  `.../candidates/flash_m512_cluster2_*`,
  `.../candidates/flash_m512_phasesms_*`, and
  `.../candidates/pro_m512_cluster2_v1_*`.

## Iteration 50: Flash M512 cluster2 BF16 candidate

- Hypothesis: after cluster2 plus phase waves/stages closes most of the Flash
  M512 gap, packed BF16x2 scaled accumulation may remove enough L1 promotion
  work to establish a stable lead while retaining BF16's FP32-sized exponent
  range.
- Parent: Flash M512, E5M2 combine, non-direct N-major scheduling, cluster2,
  L1/L2 EPW16/4, and L1/L2 stage4/4. Built-in event attribution measured
  roughly 220 us in L1 and 125 us in L2, confirming L1 as the remaining
  optimization target.
- Feature screen (max-rank, median-10): early weight-SF 353.793 us, shared
  weight-SF prefetch 346.818 us, adjacent scale domain 365.792 us, and global
  BF16x2 accumulation 345.602 us. Only BF16x2 was retained for confirmation.
- Interleaved median-20 confirmation:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | cluster2 + BF16x2 | 367.473, 345.953, 341.746 | 345.953 |
  | PR323 | 351.026, 355.378, 344.273 | 351.026 |

  The candidate leads the fresh PR323 median by 1.45%, despite one slow first
  observation. Exact eight-rank focused correctness passed at
  `calc_diff=0.0020 < 0.01`.
- Decision: retain this as the current Flash M512 H200 candidate, but do not
  encode an automatic selector until it passes broader precision and
  cross-seed confirmation. Cluster2 failed to launch for Pro M512, so this
  candidate is Flash-specific. H20 and M<128 behavior remain unchanged.
- Raw artifacts:
  `.../candidates/flash_m512_cluster2_feature_v1_*`,
  `.../candidates/flash_m512_cluster2_bf16_confirm_s101_n20/`, and
  `.../candidates/flash_m512_cluster2_bf16_correctness_smoke/`.

## Iteration 51: clean-rebuild and cross-seed Flash M512 validation

- A replacement 8x H200 allocation (`2963787`, `viking-prod-651`) was used
  after the prior allocation expired.  Before rebuilding with CUDA 13.2, the
  remote host runtime was restored byte-for-byte to the committed local
  source; this removed the temporary phase-specific BF16 experiment plumbing.
- A clean-JIT, three-observation median-20 confirmation at seed 101 produced:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | cluster2 + global BF16x2 | 344.348, 344.992, 341.151 | 344.348 |
  | PR323 | 344.735, 353.677, 345.408 | 345.408 |

  The candidate remains faster by 0.31%, but its margin is substantially
  smaller than the earlier 1.45% observation set.
- Independent median-20 route seeds did not establish a distribution-robust
  lead:

  | seed | ours us | PR323 us | gap |
  |---:|---:|---:|---:|
  | 7 | 346.013 | 361.611 | -4.31% |
  | 23 | 351.311 | 347.135 | +1.20% |
  | 509 | 351.951 | 349.983 | +0.56% |

  Combining those seeds with the seed-101 medians places both implementations
  at parity (ours is about 0.03% slower).  A seed-23 wave/stage/grid rescreen
  moved the identical parent from 351.311 to 339.965 us, while its best
  neighbor, L1 EPW8, was only another 0.26% faster.  The run-state movement is
  larger than the parameter signal, so neither neighbor was promoted.
- Decision: retain the candidate as useful seed-101 evidence, but withdraw the
  claim that Flash M512 is stably closed.  Do not encode an automatic selector
  until a candidate has margin across repeated runs and route seeds.
- Raw artifacts:
  `.../candidates/flash_m512_clean_rebuild_s101_n20/`,
  `.../candidates/flash_m512_crossseed_v2/`, and
  `.../candidates/flash_m512_seed23_neigh_v3_*`.

## Iteration 52: bounded Pro M512 parameter-space closure

- Clean-source phase attribution on the BN256 FP32 parent measured L1 at
  roughly 663--760 us and L2 at 355--465 us across ranks.  The two phases had
  complementary rank tails, so optimizing only one phase was insufficient.
- Existing legal knobs were screened without copying or implementing PR323's
  fused kernel.  The material observations were:

  | candidate | screen max-rank (us) | confirmed result / decision |
  |---|---:|---|
  | BN256 global BF16x2, EPW24/16 | 1118.6 | wave/stage neighbors all slower |
  | BN256 one-warp cleanup | 1107.944 | 1133.872 vs PR323 1091.168; reject |
  | L1-BN512/L2-BN256 global BF16, EPW16/16 | 1090.639 | 1107.452 vs PR323 1086.780; reject |
  | same tile, EPW16/12 | 1087.842 | 1132.843 vs PR323 1090.011; reject |
  | same tile, EPW12/12 | 1103.478 | 1119.740 vs PR323 1093.886; reject |
  | FP16x2 scaled accumulation, EPW16/12 | 1067.439 | `diff=nan`; hard reject |

- The FP16x2 point was not allowed to proceed on timing alone.  An 8-rank
  focused test of the same FP16 accumulator and BN512 calculation path
  returned `calc_diff=nan`, so it is numerically invalid under the unchanged
  `0.01` gate.
- All remaining existing parameter families were negative or invalid:
  direct scatter, global-BF16 wave/stage neighbors, adjacent/prefetch/early
  scale scheduling, L2 BN128/BN512, L2 BK256, phase-local frontend widths,
  and 112--130-SM grids.  A phase grid of L1/L2=96/112 SM entered a persistent
  scheduling hang and was terminated; same-grid reductions were legal but
  slower.  Compile-time guards correctly rejected BF16 with BK256 and BF16
  with four N64 consumer warpgroups.
- The closest repeatable historical result remains the temporary
  L1-BN512/BF16 plus L2-BN256/FP32 configuration at 1091.810 us versus PR323
  1085.427 us (+0.59%).  Its phase-specific host plumbing was deliberately
  removed, and the current clean source does not expose that mode.
- Decision: existing default-off parameter overrides do not provide a stable
  Pro M512 winner.  Further progress requires an explicitly approved,
  H200-only source experiment (for example phase-specific accumulation or a
  new tile), followed by correctness and interleaved confirmation.  No H20
  selector, generic default, or PR323 source was changed.
- Raw artifacts:
  `.../candidates/pro_m512_{cleanup,global_bf16,l1bn512,l2bn128,l2bk256,fp16acc}*`,
  `.../candidates/pro_m512_bn512_{grid,wave}*`, and the corresponding
  `*_confirm_s101_n20*` directories.

## Iteration 53: clean-node closure of the retained M128/M256 points

- The retained candidates were rerun from the clean host runtime on
  `viking-prod-651`, interleaving each implementation for three median-20
  observations at seed 101.
- Flash M128, E5M2 combine with non-direct N-major scheduling and EPW4:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | ours | 268.445, 285.053, 285.406 | 285.053 |
  | PR323 | 291.471, 386.380, 299.999 | 299.999 |

  Ours leads the median by 4.98%.  The PR323 second observation is an outlier,
  but ours also beats the other two individual PR323 observations.
- Pro M128, L1-BN512/L2-BN256 global-BF16 parent:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | ours | 865.032, 869.053, 876.086 | 869.053 |
  | PR323 | 899.903, 904.909, 911.365 | 904.909 |

  Ours leads by 3.96%.
- Pro M256, phase-local L1-BK256/L2-BK128 with 128 SMs:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | ours | 865.198, 862.303, 865.685 | 865.198 |
  | PR323 | 861.166, 872.349, 880.253 | 872.349 |

  Ours leads by 0.82%, reproducing the prior node's 0.80% margin.  Its focused
  correctness evidence remains `calc_diff=0.0020 < 0.01`.
- Decision: retain all three candidates.  The clean rebuild did not regress
  these points; only Flash and Pro M512 remain unclosed.
- Raw artifacts:
  `.../candidates/flash_m128_e4_confirm_s101_n20_v2/`,
  `.../candidates/pro_m128_bn512_clean_confirm_s101_n20_v2/`, and
  `.../candidates/pro_m256_l1bk256_sms128_clean_confirm_s101_n20_v2/`.

## Iteration 54: clean-node large-M closure and Flash M8192 retune

- Three interleaved median-20 observations on the clean H200 node confirmed
  the retained Pro L1-BN512/L2-BN256 global-BF16 parent at every M above 512:

  | M | ours median (us) | PR323 median (us) | gap |
  |---:|---:|---:|---:|
  | 1024 | 1555.068 | 1576.030 | -1.33% |
  | 2048 | 2324.749 | 2434.126 | -4.49% |
  | 4096 | 4117.554 | 4313.020 | -4.53% |
  | 8192 | 7749.587 | 7926.971 | -2.24% |

- The unmodified Flash default path was also rerun to separate retained
  defaults from opt-in candidate effects:

  | M | ours median (us) | PR323 median (us) | gap |
  |---:|---:|---:|---:|
  | 256 | 289.888 | 300.752 | -3.61% |
  | 1024 | 589.793 | 599.632 | -1.64% |
  | 2048 | 947.056 | 991.297 | -4.46% |
  | 4096 | 1740.819 | 1756.705 | -0.90% |
  | 8192 | 3344.431 | 3309.310 | +1.06% |

  M8192 no longer passed the gate under repeated clean-node measurement, so
  it was retuned instead of relying on the older single observation.
- A bounded Flash M8192 sweep retained E5M2 combine, non-direct N-major
  scheduling, stage3, and EPW32.  Its three median-20 observation maxima were
  3200.650/3177.536/3194.191 us versus PR323
  3301.935/3308.713/3302.333 us.  The median is 3194.191 versus 3302.333 us,
  a stable 3.27% lead.
- Exact eight-rank focused correctness for the retained E5M2/EPW32/stage3
  calculation path passed at `calc_diff=0.0020 < 0.01`.
- Decision: retain the Pro M1024--8192 parent, the default Flash M256 and
  M1024--4096 points, and the new explicit Flash M8192 candidate.  The only
  remaining performance gaps are Flash M512 and Pro M512.
- Raw artifacts:
  `.../candidates/pro_large_clean_confirm_s101_n20_v2/`,
  `.../candidates/flash_default_large_clean_s101_n20_v2/`,
  `.../candidates/flash_m8192_tune_v1_*`,
  `.../candidates/flash_m8192_e32_s3_n1_confirm_s101_n20_v1/`, and
  `.../candidates/flash_m8192_e32_s3_n1_correctness/`.

## Iteration 55: same-node original-baseline characterization below M128

- Hypothesis: the apparent Pro M64 regression against an older single-run
  number is allocation noise rather than a default-path source regression.
  Measure the exact pre-retune commit on the current H200 node before drawing
  any conclusion from cross-allocation numbers.
- Baseline source: detached worktree at `3552b62` (`[repair] Add SM90 dependent
  template qualifiers for CUDA 13.2`).  No candidate environment override was
  enabled.  The test used job `2963787` on `viking-prod-651`, seed 101, three
  median-20 observations, and the maximum returned latency across eight ranks.

  | M | observation maxima (us) | median us |
  |---:|---|---:|
  | 8 | 644.384, 640.864, 647.600 | 644.384 |
  | 16 | 775.392, 780.975, 777.057 | 777.057 |
  | 32 | 842.671, 835.969, 851.545 | 842.671 |
  | 64 | 867.887, 877.329, 868.240 | 868.240 |

- Decision: retain these values as the authoritative same-node denominator.
  Run the current default-off source under the identical protocol before
  accepting or rejecting the `M < 128` no-regression gate.  No source,
  selector, H20 tuning, or PR323 implementation changed in this iteration.
- Raw artifacts:
  `.../candidates/pro_small_baseline3552_same_node_full_v1/`.

## Iteration 56: same-node current-source check below M128

- Hypothesis: default-off experimental machinery in the current source does
  not materially regress the original Pro path below M128.
- Protocol: current source on job `2963787`, node `viking-prod-651`, no
  candidate environment overrides, seed 101, three median-20 observations,
  and the maximum returned latency across eight ranks.  This matches iteration
  55's original-source baseline.

  | M | current observation maxima (us) | current median us | baseline median us | gap |
  |---:|---|---:|---:|---:|
  | 16 | 780.767, 778.929, 777.024 | 778.929 | 777.057 | +0.24% |
  | 32 | 833.625, 838.465, 858.399 | 838.465 | 842.671 | -0.50% |
  | 64 | 868.591, 872.568, 865.585 | 868.591 | 868.240 | +0.04% |

- M16, M32, and M64 satisfy the no-regression gate within repeated-run noise.
  The outer summary command failed after all benchmark subprocesses ran because
  one M8 JSON line was interleaved with concurrent stdout and rejected by
  `jq`; its first two observation maxima were 650.832 and 648.363 us.  Preserve
  the raw log and recover or rerun M8 before closing the full small-M gate.
- No source, selector, H20 tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/pro_small_current_head_same_node_full_v1/`.

## Iteration 57: failed Pro M8 alternating A/B invocation

- Intended protocol: five paired, order-alternating observations of the exact
  `3552b62` baseline and current default source at Pro M8, seed 101 and
  median-20, to resolve iteration 56's 0.62% difference.
- Result: the remote login shell interpreted an embedded awk variable before
  the requested `bash -lc` payload and exited with `Illegal variable name`.
  No benchmark subprocess launched and no performance evidence was produced.
- Decision: fix only the command quoting and rerun the identical protocol.  No
  source, selector, H20 tuning, or PR323 implementation changed.

## Iteration 58: paired Pro M8 no-regression closure

- Hypothesis: iteration 56's 0.62% M8 difference is smaller than run-state
  noise and does not represent a material default-path regression.
- Protocol: exact `3552b62` baseline and current default source, Pro M8, seed
  101, median-20, eight ranks, five paired observations.  Execution order was
  alternated on every pair to reduce monotonic clock or node-state bias.

  | pair | first | baseline us | current us | current - baseline us |
  |---:|---|---:|---:|---:|
  | 1 | baseline | 679.2 | 648.1 | -31.1 |
  | 2 | current | 650.6 | 647.5 | -3.1 |
  | 3 | baseline | 643.4 | 649.2 | +5.8 |
  | 4 | current | 640.9 | 646.9 | +6.0 |
  | 5 | baseline | 646.8 | 670.4 | +23.6 |

- The baseline median is 646.8 us and the current median is 648.1 us, a
  +0.20% difference.  Both sides contain a single opposite-direction system
  outlier, and the central values overlap the earlier three-observation ranges.
- Decision: accept Pro M8 as no material regression.  Together with iteration
  56, the same-node `M < 128` Pro gate is closed.  No source, selector, H20
  tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/pro_m8_same_node_alternating_ab_v2/`.

## Iteration 59: paired Flash small-M no-regression closure

- Hypothesis: the current default-off source does not materially regress the
  original Flash path below M128; the older Flash M16 +0.41% comparison was
  cross-allocation noise.
- Protocol: exact `3552b62` baseline and current default source on job
  `2963787`, Flash M8/16/32/64, seed 101, median-20, eight ranks, three paired
  observations per M.  Baseline/current execution order alternated by
  observation.

  | M | baseline observations (us) | baseline median us | current observations (us) | current median us | gap |
  |---:|---|---:|---|---:|---:|
  | 8 | 223.8, 223.3, 223.1 | 223.3 | 218.0, 220.3, 214.3 | 218.0 | -2.37% |
  | 16 | 249.6, 252.0, 254.1 | 252.0 | 273.5, 242.6, 249.1 | 249.1 | -1.15% |
  | 32 | 257.9, 254.0, 259.6 | 257.9 | 256.5, 258.2, 255.0 | 256.5 | -0.54% |
  | 64 | 267.7, 287.6, 268.1 | 268.1 | 266.1, 268.8, 268.8 | 268.8 | +0.26% |

- Flash M8/M16/M32 are faster than the original source.  M64 differs by only
  +0.26%, within the repeated-run noise demonstrated by the 287.6-us baseline
  outlier, and is accepted as no material regression.
- Decision: the same-node `M < 128` no-regression gate is now closed for both
  Flash and Pro.  No source, selector, H20 tuning, or PR323 implementation
  changed.
- Raw artifacts:
  `.../candidates/flash_small_same_node_alternating_ab_v1/`.

## Iteration 60: Flash M512 phase frontend and BN128 screen

- Hypothesis: the retained cluster2/BF16 parent may reduce its dominant L1
  time by expanding the phase-local dispatch frontend, or improve tail
  utilization by using a narrower N tile in one phase.  These are existing
  default-off controls and require no new kernel structure.
- Parent: Flash M512, E5M2 combine, non-direct N-major scheduling, cluster2,
  global BF16x2 accumulation, L1/L2 EPW16/4 and stage4/4.
- Protocol: job `2963787`, node `viking-prod-651`, seed 101, median-10, maximum
  returned latency across eight ranks.  Screen only our implementation before
  spending paired PR323 observations on a survivor.

  | candidate | max-rank us | versus parent |
  |---|---:|---:|
  | parent | 339.3 | -- |
  | L1 dispatch warps 4 | 356.9 | +5.19% |
  | L2 dispatch warps 4 | 378.6 | +11.58% |
  | L1/L2 dispatch warps 4/4 | 346.9 | +2.24% |
  | L1 BN128 | 415.0 | +22.31% |
  | L2 BN128 | 414.3 | +22.10% |
  | L1/L2 BN128/128 | 458.0 | +34.98% |

- Decision: reject expanded phase frontends and BN128 for Flash M512.  None
  beats the same-state parent, so no PR323 confirmation is warranted.  No
  source, selector, H20 tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_frontend_tile_screen_v1/`.

## Iteration 61: interrupted Flash M512 cross-seed L1-wave screen

- Intended protocol: compare L1 EPW8/12/16 on route seeds 7/23/101/509 for
  the retained cluster2/BF16 parent, using median-20 and maximum latency across
  eight ranks.
- Result: seed 7 EPW8 completed at 342.8 us.  EPW12 then failed the existing
  legality condition because Flash has 32 local experts and the wave size must
  divide that count.  The batch stopped before EPW16 or the other seeds ran.
- Decision: remove only the illegal EPW12 point and rerun EPW8 versus EPW16
  under the same cross-seed protocol.  No source, selector, H20 tuning, or
  PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l1wave_crossseed_screen_v1/`.

## Iteration 62: Flash M512 legal L1-wave cross-seed screen

- Hypothesis: L1 EPW8 is less sensitive than the retained EPW16 parent to
  route-dependent tail imbalance at Flash M512.
- Protocol: retained cluster2/global-BF16/E5M2 parent, job `2963787`, seeds
  7/23/101/509, median-20, maximum returned latency across eight ranks.  EPW8
  and EPW16 execution order alternated by seed; all other settings were held
  fixed.

  | seed | L1 EPW8 us | L1 EPW16 us | EPW8 difference |
  |---:|---:|---:|---:|
  | 7 | 343.8 | 344.0 | -0.06% |
  | 23 | 338.1 | 345.3 | -2.09% |
  | 101 | 343.1 | 348.9 | -1.66% |
  | 509 | 345.1 | 352.0 | -1.96% |

- EPW8 wins all four route seeds.  Its cross-seed center is 343.45 us versus
  347.10 us for EPW16, a 1.05% improvement, and its range is also narrower.
- Decision: promote L1 EPW8 to paired PR323 cross-seed confirmation.  Keep L2
  at EPW4.  No source, selector, H20 tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l1wave_crossseed_screen_v2/`.

## Iteration 63: Flash M512 EPW8 cross-seed PR323 confirmation

- Hypothesis: the cross-seed L1 EPW8 improvement is large enough to turn the
  retained Flash M512 parent into a route-robust PR323 winner.
- Protocol: seeds 7/23/101/509, three median-20 observations per
  implementation and seed, maximum returned latency across eight ranks.
  Ours and unmodified PR323 alternated order on every observation.

  | seed | ours observations (us) | ours median us | PR323 observations (us) | PR323 median us | gap |
  |---:|---|---:|---|---:|---:|
  | 7 | 350.9, 346.8, 341.4 | 346.8 | 347.3, 344.4, 347.1 | 347.1 | -0.09% |
  | 23 | 343.6, 353.1, 341.4 | 343.6 | 346.1, 345.1, 348.4 | 346.1 | -0.72% |
  | 101 | 360.6, 336.9, 343.6 | 343.6 | 360.0, 349.5, 344.7 | 349.5 | -1.69% |
  | 509 | 347.5, 358.5, 341.5 | 347.5 | 347.1, 346.0, 344.8 | 346.0 | +0.43% |

- Ours wins three of four seeds.  The median of the four per-seed medians is
  345.2 us versus 346.6 us for PR323, a 0.40% aggregate lead.  However, seed
  509 remains 0.43% slower and individual ours observations still have larger
  upper tails.
- Decision: EPW8 is the best Flash M512 candidate so far, but do not call the
  strict per-seed gate closed.  Use it as the parent for one final legal L2
  wave/tail screen; reject any neighbor that does not improve seed 509 without
  losing the other seeds.  No source, H20 tuning, or PR323 implementation
  changed.
- Raw artifacts:
  `.../candidates/flash_m512_l1e8_pr_confirm_s{7,23,101,509}_v1/`.

## Iteration 64: Flash M512 L2 large-wave tail screen

- Hypothesis: after moving L1 to EPW8, a larger L2 expert wave may remove the
  remaining seed-509 tail without materially hurting the seed-101 winner.
- Protocol: retained Flash M512 cluster2/BF16/E5M2 candidate with L1 EPW8;
  compare L2 EPW4/16/32 at seeds 101 and 509, median-20, maximum latency across
  eight ranks.  Candidate order was reversed on seed 509.

  | seed | L2 EPW4 us | L2 EPW16 us | L2 EPW32 us |
  |---:|---:|---:|---:|
  | 101 | 345.2 | 350.0 | 346.2 |
  | 509 | 349.3 | 342.6 | 338.5 |

- EPW16 is not competitive.  EPW32 costs 1.0 us on seed 101 but improves seed
  509 by 10.8 us, turning the only PR323-losing route seed into a promising
  candidate while preserving enough of seed 101's prior margin.
- Decision: reject EPW16.  Extend the EPW4-versus-EPW32 screen to seeds 7 and
  23 before full PR323 confirmation.  No source, selector, H20 tuning, or
  PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l2wave_tail_screen_v1/`.

## Iteration 65: Flash M512 L2 EPW32 remaining-seed screen

- Hypothesis: L2 EPW32's seed-509 tail improvement extends to seed 7/23 and
  leaves a fixed configuration that can beat PR323 across all four routes.
- Protocol: same L1-EPW8 parent and median-20/max-rank method as iteration 64;
  compare L2 EPW4 and EPW32 at seeds 7 and 23, reversing order on seed 23.

  | seed | L2 EPW4 us | L2 EPW32 us | EPW32 difference |
  |---:|---:|---:|---:|
  | 7 | 348.3 | 339.3 | -2.58% |
  | 23 | 337.6 | 345.7 | +2.40% |

- EPW32 materially improves seed 7 and 509, modestly regresses seed 101, and
  materially regresses seed 23.  Even the seed-23 screen remains close to the
  prior same-seed PR323 median of 346.1 us, so a direct interleaved verdict is
  still necessary.
- Decision: run the fixed L1/L2 EPW8/32 candidate against PR323 for all four
  route seeds.  Do not select it from cross-run comparisons alone.  No source,
  selector, H20 tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l2wave_tail_screen_v2/`.

## Iteration 66: Flash M512 EPW8/32 cross-seed PR323 verdict

- Hypothesis: the fixed L1/L2 EPW8/32 configuration beats PR323 on all four
  route seeds by trading a small seed-23 screen regression for much better
  seed-7/509 tails.
- Protocol: seeds 7/23/101/509, three median-20 observations per
  implementation and seed, maximum latency across eight ranks, with ours and
  PR323 alternating order on every observation.

  | seed | ours observations (us) | ours median us | PR323 observations (us) | PR323 median us | gap |
  |---:|---|---:|---|---:|---:|
  | 7 | 349.9, 363.3, 344.4 | 349.9 | 363.9, 346.2, 346.9 | 346.9 | +0.86% |
  | 23 | 339.5, 337.5, 339.4 | 339.4 | 342.3, 346.0, 352.4 | 346.0 | -1.91% |
  | 101 | 342.6, 340.4, 350.3 | 342.6 | 343.7, 347.3, 345.2 | 345.2 | -0.75% |
  | 509 | 338.3, 340.7, 347.2 | 340.7 | 348.1, 349.4, 352.3 | 349.4 | -2.49% |

- EPW8/32 wins seeds 23/101/509 with useful margin, but loses seed 7 by
  0.86% because of a slow second observation.  It therefore does not satisfy
  the strict all-seed gate even though its aggregate result is stronger than
  EPW8/4.
- Decision: do not promote EPW32 yet.  Screen the intermediate legal L2 EPW8
  as the final fixed-wave compromise; it must preserve the three current wins
  and remove the seed-7 loss before confirmation.  No source, selector, H20
  tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l1e8_l2e32_pr_confirm_s{7,23,101,509}_v1/`.

## Iteration 67: Flash M512 L2 EPW8 compromise screen

- Hypothesis: L2 EPW8 balances EPW4's seed-509 weakness and EPW32's seed-7
  variability, yielding one fixed configuration with enough margin on all
  four route seeds.
- Protocol: retained cluster2/BF16/E5M2 configuration with L1/L2 EPW8/8;
  seeds 7/23/101/509, median-20, maximum latency across eight ranks.

  | seed | ours max-rank us | latest paired PR323 median us | indicative gap |
  |---:|---:|---:|---:|
  | 7 | 338.7 | 346.9 | -2.36% |
  | 23 | 345.5 | 346.0 | -0.14% |
  | 101 | 343.2 | 345.2 | -0.58% |
  | 509 | 348.8 | 349.4 | -0.17% |

- A single EPW8 observation is below the latest directly paired PR323 median
  at every route seed, unlike EPW4 or EPW32.  The margins at seeds 23 and 509
  are small, so cross-run comparison alone is not selector-grade evidence.
- Decision: promote L1/L2 EPW8/8 to the final four-seed interleaved PR323
  confirmation.  No source, selector, H20 tuning, or PR323 implementation
  changed.
- Raw artifacts:
  `.../candidates/flash_m512_l2epw8_crossseed_screen_v1/`.

## Iteration 68: Flash M512 EPW8/8 cross-seed PR323 verdict

- Hypothesis: the fixed L1/L2 EPW8/8 compromise beats PR323 on all four route
  seeds and removes the complementary EPW4/EPW32 failures.
- Protocol: seeds 7/23/101/509, three median-20 observations per
  implementation and seed, maximum latency across eight ranks, alternating
  ours and PR323 on every observation.

  | seed | ours observations (us) | ours median us | PR323 observations (us) | PR323 median us | gap |
  |---:|---|---:|---|---:|---:|
  | 7 | 344.1, 347.2, 351.1 | 347.2 | 342.6, 342.5, 353.3 | 342.6 | +1.34% |
  | 23 | 365.5, 345.5, 340.5 | 345.5 | 342.4, 352.5, 344.0 | 344.0 | +0.44% |
  | 101 | 340.4, 343.0, 354.3 | 343.0 | 345.5, 341.1, 350.2 | 345.5 | -0.72% |
  | 509 | 340.8, 343.2, 335.8 | 340.8 | 349.1, 343.4, 354.7 | 349.1 | -2.38% |

- EPW8/8 wins seeds 101/509 but loses seeds 7/23.  It is therefore less
  robust than L1/L2 EPW8/4, which won seeds 7/23/101, lost only seed 509 by
  0.43%, and led the four-seed center by 0.40%.
- Decision: reject EPW8/8 and retain EPW8/4 as the best fixed Flash M512
  candidate.  It closes the official seed-101 benchmark point by 1.69%, but
  no tested fixed L2 wave wins every route seed; retain that limitation in the
  final report rather than claiming distribution-independent performance.
  No source, selector, H20 tuning, or PR323 implementation changed.
- Raw artifacts:
  `.../candidates/flash_m512_l1e8_l2e8_pr_confirm_s{7,23,101,509}_v1/`.

## Iteration 69: Flash M512 EPW8/4 focused correctness

- Exact candidate controls: E5M2 combine, non-direct N-major scheduling,
  cluster2, global BF16x2 scaled accumulation, L1/L2 EPW8/4 and stage4/4.
- Protocol: eight H200 ranks, unchanged `calc_diff` tolerance 0.01, focused
  top-k6/M512 heuristic scenario.
- Result: `diff=0.0020 < 0.01`; the scenario passed on all eight ranks and the
  test process exited zero.
- Decision: retain EPW8/4 as the numerically valid Flash M512 benchmark
  winner.  This test changes only expert-wave scheduling relative to the prior
  correct parent and does not relax precision or tolerance.  No source,
  selector, H20 tuning, or PR323 implementation changed.
- Raw artifact:
  `.../candidates/flash_m512_l1e8_l2e4_correctness_v1/correctness.log`.

## Iteration 70: Flash M512 exact-shape correctness

- Hypothesis: the retained EPW8/4 candidate remains within the unchanged
  numerical tolerance at the actual Flash production dimensions, rather than
  only the smaller heuristic-branch test shape.
- Protocol: eight H200 ranks; H4096, I2048, E256, top-k6, M512; E5M2 combine,
  cluster2, global BF16x2 accumulation, L1/L2 EPW8/4 and stage4/4.  Compare
  directly against the existing FP32/BF16 reference with `calc_diff < 0.01`.
- Result: `diff=0.0021 < 0.01`, process exit zero.
- Decision: exact Flash M512 shape correctness is confirmed without tolerance
  relaxation.  Retain EPW8/4 as the final Flash M512 candidate.  No source,
  selector, H20 tuning, or PR323 implementation changed.
- Raw artifact:
  `.../candidates/flash_m512_l1e8_l2e4_exact_shape_correctness_v1/correctness.log`.

## Iteration 71 — Conservative automatic H200 FP8 selector smoke

- Hypothesis: an H200-only, exact-shape/exact-M selector can reproduce the previously verified explicit FP8 candidate configurations while leaving M512 and all unmatched workloads on the existing generic path.
- Changes:
  - Added H200 device detection and an optional `DG_SM90_MOE_H200_POLICY` debug override.
  - Added exact Flash/Pro workload policies for the retained M points only.
  - Applied policy defaults to base and phase-specific FP8 configuration, with explicit environment variables retaining precedence.
  - Kept the already validated global BF16 Pro candidates; added no new phase-specific BF16 path.
- Protocol: H200 job 2968183 (`viking-prod-651`), no tuning environment variables, `DG_PRINT_CONFIGS=1`, one timing iteration for compile/config smoke. Tested Flash M={128,512,8192} and Pro M={128,256,512,1024}.
- Evidence:
  - Flash M8192 selected BM64/BN256/BK128, cluster1, direct0, N-major1, cleanup0, L1/L2 EPW32 and stage3, FP8 combine enabled, BF16 scaled accumulation disabled.
  - Pro M128 selected L1 BN512/EPW16/stage2 and L2 BN256/EPW16/stage3, direct0, N-major1, cleanup0, FP8 combine enabled, global BF16 scaled accumulation enabled.
  - Pro M256 selected L1 BN256/BK256/EPW8/stage2/SMS128 and L2 BN256/BK128/EPW48/stage3/SMS128, direct0, N-major1, cleanup0, FP8 combine enabled, BF16 scaled accumulation disabled.
  - Flash M512 and Pro M512 printed only the generic base configuration and no H200 policy, confirming conservative fallthrough.
  - All smoke runs completed successfully. Timing is compile/config smoke only and is not used as the final performance verdict.
- Decision: retain the selector implementation and proceed to multi-seed numerical validation plus interleaved performance verification.
- Artifacts:
  - `$ROOT/candidates/h200_auto_selector_flash_smoke_v1`
  - `$ROOT/candidates/h200_auto_selector_pro_smoke_v1`
  - `$ROOT/jit/h200_auto_selector_smoke_v1`

## Iteration 72 — Exact Pro global-BF16 numerical-gate smoke

- Hypothesis: the actual Pro-shape validation harness can compare the retained
  E5M2 configuration with FP32 versus global packed-BF16 scaled accumulation
  on identical inputs and against a distributed FP32 golden reference.
- Protocol: H200 job `2968183`, eight ranks, H7168/I3072/E384/top-k6, M128,
  seed 101.  Both kernel runs used the same FP8 inputs, weights, routing, and
  E5M2 combine path; only `DG_SM90_MOE_BF16_SCALED_ACCUM` changed.  The
  distributed reference computed each rank's local experts and reduce-scattered
  the BF16 contribution slots.  The unchanged gate was finite output and
  `calc_diff < 0.01` on every rank for FP32-vs-golden, BF16-vs-golden, and
  BF16-vs-FP32.
- Result: all eight ranks passed.  Worst-rank differences were
  FP32-vs-golden `0.001984`, BF16-vs-golden `0.002106`, and BF16-vs-FP32
  `0.001847`; all outputs were finite.  The largest absolute BF16-vs-FP32
  element difference was 144, recorded for visibility but not used as a
  scale-independent acceptance metric.
- Decision: the harness and M128 global-BF16 path are valid.  Proceed to the
  full four-seed by five-M exact-Pro campaign.
- Raw artifact:
  `$ROOT/candidates/h200_pro_bf16_gate_smoke_v1/validation.log`.

## Iteration 73 — Exact Pro global-BF16 four-seed numerical gate

- Hypothesis: the retained global packed-BF16 scaled-accumulation candidates
  remain numerically acceptable across the complete selected Pro M set and
  route/weight seeds, not only the seed-101 smoke.
- Protocol: H200 job `2968183`, eight ranks, exact Pro H7168/I3072/E384/top-k6,
  M={128,1024,2048,4096,8192}, seeds={7,23,101,509}.  For every one of the
  20 cases, run identical FP8 inputs/weights/routes with FP32 accumulation and
  global BF16x2 accumulation while holding E5M2 combine fixed, then compare
  both outputs to the distributed FP32 golden and directly to each other.
  Require finite output and `calc_diff < 0.01` independently on every rank.
- Result: all 20 cases and all 160 per-rank outputs passed; no NaN or Inf.
  Worst values across the campaign were:
  - FP32-vs-golden: `0.00198835` at seed 23, M128.
  - BF16-vs-golden: `0.00210961` at seed 23, M128.
  - BF16-vs-FP32: `0.00184732` at seed 101, M128.
  - Maximum absolute BF16-vs-FP32 element delta: 192 at seed 23, M8192
    (reported only for scale context; the normalized acceptance metric passed).
- Decision: the previously validated global-BF16 Pro rows clear the full
  numerical gate with substantial margin.  Keep them in the conservative H200
  selector and proceed to automatic-selector performance confirmation.
- Raw artifact:
  `$ROOT/candidates/h200_pro_bf16_gate_4seed_5m_v1/validation.log`.

## Iteration 74 — Automatic H200 selector formal performance matrix

- Hypothesis: the conservative automatic selector reproduces the retained
  explicit-candidate wins at every in-scope M while M512 remains on the generic
  path.
- Protocol: H200 job `2968183`, no `DG_SM90_MOE_*` tuning variables, seed 101,
  three order-alternating observations per implementation, rank-local
  median-20, and the median of the three eight-rank maxima.  Compare the
  automatic selector with unmodified PR323 at Flash/Pro
  M={128,256,1024,2048,4096,8192}; record M512 ours-only.
- Results:

  | shape | M | ours median us | PR323 median us | gap |
  |---|---:|---:|---:|---:|
  | Flash | 128 | 271.7 | 295.0 | -7.90% |
  | Flash | 256 | 289.1 | 302.3 | -4.37% |
  | Flash | 512 | 413.8 | not gated | — |
  | Flash | 1024 | 592.8 | 605.5 | -2.11% |
  | Flash | 2048 | 950.7 | 999.6 | -4.89% |
  | Flash | 4096 | 1745.5 | 1760.3 | -0.84% |
  | Flash | 8192 | 3192.2 | 3307.2 | -3.48% |
  | Pro | 128 | 880.9 | 899.8 | -2.10% |
  | Pro | 256 | 866.4 | 862.5 | +0.45% |
  | Pro | 512 | 1231.3 | not gated | — |
  | Pro | 1024 | 1551.9 | 1581.6 | -1.88% |
  | Pro | 2048 | 2323.7 | 2435.0 | -4.57% |
  | Pro | 4096 | 4098.5 | 4296.3 | -4.60% |
  | Pro | 8192 | 7715.6 | 7897.0 | -2.30% |

- Validation: the strict parser accepted all 78 leaf runs, each with eight
  ranks, 20 timed samples per rank, three observation IDs, zero exit status,
  and identical routes between paired implementations.
- Decision: 11/12 in-scope points beat PR323.  Do not call the performance gate
  closed because Pro M256 is 0.45% slower in this formal run, despite its prior
  0.82% win.  Preserve every other automatic row and perform a focused paired
  Pro-M256 remeasurement/tuning step only; M512 remains excluded.
- Raw artifacts:
  - `$ROOT/candidates/h200_auto_selector_final_v1/report.md`
  - `$ROOT/candidates/h200_auto_selector_final_v1/summary.csv`
  - `$ROOT/candidates/h200_auto_selector_final_v1/logs/`

## Iteration 75 — Failed Pro M256 five-observation invocation

- Intended protocol: rerun the unchanged automatic Pro-M256 selector against
  PR323 for five order-alternating median-20 observations.
- Result: shell redirection attempted to open the candidate driver log before
  its parent directory existed, so the command exited immediately with
  `No such file or directory`.  No benchmark subprocess or GPU kernel ran.
- Decision: create the candidate directory before redirection and rerun the
  identical protocol.  No source, selector, H20 tuning, or PR323 code changed.

## Iteration 76 — Unchanged Pro M256 five-observation remeasurement

- Hypothesis: the formal matrix's +0.45% Pro-M256 result may be a three-run
  ordering fluctuation because the identical configuration previously led
  PR323 by about 0.8%.
- Protocol: unchanged automatic selector, no tuning environment variables,
  seed 101, five order-alternating observations, rank-local median-20, and
  maximum latency across eight ranks.
- Result:

  | implementation | observation maxima (us) | median us |
  |---|---|---:|
  | ours | 881.879, 869.752, 865.712, 863.770, 864.007 | 865.712 |
  | PR323 | 870.522, 869.952, 859.584, 861.502, 863.407 | 863.407 |

  Ours is 0.27% slower.  The result narrows the formal +0.45% gap but does not
  close the strict faster-than-PR323 gate.
- Decision: measurement noise alone is not sufficient evidence to accept the
  current row.  Keep all other selector rows fixed and perform a bounded
  Pro-M256 phase-grid screen using only existing H200 experiment controls.
- Raw artifacts:
  `$ROOT/candidates/h200_auto_pro_m256_remeasure_v1/`.

## Iteration 77 — Failed Pro M256 phase-grid screen invocation

- Intended protocol: two median-20 observations for nine bounded L1/L2 SM-grid
  combinations around the retained 128/128 point.
- Result: as in iteration 75, driver-log redirection preceded creation of the
  per-candidate directory.  The first shell redirection failed immediately;
  no benchmark subprocess or GPU kernel ran.
- Decision: explicitly create each candidate directory inside the loop and
  rerun the unchanged screen.  No source or selector setting changed.

## Iteration 78 — Aborted Pro M256 asymmetric phase-grid screen

- Hypothesis: keeping the L1-dominant phase at 128 SMs while changing only the
  L2 grid could recover the remaining few microseconds at Pro M256.
- Protocol: intended two median-20 observations for nine bounded L1/L2 grids,
  beginning with 128/128 control and 128/124.
- Result: the 128/128 control completed at 889.342 and 861.374 us.  The first
  128/124 process emitted all rank run configurations but did not reach the
  profiler after more than four minutes.  This reproduces the prior unsafe
  asymmetric-grid behavior.  The batch was interrupted; later combinations
  did not run.
- Decision: reject asymmetric L1/L2 SM grids as operationally unsafe and do
  not encode one in the selector.  Clean any surviving processes, verify GPU
  health, and restrict further M256 work to symmetric grids or already-safe
  scheduling controls.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_phasegrid_128_{128,124}_v1/`.

## Iteration 79 — Pro M256 safe symmetric-grid closure

- Hypothesis: an untested symmetric grid immediately adjacent to 128 SMs may
  improve wave alignment without the hang risk of unequal phase grids.
- Protocol: seed 101, one median-20/max-rank screen each, ordered as 128
  control, 127, 129, 126, and a final 128 control.  All non-grid settings were
  the retained automatic M256 configuration.
- Results:

  | symmetric L1/L2 grid | max-rank us |
  |---:|---:|
  | 128 (opening control) | 896.558 |
  | 127 | 912.970 |
  | 129 | 899.057 |
  | 126 | 876.944 |
  | 128 (closing control) | 869.246 |

- Decision: reject 126/127/129; each is slower than the closing 128-SM
  control, while the opening control demonstrates the run-state noise that
  makes single minima unsafe to promote.  Together with earlier 124/130/132
  results, 128 remains the best safe symmetric grid.  No selector change.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_symmetric_sms_*_v1/`.

## Iteration 80 — Pro M256 L2-BK256 screen

- Hypothesis: extending the existing two-plane BK256 implementation from L1
  to L2 could halve L2 pipeline-block advances while keeping the same FP32
  accumulation and E5M2 combine path.
- Protocol: 128-SM retained M256 configuration, seed 101, one median-20/max-rank
  observation in control / L2-BK256-stage2 / control order.
- Results: the opening control was 872.749 us, L2 BK256/stage2 was 898.086 us,
  and the closing control was 888.846 us.
- Decision: reject L2 BK256.  Its reduced pipeline depth and more complex
  per-64 scale grouping outweigh fewer block advances.  Keep L2 BK128/stage3;
  no selector or precision change.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_l2bk256_screen_*_v1/`.

## Iteration 81 — Pro M256 128-SM expert-wave interaction screen

- Hypothesis: expert waves selected before the 128-SM grid change may need
  retuning at the final grid size.
- Protocol: fixed L1/L2 BK256/BK128, stages2/3, FP32 accumulation, E5M2
  combine, and symmetric 128-SM grids.  Run two median-20/max-rank observations
  for current EPW8/48, then 8/24, 12/24, 12/48, and a closing 8/48 control.
- Results:

  | L1/L2 EPW | observation maxima (us) | center us |
  |---|---|---:|
  | 8/48 opening control | 862.078, 863.917 | 862.998 |
  | 8/24 | 862.718, 864.127 | 863.423 |
  | 12/24 | 865.967, 870.430 | 868.199 |
  | 12/48 | 868.670, 875.215 | 871.943 |
  | 8/48 closing control | 948.720, 896.749 | 922.735 |

- Decision: reject all three neighbors.  Before the late run-state degradation,
  8/24 was already slightly slower than the stable opening control, and both
  L1-EPW12 variants regressed materially.  Retain EPW8/48 and do not promote
  the anomalous closing-control timings as a source effect.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_sms128_wave_*_v1/`.

## Iteration 82 — Pro M256 final-grid N-major/stage interaction

- Hypothesis: the pre-BK256 near-parity N-major0/stage4 result may interact
  favorably with the final L1-BK256 and 128-SM configuration.
- Protocol: two median-20/max-rank observations in current N-major1/L2-stage3,
  N-major0/stage3, N-major0/stage4, N-major1/stage4, and closing-control order.
- Results:

  | N-major / L2 stage | observation maxima (us) | center us |
  |---|---|---:|
  | 1 / 3 opening control | 882.365, 865.926 | 874.146 |
  | 0 / 3 | 878.727, 876.710 | 877.719 |
  | 0 / 4 | 890.192, 871.366 | 880.779 |
  | 1 / 4 | 877.582, 887.862 | 882.722 |
  | 1 / 3 closing control | 871.918, 860.447 | 866.183 |

- Decision: reject N-major0 and L2 stage4.  Every neighbor is slower than the
  closing current control, and the opening/closing movement again argues
  against promoting isolated minima.  Keep N-major1/L2-stage3 unchanged.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_sms128_sched_*_v1/`.

## Iteration 83 — Final Pro M256 in-kernel phase profile

- Goal: replace further blind parameter sweeps with direct timing evidence for
  the retained L1-BK256/L2-BK128, 128-SM configuration.
- Protocol: enable the existing `DG_SM90_MOE_PHASE_PROFILE` counters for one
  seed-101 median-10 run.  This instrumentation is diagnostic only and is not
  selected automatically.
- Result: maximum returned rank latency was 866.247 us.  Across ranks, the
  representative per-record averages were roughly 420k--445k cycles for the
  math loop, 128k--174k for dispatch total, 56k--62k for dispatch pull,
  21k--41k for the combine barrier, about 36k for each GEMM-core block, and
  only about 2.6k for either L1 or L2 epilogue records.  Math-loop maxima
  reached about 1.28M cycles on the slow ranks, while combine-reduce itself was
  only about 2.7k cycles.
- Decision: epilogue arithmetic is not the remaining M256 limiter.  The gap is
  dominated by math-loop/task-tail and dispatch/combine synchronization;
  previous grid/wave/direct/cleanup experiments already target those regions.
  Avoid adding epilogue complexity and retain the current precision path.
- Raw artifact:
  `$ROOT/candidates/pro_m256_final_phase_profile_v1/`.

## Iteration 84 — Final Pro M256 one-warp-cleanup check

- Hypothesis: after L1 BK256 and the 128-SM grid, one-warp workspace cleanup
  might reduce the dispatch/cleanup tail exposed by iteration 83.
- Protocol: two median-20/max-rank observations in control, one-warp cleanup,
  and closing-control order; all other automatic M256 settings unchanged.
- Results: opening control 869.606/865.815 us (center 867.711), one-warp
  cleanup 860.938/874.976 us (center 867.957), and closing control
  875.479/869.255 us (center 872.367).
- Decision: reject one-warp cleanup.  It is effectively flat versus the opening
  control and does not provide the required stable margin.  Retain the
  existing two-warp cleanup path and make no selector change.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_final_cleanup_*_v1/`.

## Iteration 85 — Pro M256 low symmetric-grid screen

- Hypothesis: symmetric 96 or 112 SM grids may trade some parallelism for exact
  divisibility of the 1152 L1 and/or 1344 L2 N-tile tasks, reducing the final
  scheduler tail.  M256 had previously only screened 124--132 SMs.
- Protocol: one seed-101 median-20/max-rank observation in 128 control, 96,
  112, 120, and closing-128 order.  L1 and L2 always used the same grid.
- Results: 128 opening 874.025 us, 96 881.662 us, 112 875.502 us, 120
  875.454 us, and 128 closing 865.535 us.
- Decision: reject 96/112/120.  Exact tile-count divisibility does not offset
  lower active-SM parallelism; all are slower than the closing 128 control and
  none improves even the noisier opening control.  Retain symmetric 128 SMs.
- Raw artifacts:
  `$ROOT/candidates/pro_m256_low_symmetric_sms_*_v1/`.

## Iteration 86 — Failed asymmetric-grid equal-control invocation

- Intended protocol: force a fresh JIT of the dispatch-counter-grid change and
  run the existing H200 Pro M256 configuration with equal 128/128 phase grids,
  seed 101, and one max-rank median-5 observation.
- Result: the remote worktree did not contain
  `scripts/run_h200_fp8_candidate.sh`, so the driver exited before launching
  Python or any GPU kernel.  The follow-up summary naturally found no log and
  printed a zero placeholder; it is not a timing result.
- Decision: sync the unchanged benchmark driver and matrix runner from the
  local worktree, then rerun the identical control.  No selector, H20 tuning,
  PR323 code, or kernel source changed because of this invocation failure.

## Iteration 87 — H200 node NVLS initialization failure

- Intended protocol: rerun the unchanged equal-grid Pro M256 control after
  syncing the existing candidate driver and matrix runner.
- Result: all eight processes started, but NCCL process-group initialization
  failed before benchmark setup or CUDA-kernel JIT.  Node `viking-prod-303`
  could not bind a 2 MiB NVLink SHARP multicast allocation and returned CUDA
  error 401 (`the operation cannot be performed in the present state`).  The
  driver exited 1 and produced no timing result.
- Decision: follow NCCL's explicit recovery path and rerun with
  `NCCL_NVLS_ENABLE=0`.  This affects only NCCL's process-group transport used
  by the harness; it does not select or modify the MegaMoE kernel.  No source,
  selector, H20 tuning, or PR323 implementation changed.

## Iteration 88 — Dispatch-grid equal-control JIT smoke

- Hypothesis: separating the dispatch producer count from the phase launch
  grid preserves the existing equal-grid behavior when both values are 128.
- Protocol: new H200 job `2980566` on `viking-prod-303`, eight ranks, exact
  Pro H7168/I3072/E384/top-k6 at M256, seed 101, explicit L1/L2 grids 128/128,
  a fresh candidate JIT cache, and one max-rank median-5 observation.  Set
  `NCCL_NVLS_ENABLE=0` only to bypass this node's broken NCCL multicast
  initialization; the MegaMoE kernel configuration was unchanged.
- Result: both new JIT kernels compiled and the run exited 0.  The eight
  rank-local medians ranged from 843.139 to 862.625 us; max-rank latency was
  862.625 us.  The benchmark's setup/correctness path completed without an
  assertion or non-finite failure.
- Decision: the same-grid compatibility check passes and the control is in the
  recent 128/128 timing band.  Proceed to the bounded 128/112 and 128/132
  asymmetric Linear2 screens; do not change the selector yet.
- Raw artifact:
  `$ROOT/candidates/pro_m256_dispatchgrid_equal_128_v1/`.
