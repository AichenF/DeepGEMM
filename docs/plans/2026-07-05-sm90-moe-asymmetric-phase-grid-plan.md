# SM90 MegaMoE Asymmetric Phase-Grid Implementation Plan

## 1. Separate scheduler launch and producer counts

- Add `kNumDispatchSMs` to `MegaMoEScheduler` after `kNumRanks`, defaulting
  to `kNumSMs`.
- Add a compile-time positive-value assertion.
- Replace only the receive-counter completion target with
  `kNumDispatchSMs * kNumRanks`.
- Leave task strides and all other `kNumSMs` uses unchanged.

Files:

- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`

Evidence: source audit shows exactly one behavioral use of the new parameter.

## 2. Thread the producer count through SM90 FP8 JIT generation

- Add `kNumDispatchSMs` to the SM90 FP8 kernel template immediately after
  `kNumRanks`.
- Pass it into the scheduler instantiation.
- Add `dispatch_num_sms` to the host runtime's templated arguments.
- Emit that value in generated JIT source.
- Derive it from `l1_num_sms` for both split-phase launches.

Files:

- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`

Evidence: equal-grid generated templates use equal launch/dispatch counts;
asymmetric Linear2 templates use its own grid plus Linear1's dispatch count.

## 3. Run static and build checks

- Compile and run `tests/sm90_moe_h200_policy_test.cpp` with warnings as
  errors.
- Run `git diff --check`.
- Sync the changed files to the active H200 worktree.
- Force-build the CUDA/Python extension with the pinned CUDA 13.2 toolchain.

Evidence: host test and full extension build both pass.

## 4. Establish equal-grid safety

- Clear the JIT cache for the focused run.
- Run the current automatic Pro M256 configuration with Linear1/Linear2 at
  128/128.
- Require completion, finite output, and the existing numerical threshold.
- Compare timing with the recent equal-grid band to catch a material change.

Evidence: structured benchmark log and iteration record.

## 5. Screen asymmetric Linear2 grids

- Hold all selected Pro M256 parameters and Linear1 grid at 128.
- Run Linear2 grids 112 and 132 independently with timeout protection.
- Use the existing benchmark script, max-rank latency, and identical seed.
- Record completion, correctness, and timing for every candidate.

Evidence: raw logs plus one structured `ITERATIONS.md` entry and local commit
per completed benchmark run.

## 6. Confirm or reject the experiment

If a candidate shows a credible improvement:

- Run focused exact-shape correctness with multiple seeds.
- Run three interleaved max-rank median-20 comparisons against PR323.
- Require no correctness regression and a repeatable win, not a single noisy
  measurement.
- Update only the exact H200 Pro M256 `l2_num_sms` selector row.

If no candidate meets the gate:

- Revert the three-file experimental implementation while preserving design,
  plan, benchmark evidence, and iteration history.
- Leave the current 128/128 H200 selector unchanged.

## 7. Final audit

- Verify H20 selector behavior and entries were not edited.
- Verify no PR323 kernel code was introduced.
- Verify branch status, local commits, and that no push occurred.
- Report the exact candidate table and final HEAD.
