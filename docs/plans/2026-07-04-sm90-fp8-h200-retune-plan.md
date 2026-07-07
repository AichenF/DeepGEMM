# SM90 FP8 MegaMoE H200 Retuning Implementation Plan

## 1. Initialize the optimization workspace

- Work on `opt/megamoe-sm90-fp8-h200-retune` from commit `3552b625`.
- Add `HINTS.md` containing the user constraints and `ITERATIONS.md` containing
  the baseline plus one structured record per measured iteration.
- Preserve the existing CUDA 13.2 compatibility commit.
- Record the pinned current and PR323 source identities.

Evidence: clean Git status, branch log, and committed workspace scaffold.

## 2. Acquire and capture an H200 environment

- Request one 8-GPU H200 node with a ten-hour allocation.
- Record job ID, node, GPU model/count, driver, CUDA, PyTorch, and clock state.
- Reuse the validated CUDA 13.2 Python environment where available.
- Copy or fetch the local optimization branch without pushing it.

Evidence: environment manifest and successful eight-rank CUDA smoke.

## 3. Build the pinned implementations

- Build the current FP8 branch at the local optimization HEAD.
- Reuse or rebuild PR323 at exact head `8ddf7f96` plus syntax-only fix
  `184d74cb`.
- Isolate JIT and extension caches by implementation and candidate identity.

Evidence: build logs, exact `git rev-parse` output, and clean tracked source
state before tuning.

## 4. Establish correctness and performance baselines

- Run the SM90 FP8 correctness suite before tuning.
- Run Flash and Pro at M 8 through 8192 with the same routes used in the prior
  comparison.
- Reproduce current FP8 and PR323 max-rank latency within normal run variance.
- Dump the selected configuration and generated JIT template arguments for
  every required M point.

Evidence: correctness log, baseline CSV, config matrix, and raw logs.

## 5. Run the H200-only parameter search

Use existing force controls only in the experiment harness. Search in this
order so that each axis has an attributable result:

1. Pro direct L2 scatter on/off, beginning at M 512.
2. Pipeline stages around the legal/default value.
3. Experts per wave across divisors of local experts.
4. Block M/N and legal epilogue warpgroup layouts.
5. Dispatch warp counts.
6. L2 N-major and one-warp cleanup.
7. Flash-specific versions of the same axes.
8. Legal swapAB boundary behavior at M 128 and the no-regression small-M set.

Use median-10 one-seed screens for ranking, then balanced multi-seed
confirmation for the beam-search survivors. Log every measured source change
or promoted selector experiment as an AKO iteration and commit it before
starting the next iteration.

Evidence: per-iteration raw output, `ITERATIONS.md`, and Git history.

## 6. Encode the winning H200 selector

- Add explicit H200 detection rather than using the generic high-SM bucket.
- Encode stable choices using shape and expected-tokens-per-expert buckets.
- Keep non-H200 selector behavior byte-for-byte or configuration-for-
  configuration unchanged.
- Remove temporary forcing variables from production behavior; existing debug
  controls may remain only if they predate this work.

Evidence: selector unit/config-matrix tests covering H200 and non-H200
profiles.

## 7. Re-run correctness and neighboring-point checks

- Rebuild from a fresh JIT cache.
- Run the full existing correctness suite.
- Verify required Flash/Pro M points and adjacent load buckets.
- Confirm finite output, tolerance, rank completeness, and route identity.

Evidence: final correctness log and config matrix.

## 8. Run the final performance gate

- Seeds 101, 202, and 303.
- Two balanced observations per implementation and seed.
- Median of 30 timed samples per rank and observation.
- Require all eight rank records.
- Require every M >= 128 point to beat PR323 by confirmed max-rank latency.
- Require M < 128 to avoid a confirmed regression from commit `3552b625`.

If a nominal gap is under 1%, collect additional repetitions before deciding.

Evidence: strict parser output, final CSV/Markdown table, and raw-log hashes.

## 9. Finalize locally

- Leave HEAD at the best verified selector implementation.
- Commit source, tests, scripts, reports, and iteration records.
- Confirm the worktree is clean and the branch has not been pushed.
- Do not create or update a remote branch.

Evidence: final commit hash, clean `git status`, and no upstream/push record.
