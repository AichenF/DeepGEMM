# SM90 FP8 H200 Conservative Selector Implementation Plan

## 1. Capture authoritative candidate configurations

- Recover every retained candidate's complete environment/config record from
  the H200 artifacts.
- Convert the records into a single table keyed by `(shape, M)`.
- Verify that M512 has no automatic entry and that no entry requires
  phase-specific BF16 accumulation.

Evidence: checked-in selector table in tests or comments whose values match
the raw candidate `config.txt` files.

## 2. Add explicit H200 identification

- Add a small device-runtime predicate based on `cudaDeviceProp::name`
  containing `H200`.
- Keep architecture/SM-count profiles unchanged.
- Add a test seam that permits selector tests without requiring an H200 host.

Files:

- `csrc/jit/device_runtime.hpp`
- `csrc/jit_kernels/heuristics/mega_moe.hpp`

Evidence: unit/config tests distinguish H200 from H100/H20 and unmatched SM90
devices.

## 3. Implement a structured H200 MegaMoE policy

- Represent automatic global features and phase-local tile/wave/stage/SM
  choices in one policy object.
- Match the complete Flash or Pro workload identity before selecting a row.
- Add only the approved M rows; all other M values, especially M512, return
  an empty policy and preserve the current configuration.
- Let explicit experiment environment variables override automatic defaults.
- Keep global BF16 only on the approved Pro rows; do not add phase-specific
  BF16 state or controls.

Files:

- `csrc/jit_kernels/heuristics/mega_moe.hpp`
- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`

Evidence: generated configs match retained candidate records exactly.

## 4. Add selector and non-regression tests

- Cover Flash M128/M8192 and Pro M128/M256/M1024--8192 selected rows.
- Cover M8/16/32/64, M512, unmatched M, unmatched shapes, H20, H100, and
  generic SM90 fallthrough.
- Assert that no policy requests phase-specific BF16.
- Assert explicit experiment overrides still take precedence.

Files:

- `tests/test_mega_moe_sm90.py` or a focused host-selector test module.

Evidence: local CPU/config tests pass before GPU testing.

## 5. Rebuild on H200 and run the global-BF16 numerical gate

- Acquire a replacement 8x H200 allocation if the current job expired.
- Clean-build the host extension and fresh JIT cache from the committed source.
- On the actual Pro shape, compare FP32, global BF16, and the golden reference
  at M128/1024/2048/4096/8192 across multiple seeds.
- Require finite output and `calc_diff < 0.01` on every rank; report the worst
  seed and direct BF16-versus-FP32 difference.

Evidence: raw logs plus an `ITERATIONS.md` entry and commit.

## 6. Confirm the automatic selector performance matrix

- Run without tuning environment variables.
- Flash and Pro: M8/16/32/64/128/256/512/1024/2048/4096/8192.
- Interleave with PR323 at every M >= 128 except M512.
- Require every in-scope point to beat PR323 using three max-rank median-20
  observations.
- Compare M8/16/32/64 with exact baseline `3552b62` on the same node.
- Record M512 for visibility but do not gate on it.

Evidence: complete table, raw logs, and an `ITERATIONS.md` entry.

## 7. Remove rejected experiment paths

- Delete default-off implementations not used by any retained selector row,
  prioritizing native/chunked FP16 WGMMA and invalid accumulator candidates.
- Keep only mechanisms required by the automatic selector or existing generic
  behavior.
- Rebuild and rerun focused correctness plus the selected performance points
  after each cleanup group.

Evidence: source diff is reduced without changing selected generated configs or
validated timings beyond noise.

## 8. Final audit and handoff

- Verify branch/worktree cleanliness and no upstream/push.
- Verify all required commits are local.
- Recheck every design requirement against current source and raw evidence.
- Leave HEAD at the best fully verified selector implementation.

Do not push.
