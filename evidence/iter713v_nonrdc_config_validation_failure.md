# Iteration 713v: non-RDC rebuild is rejected by configuration validation

Date: 2026-09-06

The corrected Python command reaches module configuration validation but exits
before JIT because the environment did not explicitly select every prerequisite
of the M128 unbounded/whole-W2-phase experiment.  The validator requires the
selected compact-W13 schedule-0 path with the requested bound-8 contract,
rotations disabled, and an allowed W2 form.

No compilation or CUDA work occurs.  Inspect the source validator and recreate
the complete exact flag set before retrying; do not weaken validation or build
a merely similar control artifact.

Raw log: `bench/results/iter713v_miniforge_nonrdc_build_20260906.log`.
