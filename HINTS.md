# SM90 FP8 H200 Retuning Constraints

- Optimize only the existing SM90 FP8 MegaMoE implementation. NVFP4 is out of
  scope for this work.
- All new parameter decisions must be measured on NVIDIA H200. Historical H20
  tuning results are not valid evidence for choosing an H200 parameter.
- Do not import, restore, or implement the PR323 fused single-kernel path. Keep
  the current split L1/L2 kernel architecture.
- Tune the Flash shape `(H=4096, I=2048, E=256, top-k=6)` and the Pro shape
  `(H=7168, I=3072, E=384, top-k=6)`.
- Required M points are `8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  8192`.
- At every point with `M >= 128`, confirmed max-rank latency must be lower than
  the pinned PR323 FP8 implementation.
- Points with `M < 128` must not show a confirmed regression from the current
  `megamoe_sm90_opt` FP8 baseline.
- Use identical routes, capacity, rank topology, and timing boundaries for
  comparisons. Treat a nominal gap below 1% as unconfirmed until repeated.
- H200-specific selector choices must not alter H20, H100, or generic SM90
  behavior.
- Temporary experiment controls are allowed, but the final production selector
  must not depend on a new environment variable.
- Commits are allowed. Do not push any commit or branch.

