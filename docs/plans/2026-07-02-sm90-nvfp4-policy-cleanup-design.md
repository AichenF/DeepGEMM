# SM90 NVFP4 Policy Cleanup

## Goal

Remove experiment-only controls from the finalized SM90 NVFP4 MegaMoE path while preserving compile-time specialization for policies that vary by shape or expected tokens per local expert.

## Decisions

- Delete the CTA receive-count cache implementation. It is never selected by the automatic policy and no stable winning range was found.
- Remove the `DG_SM90_NVFP4_DP4A_SELECTOR_PACK` and `DG_SM90_NVFP4_HYBRID_LOW_SELECTOR_PACK` kill switches. The measured expected-token ranges become the only policy source.
- Keep `dp4a_selector_pack`, `hybrid_low_selector_pack`, and `swap_ab` as JIT specialization inputs because their values vary across generated kernels.
- Keep the Pro paired-LUT choice encoded directly from `kIntermediateHidden >= 3072`; it needs no environment variable or host argument.
- Leave loader-dequant and phase-profiling controls unchanged because they are outside this cleanup: loader-dequant selects distinct shape-dependent implementations, and phase profiling is diagnostic.

## Code Changes

- Remove `cache_recv_counts` from host runtime arguments and generated fused-kernel template arguments.
- Remove `kCacheRecvCounts`, the cache barrier, cache population/fetch helpers, and their call sites.
- Restore direct scheduler receive-count fetches at each scheduling consumer.
- Collapse DP4A and hybrid policy target variables into final automatic booleans.

## Verification

- Search production sources to ensure the three removed environment variables and all cache identifiers are gone.
- Run formatting/diff checks and compile the affected JIT kernels.
- Run the existing 24-case correctness matrix.
- Re-run representative Flash and Pro boundary benchmarks around expected-token policy transitions.
