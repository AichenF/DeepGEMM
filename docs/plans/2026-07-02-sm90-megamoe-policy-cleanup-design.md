# SM90 MegaMoE Policy Cleanup Design

## Goal

Reduce the retained SM90 MegaMoE implementation to production paths and make
kernel selection readable without changing the selected Flash or Pro kernel
configuration at any benchmarked token count.

## Behavioral Contract

- Preserve the complete `MegaMoESM90Config` tuple for Flash and Pro at M values
  8, 16, 32, 64, 128, 256, 512, 819, 1024, 1280, 1536, 2048, 3072, 4096, and 8192.
- Preserve Layer-2 numerical correctness, including top-k 6 and top-k 8 cases.
- Do not retain a measurable Flash or Pro latency regression. Compare the clean
  tree against an isolated worktree at pre-cleanup commit `be7c5a3` with the
  same seed, L2 policy, container, and idle GPUs.
- Keep swapAB, the logical SF stride, model benchmark presets, and top-k 6 test
  coverage.

## Policy Structure

Identify production profiles from expert topology, top-k, and
`intermediate_hidden`, rather than model names or an exact runtime M:

- the narrow-FFN top-k 8 profile uses 32 local experts and intermediate hidden
  2048;
- the wide-FFN top-k 6 profile uses 48 local experts and intermediate hidden
  3072;
- other shapes use the generic fallback.

Within a profile, load-sensitive decisions use
`expected_tokens_per_expert`. Keep these predicates directly in the existing
policy object: an attempted FFN-width/load-band abstraction added measurable
CPU dispatch overhead for small Flash cases even though it selected identical
GPU kernels. The swapAB environment override keeps its existing token-cutoff
interface. No rule selects a kernel for one exact raw M such as
`num_tokens == 512`.

## Cleanup Scope

- Retain the opt-in BM128/BN256/4WG split-MN compatibility path. Removing its
  host selector produced a repeatable Flash M=256 regression, while deleting
  its low-level template structure changed NVCC code generation. The path
  remains disabled by default and isolated from the production policy.
- Remove the unreliable in-process benchmark repeat/median option.
- Stop tracking `HINTS.md` and `ITERATIONS.md`, but retain both in the local
  workspace through `.git/info/exclude`.
- Delete untracked `.orig` backups and the two broken CUTLASS include symlinks.
- Rewrite the optimization branch into a small logical history only after all
  verification gates pass.

## Verification

1. Capture and compare normalized config records for all 30 Flash/Pro points.
2. Force rebuild the extension.
3. Run all Layer-2 SM90 MegaMoE correctness scenarios on eight ranks.
4. Benchmark Flash and Pro against the untouched `be7c5a3` worktree using the
   same benchmark harness, locked clocks, independent process runs, and ABBA
   ordering; reject any repeatable regression.
5. Verify the final Git tree excludes process artifacts and experimental code,
   then update the remote branch with `--force-with-lease`.
