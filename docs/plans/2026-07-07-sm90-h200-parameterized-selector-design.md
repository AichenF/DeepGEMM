# SM90 H200 Parameterized MegaMoE Selector Design

## Goal

Replace the H200 schedule table that names workloads and matches exact runtime
`M` values with a parameterized, range-based selector. The refactor must first
resolve to the same L1/L2 configurations at every currently validated Flash and
Pro point. Only after that parity gate passes may the new Pro load buckets be
tuned.

The current numerical-format policy is a separate validation contract. This
refactor must not broaden BF16 scaled accumulation or E5M2 combine coverage.

## Constraints

- Apply automatic schedule rules only on H200. Preserve H20, H100, and generic
  SM90 behavior.
- Keep the split L1/L2 implementation. Do not introduce the PR323 fused kernel.
- Do not use model names in production selection.
- Do not use total experts, local experts, or top-k as direct shape-profile
  selectors. `top_k` and `local_experts` contribute only to
  `expected_tokens_per_expert = M * top_k / local_experts` and to legality
  checks such as choosing an experts-per-wave divisor.
- Use ranges rather than an exact-`M` kernel table.
- Do not commit or push unless the user explicitly requests it.

## Approaches Considered

### 1. Parameter ranges plus a resolver (selected)

Match a small H/I shape band and an expected-load interval. A rule returns only
the knobs supported by that phase. A resolver derives thread topology and
shared-memory requirements and rejects illegal combinations. This preserves
the measured configurations while keeping model identity and exact `M` out of
schedule selection.

### 2. One analytic cost model

Enumerate every legal configuration and rank it from estimated waves,
occupancy, and traffic. This is compact, but the current cost model does not
predict the H200 phase imbalance or tail behavior accurately enough to replace
measured rules.

### 3. A dense multidimensional lookup table

Key rows by H, I, M, top-k, and expert topology. This reproduces measurements
exactly but keeps the current discontinuities and encodes model identities
indirectly. It is rejected.

## Inputs and Derived Features

The schedule selector receives:

```text
hidden
intermediate_hidden
num_tokens
top_k
local_experts
device_sms
```

It derives routed load as the rational value:

```text
routed_tokens = num_tokens * top_k
expected_tokens_per_expert = routed_tokens / local_experts
```

Range comparisons use cross multiplication instead of floating-point equality,
so a boundary such as `expected <= 32` is exact.

L1 and L2 derive their matrix dimensions independently:

```text
L1: N = 2 * intermediate_hidden, K = hidden
L2: N = hidden,                  K = intermediate_hidden
```

## Selector Structure

The selector has three layers:

1. The existing generic SM90 heuristic produces a complete legal base config.
2. H200 rules return sparse common, L1, and L2 patches.
3. The resolver applies the patches, derives coupled values, and validates the
   result before JIT launch.

The schedule patch contains only live choices:

```text
common: block_m, cluster_size
L1:     block_n, block_k, experts_per_wave, stages, num_sms, nmajor
L2:     block_n, block_k, experts_per_wave, stages, num_sms, nmajor,
        direct_scatter, one_warp_cleanup
```

Dispatch threads, non-epilogue threads, epilogue warpgroups, shared-memory
size, TMA box sizes, and swizzles are derived from the selected tiles and
frontend mode. They are not independent H200 table fields. Existing
environment controls remain experiment overrides but are not production
selector inputs.

## Shape and Load Rules

Rules use paired H/I bands rather than model names:

- compact-FFN band: `3072 <= H < 5120` and `1536 <= I < 2560`;
- wide-FFN band: `5120 <= H <= 8192` and `2560 <= I <= 4096`.

A shape inside a band still falls back to the generic selector if a requested
tile does not divide its L1/L2 dimensions or another kernel legality check
fails.

The initial behavior-preserving load bands are:

| H/I band | Expected tokens/expert | Initial schedule behavior |
|---|---:|---|
| compact | `(12, 32]` | Preserve current Flash M128 config |
| compact | `(128, 256]` | Preserve current Flash M1024 config |
| compact | `(1024, +inf)` | Preserve current Flash M8192 config |
| wide | `(8, 24]` | Preserve current Pro M128 config |
| wide | `(24, 48]` | Preserve current Pro M256 config |
| wide | `(48, 96]` | BN512 L1 with L1/L2 N-major; validated at Pro M512 |
| wide | `(96, 192]` | BN512 L1 with L1/L2 N-major; validated at Pro M1024 |
| wide | `(192, +inf)` | Preserve the validated large-load M-major schedule |

These intervals keep all 22 currently validated Flash/Pro M points on their
existing schedule or existing generic fallback. They also remove exact-`M`
selection for intermediate runtime loads.

## Numerical Policy

Schedule and numerical decisions remain separate. Numerical modes retain the
current exact validated matrix and its FP32 exceptions. They may be expressed
with neutral H/I predicates, but this change does not interpolate numerical
formats across an H/I or expected-load range.

In particular, adding a schedule band does not automatically enable E5M2
combine or packed BF16x2 accumulation for a previously unvalidated case.
Measured wide-FFN BN512 rules carry an explicit packed-BF16 capability
requirement: schedule selection still matches H/I and expected load, but policy
composition falls back to the generic schedule when the independent numerical
policy has not validated packed BF16 for that runtime case. This prevents a
range rule from applying a register-heavy FP32 instantiation such as M768.

## Error Handling and Fallback

- No matching H/I plus load rule: use the complete generic SM90 config.
- A sparse field left unset: inherit the generic value.
- An experts-per-wave value that is not a positive divisor of local experts:
  reject the patch.
- A tile that is unsupported or does not divide the required L1/L2 dimensions:
  reject the patch.
- A requested policy stage count that does not fit shared memory: reject it
  instead of silently clamping it. Explicit experiment overrides retain their
  existing diagnostic behavior.
- Unknown devices and non-H200 SM90 devices never receive an H200 patch.

## Verification and Tuning Gates

1. Compile and run a CPU golden test comparing the old and new resolved
   schedule at all 22 Flash/Pro points.
2. Test each load boundary immediately below, at, and above the threshold.
3. Test unmatched H/I pairs and illegal divisibility cases for generic
   fallback.
4. Confirm that numerical-format output is unchanged at all validated points.
5. Rebuild on H200 and compare selected L1/L2 configs before measuring latency.
6. Run the existing correctness reference with the unchanged tolerance and
   multi-seed coverage.
7. Tune the wide `(48, 96]` bucket for Pro M512 and retune the wide
   `(96, +inf)` bucket for M1024. The final same-node, interleaved, cold-L2
   target is at least 10% below the measured DeepEP high-throughput baseline:
   M512 `< 1060.8 us`, M1024 `< 1455.8 us`.
8. Re-run the full Flash/Pro matrix to detect bucket-boundary regressions.
