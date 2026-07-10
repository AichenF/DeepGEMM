# SM90 MegaMoE Range Selector Design

## Goal

Replace the layered exact-point SM90 MegaMoE tuning with one selector based on
hardware class, H/I shape ranges, routed-load ranges, and derived legal
parallelism. Keep the measured H20 and H200 performance while making every
cross-rank protocol choice independent of rank-local token count.

## Selector Structure

The generic SM90 heuristic remains the complete legal fallback. A single
hardware-specific range selector may patch its local scheduling fields:

- LowSm: H20 ranges.
- HighSm: H200/high-SM Hopper ranges.
- Generic: no tuned patch.

The old `Sm90MoeProfileTuning` layer is removed. Production selection must not
use exact token/expert identities, `load.equals()`, or `load.is_one_of()`.
Routed load is compared exactly by cross multiplication:

```text
routed_load = num_tokens * num_topk / num_experts_per_rank
```

Shape matching uses the existing Compact and Wide H/I intervals. Top-k and
local expert count are not workload fingerprints; they contribute to routed
load and legality/parallelism derivation only.

## Derived Parallelism

Load bands specify a target experts-per-wave value when measurements justify
one. The resolver clamps the target to the available local experts and chooses
a legal divisor. Tile divisibility, stage capacity, thread topology, and shared
memory continue to be derived and validated before a candidate is accepted.

This keeps the E32/E48 measured configurations while allowing other expert
counts to use the same shape/load policy without exact identity checks.

## Cross-Rank Invariants

`num_sms` is not a tuning field. Both phases always use `launch_num_sms`, and
launch legality requires L1 and L2 to equal it. On a full H200 this is 132.
Rank-local routed load may change tiles, stages, EPW, and local scheduler modes,
but it cannot change a cross-rank completion count.

FP8 combine remains an explicit engine/runtime ABI choice and is never derived
from local load. No additional rank broadcast is added.

## Numerical Policy

BF16 scaled accumulation remains separate from schedule selection. HighSm
eligibility uses shape and routed-load ranges plus kernel legality, without
exact E32/E48 or direct top-k matching. LowSm keeps BF16 disabled.

## LowSm Migration

The old H20 point table is not mechanically wrapped in narrow intervals.
Coherent load bands are formed from scheduling transitions, with experimental
hints acting only as modifiers over a band. The existing Flash, MiMo, and Pro
33-point matrix is the performance baseline, not a production lookup table.

## Verification

- Add a host selector golden covering both hardware classes, range boundaries,
  alternate expert/top-k combinations at equal routed load, legal fallback,
  and the `num_sms == launch_num_sms` invariant.
- Run local formatting/compile checks and the selector test.
- Rebuild and run SM90 correctness for representative Flash and Pro cases.
- Run same-machine H20 Flash/MiMo/Pro and H200 Flash/Pro matrices against the
  current commit. Do not accept a stable performance regression; noisy small-M
  points require repeated paired measurements before changing a range.
- Do not add an uneven-M distributed test for this change.
