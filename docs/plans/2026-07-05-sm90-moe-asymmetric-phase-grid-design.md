# SM90 MegaMoE Asymmetric Phase-Grid Design

## Goal

Allow the split SM90 FP8 MegaMoE Linear2 kernel to launch with a different
CTA/SM count from Linear1 without changing the producer count that Linear2
waits for.  Use the new control only to investigate the remaining H200 Pro
M256 performance gap against PR323.

This is an H200 tuning experiment.  It must not change existing H20 tuning,
copy or fuse PR323 code, or alter same-grid behavior.

## Current failure mode

In split-kernel mode, Linear1 dispatch CTAs finalize each expert's receive
counter.  The counter's high 32 bits therefore reach
`l1_num_sms * num_ranks`.

`MegaMoEScheduler::fetch_expert_recv_count()` currently waits for
`kNumSMs * kNumRanks`, where `kNumSMs` is also the current phase's launch
grid and work-stride size.  When Linear2 is launched with a different grid,
it waits for a count that Linear1 can never produce and hangs.

## Selected approach

Introduce one compile-time scheduler parameter for the dispatch-counter
producer count, named `kNumDispatchSMs`, with a default equal to `kNumSMs`.

- Keep `kNumSMs` as the current kernel's launch grid and scheduler stride.
- Use `kNumDispatchSMs` only in the expert receive-counter completion test.
- Instantiate both split phases with `kNumDispatchSMs = l1_num_sms`, because
  Linear1 dispatch is the counter producer.
- Keep Linear2's own `kNumSMs = l2_num_sms`, so its block scheduling,
  cleanup distribution, barriers, combine work, and launch shape remain tied
  to the Linear2 grid.
- Preserve the default `kNumDispatchSMs = kNumSMs` so other scheduler users
  and all equal-grid configurations retain their existing behavior.

The host JIT argument is phase-independent producer metadata.  It is not a
new public tuning knob: it is derived from the already selected Linear1 grid.

## Alternatives considered

### 1. Separate dispatch-counter grid (selected)

Decouple only the completion count that crosses the phase boundary.  This is
the smallest change matching the producer/consumer ownership of the counter.

### 2. Use the Linear1 grid for all Linear2 scheduler logic

This avoids the hang but defeats the purpose of an asymmetric Linear2 launch
and can misassign work because Linear2 would launch one grid while striding as
another.  Rejected.

### 3. Change the counter protocol to a runtime expected count

A runtime field could avoid extra JIT variants, but it broadens the kernel ABI
and symmetric workspace protocol for a narrow experiment.  Rejected unless a
future use case needs fully dynamic phase grids.

## Safety invariants

- `kNumDispatchSMs` is positive and represents the actual Linear1 launch grid.
- Only the high-32-bit receive-counter completion comparison uses it.
- Every other existing `kNumSMs` use remains unchanged.
- Equal Linear1/Linear2 grids generate the same effective wait count as before.
- Existing H200 selector entries stay unchanged until an asymmetric candidate
  passes correctness and repeated performance gates.
- H20, H100, generic SM90, and unmatched shapes keep their existing selector
  paths.

## Validation

1. Compile the focused host selector test and the CUDA extension.
2. Confirm the existing equal-grid Pro M256 path still completes and is
   numerically correct.
3. Screen H200 Pro M256 with Linear1 fixed at 128 and Linear2 at 112 and 132.
4. Reject candidates that hang, fail finite/diff checks, or lack a stable
   performance improvement.
5. For a winner, run exact M256 correctness and three interleaved max-rank
   median-20 observations against PR323.
6. Update the H200 policy only if the candidate is repeatably faster; otherwise
   revert the experimental source change and keep the current selector.

No result is pushed.
