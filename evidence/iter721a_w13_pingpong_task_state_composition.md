# Iteration 721a: W13 ping-pong task state composition

## Hypothesis

The selected M128 compact W13 outline executes several grid-stride tasks per
CTA.  The ordinary route-GEMM task initializes its TMA barriers, synchronizes
the CTA, executes the task, and the caller synchronizes the CTA again before
the next invocation.  The caller barrier and the next invocation's entry
barrier are consecutive task-boundary rendezvous.

Alternate two sets of TMA mbarriers and route metadata by task sequence.  A
fast warp may initialize/write task `t+1`'s alternate state while a lagging
warp finishes task `t`, and the existing task-entry CTA rendezvous still
prevents reuse of common dynamic shared-memory stages until every warp has
finished `t`.  Task `t+2` cannot reinitialize task `t`'s bank until every warp
has passed task `t+1`'s entry rendezvous.  This makes the caller's post-task
rendezvous redundant without weakening task-state lifetime.

The shared LUT is constant and is initialized only for sequence zero in this
mode, avoiding a same-address write/read race before the entry rendezvous.
Weight/activation dynamic shared memory remains single-buffered and is not
touched for the new task until after that rendezvous.

## Scope and rollback

- opt-in: `V4_SINGLE_LAUNCH_W13_COMPACT_PINGPONG_STATE=1`;
- M128 compact-ABI, bound-9, valid-task, schedule-0 path only;
- existing task order, WGMMA math, global partial layout, activation, W2, and
  embedded CAR transport are unchanged;
- M8/M16/M32/M64 and the default production configuration are unchanged;
- rollback is the environment flag at zero.

## Gates

1. Fresh JIT must compile with no local-memory spill and retain the selected
   M128 occupancy/resource envelope.
2. Compute-only and TP4 end-to-end output must pass the established numerical
   checks; the kernel launch count must remain one business kernel per rank.
3. Only then compare paired CUDA Graph cold-L2 M128 samples against the exact
   flag-zero control.  The 256 MiB L2 clear is excluded from timing and is
   issued before every replay on every TP rank.

No performance or correctness result is claimed by this composition commit.
