# SM90 NVFP4 proxy-publication integration

## Scope

Integrate only the correctness fixes from `bd4cee1`, `d67b701`, and
`6a070f1` into the current MiMo tuning tree.  Do not import selector or
performance changes from `e5944f9`, and do not disturb the tuning changes
already present in this worktree.

The frozen pre-fix tracked diff has SHA-256
`77f4a42b3b0832aafc9b288438a26c490591fa4ef47aa5cb5130246bf7d95124`.

## Publication protocol

Every thread that writes the dequantized B tile through the generic shared-
memory proxy executes `fence_view_async_shared()`.  The complete writer set
then rendezvous before one or more leaders publish completion through the
stage mbarrier.  Math-side dequant uses a 128-thread warpgroup barrier before
that warpgroup can issue WGMMA.  The defensive split math-side paths use the
epilogue-wide barrier because their B tile is shared by all epilogue
warpgroups.

The locally tuned per128 braided-scratch loader already uses a valid grouped
publication protocol: each loader warp fences all writers, synchronizes its
lanes, and contributes one arrival to an mbarrier initialized with exactly
the loader-warp count.  Keep that protocol and add only the missing fallback
and math-side synchronization.

## Validation

First build and run focused correctness/stress coverage for loader- and
math-side dequant routes.  Then compare the frozen pre-fix and fixed trees on
the same H200 allocation with identical inputs and selector settings.  Flush
L2 before every timed launch.  Report absolute latency and relative
regression, with large-M measurements limited to three timed launches at
seed 0.
