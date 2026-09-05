# Iteration 642: compact-W13 split-major M128 compute correctness

## Protocol

- H20 physical GPU1, 78 SMs.
- TP4 random-route M128, seed 20260902, W13 SplitK2, 1,992 padded rows.
- Production compact W13 phase and 702x128 M128 launch with only
  `V4_SINGLE_LAUNCH_W13_COMPACT_SPLIT_MAJOR_TASKS=1` added.
- Separate excluded 256 MiB L2 clear immediately before the measured launch.
- TP communication disabled; full routed W2 output compared locally.
- All packed barrier generations seeded at `2^22-1` to test wraparound.

## Result

The candidate is bitwise identical to the independent same-source
multi-kernel local reference:

```text
cosine = 1.0
relative L2 = 0.0
finite = true
```

The packed barrier words finish at `[0, 0, 0, 0]`, proving clean generation
wrap and complete progress through route preparation, reordered W13,
activation/requantization, and W2.  This gate contains no latency claim.
