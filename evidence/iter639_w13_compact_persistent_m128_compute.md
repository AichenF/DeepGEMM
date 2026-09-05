# Iteration 639: compact-W13 persistent-state M128 compute gate

## Protocol

- H20 physical GPU1, 78 SMs.
- TP4 random-route M128, W13 split-K2, 1,992 padded route rows.
- Selected compact W13 phase and M128 702x128 launch with only
  `V4_SINGLE_LAUNCH_W13_COMPACT_PERSISTENT_STATE=1` added.
- Separate 256 MiB L2 clear immediately before the launch; clear excluded.
- Compute-only comparison; TP communication deliberately disabled.

## Result

The complete final W2 route tensor is bitwise identical to the independent
same-source multi-kernel local reference:

```text
cosine = 1.0
relative L2 = 0.0
finite = true
```

All four packed whole-grid barrier words wrap from generation `2^22-1` to
`[0, 0, 0, 0]`.  This validates the W13 persistent state across consecutive
tasks inside the noinline compact callee, its final split-K partial
publication and transition into the caller's phase barrier.  No end-to-end
latency is claimed by this correctness gate.
