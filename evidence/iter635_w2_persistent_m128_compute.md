# Iteration 635: W2-only persistent-state M128 compute gate

## Protocol

- Physical GPU: H20 GPU1, 78 SMs.
- Shape: TP4 specialization, random-route M128, W13 split-K2, 1,992 padded
  route rows.
- Candidate: selected production one-launch compute body plus
  `V4_SINGLE_LAUNCH_W2_PERSISTENT_STATE=1`.
- Cache policy: a separate 256 MiB clear immediately preceded the CUDA
  launch; clear cost was excluded.
- Scope: compute-only state-machine validation.  The TP collective was
  deliberately disabled, and no end-to-end timing claim is made.

## Result

The complete candidate `down` tensor is bitwise identical to the separately
launched same-source multi-kernel local reference:

```text
cosine = 1.0
relative L2 = 0.0
finite = true
```

All four packed whole-grid barrier words were initialized at generation
`2^22-1` and wrapped to `[0, 0, 0, 0]`.  The launch therefore validates W2's
cross-task mbarrier parity, alternating route metadata, final-task output
publication and entry into the following grid barrier at M128.  The candidate
passes the correctness gate and is eligible for a paired distributed timing
screen.
