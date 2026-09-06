# Iteration 694: W2 K128 merged-group resource/correctness gate

The candidate is the unchanged Iteration-693 source, executed through the
pinned benchmark environment runner.  It changes only the flat TP4
single-launch W2 instantiation; the same-source multi reference keeps the
selected per-K32 WGMMA commit/wait schedule.

The exact M128/split-K2 cubin entry reports `REG56 STACK32 SHARED2048
LOCAL0 CONSTANT[0]1361`, preserving nine resident CTAs per H20 SM.  On
physical GPU1 with random M128 routes and seed 20260902, the full routed W2
tensor is bitwise equal to the independent multi reference (`cosine=1`,
`rel_l2=0`, finite).  The run uses caller-provided FP8-E4M3 X, FP32
group-128 scales, MXFP4 weights, and an excluded 256 MiB clear before the
measured diagnostic launch.  The deliberately wrapped packed generations
finish at `[0,0,0,0]`.

No TP collective or latency comparison is claimed by this gate.
