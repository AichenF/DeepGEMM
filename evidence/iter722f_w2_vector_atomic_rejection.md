# Iteration 722f — reject Hopper vector-atomic W2 producer combine

- **Candidate:** `520b94f`, default-off
  `V4_SINGLE_LAUNCH_W2_PRODUCER_VECTOR_ATOMIC_COMBINE=1`.  Four scalar FP32
  atomics per W2 lane were replaced by two element-wise `float2` atomics in
  half the lanes after four-lane shuffles.
- **Static/resource gate:** CUDA 12.8 lowered the M128/SplitK2 entry to eight
  static `REDG.E.ADD.F32x2.FTZ.RN.STRONG.GPU` instructions.  TP4 entries are
  `REG64 STACK32 SHARED4096 LOCAL0`, so there is no local-memory spill and the
  matched bound-8 residency is preserved.
- **Local correctness:** physical H20 GPU1, random seed 20260902, separate
  excluded 256 MiB cold-L2 clear, packed-generation wrap enabled.  M8 local
  sum cosine/relative-L2 are `0.9999999999999968 / 8.6160e-8`; M128 are
  `0.9999999999999969 / 8.5500e-8`.  Both are finite and all packed barrier
  words wrap to `[0,0,0,0]`.
- **TP4 protocol:** GPUs 1/5/6/7, same-process matched ordinary one-kernel
  control versus vector-atomic candidate, CUDA Graph, two balanced batches x
  twenty rank-max replays per arm, four warmups, and a separate excluded
  256 MiB L2 clear immediately before every replay.  Both paths include their
  complete in-kernel M128 P2P two-shot TP collective.
- **M128 cold-L2 result:** ordinary control min/median/max
  `0.343328/0.346480/0.476896 ms`; vector atomic
  `0.351552/0.354592/0.357472 ms`.  Control/candidate is `0.977123x`, so the
  candidate is **2.3410% slower**.  Both batch medians regress
  (`0.346304 -> 0.354656` and `0.346672 -> 0.354592 ms`).
- **Correctness:** the four-rank final BF16 tensors are exact between arms;
  all outputs are finite.
- **Decision:** reject immediately on the predeclared M128 hard gate and do
  not spend M8 distributed or all-five-M timing.  Vector issue halves static
  atomic instructions but shuffle/repacking and/or vector-atomic execution
  cost more than the removed terminal k6 reread.  Restore production source;
  the result does not change the latest 10.101% one-kernel gap.
- **Raw artifacts:**
  `bench/results/iter722b_w2_vector_atomic_m8_jit_correctness_20260906.log`,
  `bench/results/iter722c_w2_vector_atomic_m128_correctness_20260906.log`,
  `bench/results/iter722d_w2_vector_atomic_resources_20260906.log`, and
  `bench/results/iter722e_w2_vector_atomic_tp4_m128_cold_screen_20260906.log`.
