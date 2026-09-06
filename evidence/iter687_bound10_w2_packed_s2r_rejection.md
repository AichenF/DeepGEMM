# Iteration 687: reject packed-weight-only W2 S2R

## Question

Can the M128 one-kernel path recover the multi-kernel phase's lower register
footprint while retaining most of the selected W2 S2R latency hiding?

The candidate keeps only the next packed MXFP4 weight words live across the
current QGMMA. E8M0 scale loads and LUT/dequantization stay in the current
iteration. This reduces the launch bound from nine to ten 128-thread CTAs per
H20 SM without changing routing, W13, activation, W2 arithmetic, or TP4
communication.

## Build and correctness

- Candidate SHA-256:
  `bfe44f6a9a657ffe04c5d5873fa4aea7baeee66e85a3755c8820a220bed86ad4`
- JIT extension: `v4tp_dd8a6804edd0c89d31a0_v178mspec`
- M128 split-K2 and split-K4 resources:
  `REG48 STACK48 SHARED2048 LOCAL0 CONSTANT[0]1361`
- Selected production resources: `REG56 STACK32 SHARED2048 LOCAL0`
- Local M128 full output: accepted, cosine 1.0, rel-L2 0.0, finite, packed
  generation wrapped to `[0,0,0,0]`.
- Distributed TP4 output: accepted on every rank, cosine-min
  `0.9999955976741226`, rel-L2-max `0.0029672639980990933`, finite and
  `allreduce_ok=true`.

Thus the test reached the intended ten-CTA occupancy contract and is not a
register-bound launch failure.

## Cold-L2 CUDA Graph screen

Physical H20 GPUs 0/5/6/7; M128 random routing, seed 20260902; 248 active
experts, 768 routed rows and 1,992 padded rows. Each implementation replay has
its own excluded 256 MiB L2 clear. Measurement is replay-paired, with four
warmups and 2 x 20 cold samples per implementation.

| Path | Min (ms) | Median (ms) | Max (ms) |
|---|---:|---:|---:|
| Same-source multi-kernel control | 0.304096 | 0.305472 | 0.332064 |
| Packed-only-S2R one kernel | 0.360320 | 0.363680 | 0.373600 |

Candidate/control is `1.1905509784`: the candidate is 19.06% slower. The
candidate is also about 4.02% slower than the adjacent exact-production
one-kernel anchor of 0.349616 ms. Compared with the fully disabled W2-S2R
candidate from Iteration 679 (about 0.368512 ms), packed-only prefetch recovers
some latency, but not enough to match full S2R.

## Verdict

Reject. Lowering the parent to 48 registers and admitting a tenth CTA is not
the right transfer by itself. Exposing scale/LUT decode on the W2 inner-loop
critical path outweighs the extra resident CTA. Restore the exact selected
source hash `7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
