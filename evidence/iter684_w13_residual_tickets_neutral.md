# Iteration 684: residual-only W13 tickets are neutral

Date: 2026-09-06

## Hypothesis and isolation

The multi-kernel W13 launch lets hardware assign its residual blocks to newly
available SM slots. This candidate approximates only that property. All 702
resident CTAs execute the selected five complete static waves, then lane zero
per CTA performs one atomic ticket allocation. Tickets below the dynamic
residual count execute one final W13 task. The cursor is reset before the
already-required route-to-W13 grid barrier. Logical task count, arithmetic,
weights, W2 and communication remain unchanged. The opt-in was
`V4_SINGLE_LAUNCH_W13_RESIDUAL_TICKETS=1`.

## Static and correctness gates

- Candidate SHA-256:
  `49d6daea3151518d6295aeb48b50ae3c3fbe6d837b889991faac8dae7a5c9759`.
- JIT extension: `v4tp_6b3e653b928821d0d829_v178mspec`.
- M128 split-K2/4 remains `REG56 STACK32 SHARED2048 LOCAL0`, preserving nine
  resident CTAs/SM.
- Single-GPU random-route M128 is bitwise equal to the independent local
  reference: cosine 1, relative L2 0, finite, with 1,992 padded rows.
- Packed generation wrap returns exactly `[0,0,0,0]`.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; replay-interleaved CUDA
Graphs; two batches of 20 samples after four warmups. Every replay had a
separate excluded 256 MiB L2 clear. Times are TP-rank-max.

| implementation/window | min (ms) | median (ms) | max (ms) | one/multi |
|---|---:|---:|---:|---:|
| exact production multi anchor | 0.302816 | 0.305008 | 0.329184 | - |
| exact production one anchor | 0.347008 | 0.349616 | 0.393728 | 1.146252 |
| ticket-run multi | 0.302944 | 0.305072 | 0.333728 | - |
| one, residual tickets | 0.347776 | 0.349808 | 0.374976 | 1.146641 |

Ticket candidate batches are 0.349712 and 0.350096 ms. Ratio normalization
is a 0.034% regression; direct one-kernel latency is 0.055% worse. All-rank
correctness and embedded P2P two-shot allreduce checks pass, with cosine
0.99999560 and relative L2 0.00296726.

## Verdict

Reject and restore exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
The measured static residual already has the optimal maximum task count per
SM. Dynamic completion order produces no gain distinguishable from noise and
cannot repay the 702 ticket atomics. This closes the last bounded emulation of
standalone residual-block turnover.

Raw evidence:

- `bench/results/iter684_w13_residual_tickets_m128_jit_correctness_20260906.log`
- `bench/results/iter684_w13_residual_tickets_m128_resources_20260906.log`
- `bench/results/iter684_w13_residual_tickets_tp4_m128_cold_short_20260906.log`
