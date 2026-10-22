# Dynamic expert load balancing for the SM90 MXFP4 MegaMoE kernel

Branch: `megamoe_mxfp4_dynamic_eplb`, cut from `megamoe_mxfp4_dev_m`.

## Problem

The fused MXFP4 MegaMoE kernel executes every expert on the rank that owns
its weights. Under routing skew the rank hosting the hottest experts decides
the layer latency while the others idle. Both references (TensorRT-LLM
DynamicEPLB, MegaFlux) solve this the same way: give every rank a few spare
*redundant slots*, replicate hot experts into them at runtime, and send a share of
those experts' token blocks to the replicas, leaving router outputs unchanged.

## Contract

* **Physical slots.** A rank holds `H = E / R` resident experts in local slots
  `[0, H)` and `S` redundant slots in `[H, H + S)`. Physical slot id of
  `(rank r, local slot j)` is `r * (H + S) + j`. The MegaMoE kernel, its
  symmetric buffer and its weight tensors are all built for
  `E_phys = R * (H + S)` "experts"; it never sees logical ids. This is the
  same slot-id routing TensorRT-LLM already uses for the SM100 DeepGEMM
  MegaMoE backend, so the kernel body does not change.
* **Logical input.** The caller provides `topk_idx` in logical expert space
  `[0, E)` plus `topk_weights`, exactly as today.
* **Per-iteration pipeline** (per MoE layer):
  1. `plan`: one-CTA kernel. Pushes this rank's expert histogram to every peer
     over the symmetric workspace, waits for all peers, then runs a
     deterministic greedy block-granular planner (MegaFlux A.1, load-only)
     identically on every rank. Outputs the slot table
     `slot_expert[R][S]`, per-expert copy list with cumulative block counts,
     and this rank's weight-copy list (slots whose expert changed).
  2. `copy_weights`: copies changed redundant slots' MXFP4 planes (packed L1/L2,
     tile-major scales, global scales) from the resident rank over NVLink. Weights
     live in a symmetric arena so peers can read them. Unchanged slots are
     not copied (slot cache).
  3. `remap`: the logical-to-physical remap of the top-k ids; rewrites the
     symmetric-buffer `topk_idx` to physical slot ids.
     Token `(t, k)` of expert `e` gets global ordinal
     `base_rank(e) + local_position`, block `ordinal / block_m`, and the
     copy whose cumulative block range contains it.
  4. `mxfp4_mega_moe` runs unchanged on physical slots.
* **Correctness invariant.** Any physical slot holding expert `e`'s weights
  produces the same result as the resident rank, so the plan only affects balance.
  Reference checks therefore compare against the logical-expert reference.

## Planner (device, one warp does the sequential part)

Inputs: `c[r][e]` token counts, previous `slot_expert`, block size `beta`
(= the kernel's `BLOCK_M` for this batch), replica open gate `lambda_open`
(blocks a new, uncached replica must receive to be worth a weight copy).

1. `b_e = ceil(sum_r c[r][e] / beta)`, `q[e][h(e)] = b_e`, loads
   `l_r = sum_e q[e][r]`, target `tau = ceil(sum_e b_e / R)`.
2. Up to `S + 2` rounds: visit overloaded resident ranks by decreasing overload. For
   each resident expert by decreasing remaining blocks, move
   `v = min(q[e][d], max(0, tau - l_r))` blocks to the least-loaded eligible
   receiver, preferring ranks that already hold `e` (an active copy or a
   cached slot), then ranks below `tau` with a free slot; a new uncached copy
   needs `v >= lambda_open`. A donor stops at `l_d <= tau`. A round with no
   movement ends the search.
3. Refinement: drop replicas whose allocation fits on another copy without
   raising the maximum load; then single-block moves from the most to the
   least loaded rank sharing an expert while their loads differ by more than one.
4. Slot assignment per rank: keep cached slots still in use, fill free slots
   (preferring stale ones) with new experts. Emits `slot_expert` and the
   per-rank copy list.

Ties break by index everywhere, so all ranks compute the same plan from the
same histogram matrix.

## Files

* `deep_gemm/include/deep_gemm/impls/smxx_mega_moe_eplb.cuh` - plan, remap,
  copy kernels.
* `csrc/jit_kernels/impls/smxx_mega_moe_eplb.hpp` - JIT runtimes.
* `csrc/apis/mega.hpp` - `mega_moe_eplb_plan`, `mega_moe_eplb_remap`,
  `mega_moe_eplb_copy_weights`, `sm90_mxfp4_mega_moe_block_m`.
* `deep_gemm/mega/eplb.py` - `DynamicEPLB` (workspace, weight arena, slot
  state, `forward`) and `plan_reference` (pure-Python planner used by tests).
* `tests/test_mega_moe_eplb_planner.py` - CPU tests of the reference planner.
* `tests/test_mxfp4_mega_moe_sm90_eplb.py` - multi-GPU correctness against
  the logical reference under skewed routing, planner-vs-reference agreement,
  and a skewed-routing benchmark with EPLB on and off.

## Out of scope for this cycle

In-kernel readiness gating of redundant slots (overlapping the weight copy with
resident-expert tasks) and multi-candidate planner search. Both are additive.

## Usage

```python
eplb = deep_gemm.DynamicEPLB(group, num_experts, num_redundant_slots,
                             num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
eplb.set_home_weights(l1, l2, l1_global_scales, l2_global_scales)   # transformed MXFP4 weights
y = eplb.forward(x_fp8, x_sf, topk_idx_logical, topk_weights)        # plan + copy + remap + kernel
```

`plan`, `copy_weights` and `remap` are also exposed separately. Every rank
must issue the same number of plans: the planner waits for its peers in the
kernel (30 s budget, then it traps), so host-side skew larger than that, for
example one rank JIT-compiling while another already planned, deadlocks it.
`DG_EPLB_DEBUG=1` prints the generation of every plan launch and entry.

## Measured (SM90, 2026-10-10)

Synthetic skew: three hot experts owned by rank 0 receive half of all routed
slots. `tests/test_mxfp4_mega_moe_sm90_eplb.py`, kernel time from kineto,
rank maximum, median of 20 calls, L2 flushed between calls.

EP2, 96 experts (48 per rank), 4 redundant slots, top-k 8, hidden 6144,
intermediate 2048:

| M per rank | busiest rank blocks | fixed (us) | EPLB kernel (us) | plan + copy + remap (us) | kernel-only | end to end |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 54 -> 51 | 417 | 403 | 43 + 1 + 2 | 1.035x | 0.930x |
| 256 | 59 -> 54 | 531 | 508 | 48 + 1 + 2 | 1.045x | 0.949x |
| 1024 | 179 -> 157 | 1518 | 1355 | 58 + 1 + 3 | 1.120x | 1.071x |

EP4, 192 experts (48 per rank), 4 redundant slots, same shape and skew:

| M per rank | busiest rank blocks | fixed (us) | EPLB kernel (us) | plan + copy + remap (us) | kernel-only | end to end |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 57 -> 48 | 434 | 370 | 50 + 1 + 2 | 1.170x | 1.023x |
| 256 | 72 -> 55 | 632 | 505 | 77 + 1 + 2 | 1.253x | 1.081x |
| 1024 | 223 -> 152 | 1867 | 1323 | 60 + 1 + 4 | 1.411x | 1.349x |

EP8, MiMo 2.5 shape (384 experts, 48 per rank, top-8, 6144/2048), 4 redundant slot
slots, same skew (now half of all routed slots land on three experts of one
rank, so the baseline is far more unbalanced than at EP2/EP4):

| M per rank | busiest rank blocks | fixed (us) | EPLB kernel (us) | plan + copy + remap (us) | kernel-only | end to end |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 65 -> 38 | 490 | 320 | 150 + 1 + 2 | 1.54x | 1.04x |
| 256 | 94 -> 55 | 845 | 520 | 236 + 1 + 2 | 1.63x | 1.12x |
| 1024 | 309 -> 191 | 2643 | 1540 | 196 + 1 + 3 | 1.72x | 1.52x |
| 4096 | 1207 -> 747 | 10183 | 6394 | 287 + 1 + 7 | 1.59x | 1.52x |

EP8, DeepSeek V4 Flash shape (256 experts, 32 per rank, top-6, 4096/2048),
4 redundant slots, same skew:

| M per rank | busiest rank blocks | fixed (us) | EPLB kernel (us) | plan + copy + remap (us) | kernel-only | end to end |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 55 -> 27 | 314 | 188 | 115 + 1 + 2 | 1.67x | 1.03x |
| 256 | 77 -> 45 | 649 | 299 | 108 + 1 + 2 | 2.17x | 1.61x |
| 1024 | 271 -> 148 | 1680 | 885 | 110 + 1 + 3 | 1.90x | 1.69x |
| 4096 | 1070 -> 604 | 6066 | 3400 | 125 + 1 + 8 | 1.78x | 1.72x |

* The kernel-only gain tracks the planned reduction of the busiest rank's
  block count, as it should. The same hot experts hurt more as the EP width
  grows because they are a larger share of one rank's work.
* At EP8 the planner costs 110-290 us per call, which is what caps the
  decode-size batches at parity. The EP1 figure (6-13 us) says most of it is
  waiting for the slowest peer's launch, but the 8-rank planner compute has
  not been measured in isolation yet.
* A cold copy at EP8 moves up to four slots per rank (19 replicas in total)
  and takes 0.5-1.1 ms rank-maximum; it is paid once per slot change.
* These skews are synthetic. Replays of routing captured from the two models
  on GSM8K and ShareGPT are the next measurement.
* Reaching 8 visible GPUs on the shared nodes needed the test to run in the
  batch step itself: an overlapping `srun` step inside a holder job only
  received half of the job's GPUs.
* The "plan" column at EP2 is mostly peer wait: the same planner measures
  6-13 us at EP1 (48 experts, M 64-1024), where there is nobody to wait for.
  In the fixed arm that skew is absorbed inside the kernel's first NVLink
  barrier instead, so end to end is the fair comparison.
* A cold copy of one redundant slot (20.3 MiB: packed L1/L2, tile-major scales,
  global scales) takes about 170 us over NVLink and is paid only when a slot's
  expert changes; the steady-state loop above copies nothing.
* Two empty redundant slots cost at most 1% at EP1 (50 vs 48 slots). The
  experts-per-wave divisor they force (25 instead of 48) makes no measurable
  difference (A/B with `DG_MXFP4_EPW`).
* Accuracy against the logical-expert reference is unchanged by replication:
  per-token cosine 0.999 mean, 0.998-0.999 min, at every M, with and without
  replicas.

## Known limits

* The planner is serialized with the layer. Overlapping it with the previous
  layer and gating redundant slots on weight readiness inside the kernel are the
  next steps (see TensorRT-LLM's generation scheme and MegaFlux's pipelined
  replication).
* One greedy candidate; MegaFlux evaluates four in parallel warps.
* Both ranks compute the plan redundantly, which is what keeps them
  consistent without a collective.

## Follow-ups

The three follow-ups of the first version are done and described below: the
inherited product-named identifiers and file names are now
`sm90_mxfp4_mega_moe_fused*` / `SM90MXFP4Fused*`; the weight copy and the
readiness gating moved into the kernel (in-kernel weight copy); the planner was replayed
on real routing and measured end to end.

## Real routing traces (2026-10-12)

Captured with `tests/eplb_tools/` from the two models served on SM90 with
512 prompts per dataset, prefill only; one entry per MoE layer.

| model | dataset | layers x tokens | routing skew kappa (min / median / max) |
|---|---|---|---|
| MiMo 2.5 | GSM8K | 69 x 31.5K | 1.37 / 5.74 / 12.69 |
| MiMo 2.5 | ShareGPT | 69 x 326K | 1.06 / 1.31 / 1.77 |
| DeepSeek V4 Flash | GSM8K | 40 x 31.5K | 3.79 / 8.06 / 14.18 |
| DeepSeek V4 Flash | ShareGPT | 40 x 322K | 1.39 / 1.49 / 1.94 |

GSM8K routing is strongly skewed on both models (one DeepSeek expert takes
13.6% of all routed slots); ShareGPT is close to uniform. The first replay
attempts reported wrong outputs for both the EPLB and the fixed-placement
kernel while they agreed with each other exactly: the test's reference
assumed every rank holds the same tokens (true for the synthetic generator,
false for traces). The reference now gathers all ranks' tokens and reduces
per destination rank.

### Replay of real routing, EP8, 4 redundant slots

Four layers per trace (evenly spaced), each rank taking a different slice of
the captured batch, planner and remap checked against the Python reference,
outputs checked against the logical-expert reference (cosine 0.999 in every
case). End-to-end layer time ratio fixed / EPLB (rank maximum, median of 20
calls), kernel-only ratio in parentheses; the four values are the four layers.

| trace | M=256 | M=1024 | M=4096 |
|---|---|---|---|
| DeepSeek V4 Flash, GSM8K | 1.09 / 0.93 / 1.02 / 1.11 (1.37 / 1.30 / 1.29 / 1.39) | 1.38 / 1.33 / 1.12 / 1.24 (1.52 / 1.49 / 1.29 / 1.40) | 1.55 / 1.54 / 1.25 / 1.43 (1.62 / 1.60 / 1.30 / 1.49) |
| DeepSeek V4 Flash, ShareGPT | 0.91 / 0.92 / 0.93 / 0.98 (1.14 / 1.13 / 1.18 / 1.21) | 1.00 / 1.29 / 1.25 / 1.12 (1.11 / 1.43 / 1.50 / 1.28) | 1.04 / 1.10 / 1.15 / 1.15 (1.10 / 1.15 / 1.19 / 1.20) |
| MiMo 2.5, GSM8K | 0.89 / 0.88 / 1.23 / 1.22 (1.05 / 1.05 / 1.45 / 1.54) | 1.05 / 1.01 / 1.63 / 1.69 (1.13 / 1.09 / 1.76 / 1.82) | 1.11 / 1.03 / 1.80 / 1.94 (1.14 / 1.06 / 1.86 / 1.99) |
| MiMo 2.5, ShareGPT | 0.91 / 0.83 / 0.95 / 0.92 (1.08 / 1.06 / 1.15 / 1.12) | 0.96 / 1.00 / 1.21 / 1.16 (1.05 / 1.09 / 1.31 / 1.26) | 1.00 / 0.98 / 1.08 / 1.13 (1.03 / 1.01 / 1.11 / 1.16) |

Absolute layer times, DeepSeek V4 Flash GSM8K layer 0: fixed 417 / 1091 /
3838 us, EPLB 383 / 793 / 2472 us at M = 256 / 1024 / 4096. MiMo 2.5 GSM8K
layer 51: fixed 860 / 2566 / 9555 us, EPLB 704 / 1521 / 4935 us.

What the numbers say:

* The gain is set by the layer's skew. MiMo's later layers (34, 51) are
  strongly skewed and gain 1.6-1.9x at M >= 1024; its early layers are near
  uniform and gain 1-1.1x. DeepSeek's GSM8K routing is skewed in every layer
  (1.1-1.55x at M >= 1024). ShareGPT routing is close to uniform for both
  models and gains 1.0-1.3x.
* The kernel-only gain is always positive (1.01-1.99x). The end-to-end gain is
  the kernel gain minus the planner, 70-150 us per call at EP8, which is why
  M=256 lands at 0.83-1.23x. The planner is the next target; its compute is
  6-13 us at EP1, so most of the EP8 cost is cross-rank skew and launch gaps
  that overlapping it with the previous layer would hide.
* Steady state copies nothing: the redundant slots keep their experts across the
  repeated layer. First-time copies cost 120-220 us per slot.

## End-to-end serving (2026-10-12)

Serving integration: `tests/eplb_tools/sglang_patch/` (SGLang, `--moe-a2a-backend megamoe`, SM90 MXFP4 MegaMoE kernel on both arms). DeepSeek V4 Flash, TP = EP = 4 on one node, `bench_serving` ShareGPT, 256 prompts, 128 output tokens, concurrency 32, request rate inf. The arms differ only in `SGLANG_MEGAMOE_DYNAMIC_EPLB_REDUNDANT_SLOTS` (0 vs 4).

| metric | fixed | 4 redundant slots | change |
|---|---|---|---|
| total token throughput (tok/s) | 1766 | 1858 | +5.2% |
| output token throughput (tok/s) | 498 | 523 | +5.2% |
| mean / median / P99 TTFT (ms) | 1440 / 1100 / 4800 | 1054 / 893 / 4251 | -27% / -19% / -11% |
| mean E2E latency (ms) | 8222 | 7813 | -5% |
| mean TPOT (ms) | 53.4 | 53.2 | 0% |

Prefill improves, decode is unchanged, as the kernel-level study predicts (the balancer only changes the expert placement; decode batches are too small for the imbalance to matter). Integration lessons folded into the adapter: the FP4 group scales arrive as float32 from the loader; the MXFP4 dispatch must run before the FP8 symmetric buffer is created; the balancer must be allocated at weight-build time with one shared physical-slot buffer across layers, otherwise CUDA graph capture runs out of memory. EP8 and prefill-only GSM8K / ShareGPT runs are queued.

### EP8, first serving run (planner on every forward)

DeepSeek V4 Flash, TP = EP = 8, same client settings; prefill-only runs use 4 output tokens (the client rejects fewer). GSM8K accuracy (greedy, zero-shot, 300 questions) is 285/300 on both arms.

| run | metric | fixed | 4 redundant slots | change |
|---|---|---|---|---|
| GSM8K prefill | input tok/s | 2198 | 2171 | -1% |
| GSM8K prefill | mean / median / P99 TTFT (ms) | 847 / 414 / 3939 | 812 / 439 / 3636 | -4% / +6% / -8% |
| ShareGPT prefill | input tok/s | 7324 | 12174 | +66% |
| ShareGPT prefill | mean / median / P99 TTFT (ms) | 1157 / 743 / 4204 | 621 / 599 / 1028 | -46% / -19% / -76% |
| ShareGPT 128-token gen | total tok/s | 5727 | 4053 | -29% |
| ShareGPT 128-token gen | mean TTFT / TPOT (ms) | 610 / 15.1 | 570 / 23.7 | -7% / +57% |

Reading: long ShareGPT prompts fill the 4096-token prefill chunks, where the kernel-level gains apply, and TTFT drops sharply. GSM8K prompts are short, so the per-forward planner cost (43 layers at EP8) cancels the gain. Decode gets slower for the same reason: a plan was computed for every decode step although a decode batch cannot be rebalanced profitably. The adapter therefore now plans only when the local token count reaches `SGLANG_MEGAMOE_DYNAMIC_EPLB_MIN_TOKENS` (default 512) and otherwise remaps onto the last placement, which is correct for any placement. The rerun with that rule is the number reported in the PR.

### EP8, interleaved passes on one node (2026-10-13)

Same arms, but the kernel-level gate (EP8, model shape, remap-only before and
after planning) runs first inside the serving image, and the four passes run
back to back on one node in the order fixed / 4 slots / fixed / 4 slots so
node and cache effects cancel. The planner runs only for forwards with at
least 512 local tokens. The balancer is provably active in its arms: weights
take 31.81 GB per GPU instead of 30.78 GB (43 layers of 4 redundant slots) and
the KV pool is correspondingly smaller.

| run | metric | fixed p1 (cold) | 4 slots p2 | fixed p3 | 4 slots p4 |
|---|---|---|---|---|---|
| GSM8K prefill | input tok/s | 4200 | 4201 | 4121 | 4194 |
| GSM8K prefill | mean / P99 TTFT (ms) | 419 / 428 | 415 / 445 | 426 / 524 | 414 / 438 |
| ShareGPT prefill | input tok/s | 11004 | 11928 | 11919 | 11923 |
| ShareGPT prefill | mean / P99 TTFT (ms) | 705 / 1267 | 645 / 1051 | 651 / 1039 | 649 / 1017 |
| ShareGPT 128-token gen | total tok/s | 5728 | 5690 | 5769 | 5689 |
| ShareGPT 128-token gen | mean TTFT / TPOT (ms) | 612 / 15.1 | 594 / 15.4 | 595 / 15.1 | 597 / 15.4 |
| GSM8K accuracy (300 q) | correct | 288 | 285 | 288 | 288 |

Reading: on these client settings the two arms are identical within 1%. The
first pass of a fresh node is slower on every metric (cold caches and JIT),
which is what the "first serving run" table above measured as a 66% gain;
that table is superseded. Decode is unchanged by design.

### Long prompts, kernel ceiling and the adapter trace (2026-10-13)

Same job layout plus a long-prompt ShareGPT subset (first turns of 8000+
characters, mean 11k characters, about 2750 tokens; 256 prompts, concurrency
64) and the adapter trace. Steady-state passes 3 (fixed) and 4 (4 slots):

| run | metric | fixed | 4 slots | change |
|---|---|---|---|---|
| ShareGPT-long prefill | input tok/s | 26664 | 26804 | +0.5% |
| ShareGPT-long prefill | mean / P99 TTFT (ms) | 3465 / 6674 | 3453 / 6596 | 0% / -1% |
| ShareGPT-long 64-token gen | input tok/s | 24089 | 23594 | -2% |
| ShareGPT-long 64-token gen | mean TTFT / TPOT (ms) | 3359 / 62.4 | 3435 / 63.6 | +2% / +2% |
| GSM8K accuracy (300 q) | correct | 288 | 286 | |

The trace explains why: SGLang sequence-shards the MoE input across the
eight ranks, so a 4096-token prefill chunk reaches each rank's MoE as about
500 tokens. Planned layer forwards averaged 240-422 local tokens, decode
steps 10, and a plan made 0-2 weight copies on rank 0. That is the regime the
kernel-level study marks as break-even: the serialized planner (75-125 us
per layer) cancels the kernel gain below about M = 1024 per rank.

Kernel ceiling at the serving shape, measured in the same job (EP8, real
routing, four layers each, rank-maximum layer time, end-to-end speedup
including plan, copy and remap / kernel-only):

| trace | M per rank | fixed (us) | EPLB (us) | speedup, four layers |
|---|---|---|---|---|
| GSM8K | 1024 | 922-1072 | 788-818 | 1.17-1.32x / 1.30-1.51x |
| GSM8K | 4096 | 3092-3824 | 2451-2537 | 1.24-1.56x / 1.29-1.63x |
| ShareGPT | 1024 | 773-1063 | 795-826 | 0.94-1.32x / 1.10-1.49x |
| ShareGPT | 4096 | 2731-2931 | 2520-2583 | 1.06-1.15x / 1.10-1.20x |

So the balancer pays off once each rank's MoE sees 1024+ tokens per forward:
16384-token chunks under TP8 (2048 per rank), or DP attention where every
rank keeps its own 4096-token chunk. Those two configurations are the next
run. With DP attention the ranks hold different token counts, so the adapter
now derives the planning decision and the planner block size from the
batch's published global token counts (identical on all ranks) instead of
the local count, which also fixes the sequence-sharded case where local
counts can differ by one.

### 16384-token chunks and DP attention (2026-10-13)

Job with the kernel gate (PASS), the kernel bench at M = 512 to 4096 and six
serving arms. The two DP-attention arms (`--enable-dp-attention --dp-size 8`,
EP8) never served: every scheduler died at the first prefill with an
unspecified launch failure surfacing in the layer after the MoE, with and
without the balancer, so it is a base-path problem of the MegaMoE backend
under DP attention in this image, not of the balancer. SGLang also divides
the chunk by the DP size (`chunked_prefill_size` became 512), so that
configuration would not have given the MoE large batches anyway. Dropped.

TP = EP = 8 with 16384-token chunks (cap 16384) served fine. The trace
confirms the regime: planned layer forwards averaged 1905 local tokens,
remap-only forwards 55. Steady-state passes 5 (fixed) and 6 (4 slots); pass
1 (fixed, cold node) is excluded as before:

| run | metric | fixed | 4 slots | change |
|---|---|---|---|---|
| GSM8K prefill, conc 64 | input tok/s | 10569 | 9909 | -6% |
| GSM8K prefill, conc 64 | mean / P99 TTFT (ms) | 317 / 388 | 341 / 407 | +8% / +5% |
| ShareGPT-long prefill, conc 128 | input tok/s | 47346 | 47715 | +0.8% |
| ShareGPT-long prefill, conc 128 | mean / P99 TTFT (ms) | 4147 / 7337 | 4096 / 7312 | -1% / 0% |
| ShareGPT-long 64-token gen | input tok/s | 41617 | 41542 | -0.2% |
| ShareGPT-long 64-token gen | mean TPOT (ms) | 69.4 | 69.7 | 0% |
| GSM8K accuracy (300 q) | correct | 285 | 288 | |

Kernel ceiling at this per-rank batch from the same job (M = 2048, EP8, four
layers, end to end including plan, copy and remap): GSM8K routing fixed
1631-1993 us vs 1355-1396 us (1.20-1.47x), ShareGPT routing 1548-1871 us vs
1360-1378 us (1.12-1.37x). The serving arms realize far less than that, so
either the MoE layers are a smaller share of the step than the kernel bench
implies, or the plan inside the server is weaker than on the captured
traces. The GSM8K arm is the break-even band: about 800 tokens per rank, so
the planner runs on every layer and does not recover its cost.

Next run, same script: the adapter now times every N-th large MoE layer
forward inside the server with CUDA events (both arms, read back one period
later so nothing stalls) and prints, for the balanced arm, the rank loads of
the plan that forward used (static-placement-only vs planned, in token blocks, from the
planner's copy table). That gives the per-layer MoE time fixed vs balanced
at the same point of the same request stream, and shows whether the plan
replicates. A third arm plans only from 1024 tokens per rank.

### Inside the server: per-layer MoE time, fixed vs balanced (2026-10-13)

Six arms at 16384-token chunks (TP = EP = 8), order fixed / 4 slots / 4
slots with planning from 1024 tokens per rank / fixed / 4 slots / 4 slots
from 1024, with the adapter timing every 150th MoE layer forward of at least
512 local tokens (25 per arm, 24 of them at 1900-2000 local tokens, same
points of the same request stream in every arm).

| arm | layer forward, mean (min-max) us | static-placement-only max / mean load | planned max / mean | replicas |
|---|---|---|---|---|
| fixed, pass 1 | 1684 (1439-2079) | | | |
| fixed, pass 4 | 1690 (1444-2109) | | | |
| 4 slots, pass 2 | 1729 (1623-1815) | 1.316 | 1.003 | 7.7 |
| 4 slots, pass 5 | 1731 (1619-1844) | 1.316 | 1.002 | 7.6 |
| 4 slots from 1024, pass 3 | 1727 (1610-1811) | 1.316 | 1.003 | 7.6 |
| 4 slots from 1024, pass 6 | 1723 (1642-1801) | 1.317 | 1.002 | 8.0 |

Loads are token blocks per rank from the planner's copy table. So inside the
server the plan does what it should: the hottest rank carries 1.32x the mean
with fixed placement and 1.00x after planning, with about one replica per
rank, and the layer time loses its layer-to-layer spread (fixed 1440-2110
us, balanced 1620-1845 us). But the balanced layer is 2.5% slower on
average, not faster, and that is exactly the end-to-end result (all arms
within 1%, ShareGPT-long 46.8-47.7k input tok/s, GSM8K accuracy 285-289 of
300 on every arm). Planning only from 1024 tokens per rank changes nothing
at this chunk size, and it removes the GSM8K penalty seen earlier (about 800
tokens per rank; 10462 vs 10457 fixed input tok/s).

Why the kernel ceiling does not show up: the kernel bench repeats the same
routing, so after the first iteration a plan changes no slot and the weight
copy is free (1.2 us). In the server consecutive chunks route differently,
so each plan re-fills about one redundant slot per rank per layer, a 12.6 MB
MXFP4 expert over NVLink, serialized in front of the kernel together with
the planner. With a 1.32x imbalance the kernel can save at most about a
quarter of its time, and that is what the plan and the copies cost. The
next run splits the traced time into prepare (plan, copy, remap) and kernel
and prints the copy count per plan, and adds two arms that amortize the
copies: `SGLANG_MEGAMOE_DYNAMIC_EPLB_REPLAN_EVERY=4` (a layer re-plans every
4th prefill forward and remaps the others onto the last plan) and
`SGLANG_MEGAMOE_DYNAMIC_EPLB_OPEN_GATE_BLOCKS=16` (a fresh replica must be
worth 16 blocks).

## Division of labour with METRO (decided 2026-10-14)

Two balancers, one per regime, by decision of the project resident rank after the
planner comparison on identical real routing traces
(`2026-10-12-metro-vs-megamoe-eplb.md`):

- **Prefill: this balancer.** Large per-rank batches, compute-bound kernel;
  the plan minimizes the busiest rank's token blocks and moves weights into
  redundant slots. On the traces it brings the busiest rank to within 0.5 block
  of the mean at M = 1024 per rank, where METRO's one-replica-per-expert
  routing cannot (it raised the block count above static EPLB).
- **Decode: METRO (moe_load_balancer, routing-only).** Fixed redundant
  placement, no weight movement, a per-layer choice of replica that
  minimizes the busiest rank's activated experts; 4-7 us per layer at EP8.
  On the traces it reaches the same busiest-rank expert count as this
  balancer at M = 8 per rank with zero copies, where this balancer would
  copy 3.5-5.6 slots per step.

Hand-over point in the serving adapter: forwards below
`SGLANG_MEGAMOE_DYNAMIC_EPLB_MIN_TOKENS` tokens per rank (decode steps, small
chunks) skip the planner and only remap onto the last plan. That branch is
where the decode balancer attaches: the redundant slots already hold replicas,
and a routing-only choice among them needs the remap kernel and no copy.
Remaining work for the prefill side is to hide the plan and copy cost
(re-plan interval, overlap with the previous layer), which the in-server
timing above identifies as the whole gap between the kernel ceiling and the
end-to-end result.

## Acceptance line and the path to it (2026-10-14)

The MegaFlux paper (arXiv 2610.00671) reports, on 8 GPUs at EP8, a MoE-layer
geomean of 1.45x (peak 2.14x) against the same megakernel with fixed
placement under trace-derived skewed routing at 1K-64K tokens per rank, and
1.13-1.26x end to end for vLLM prefill of DeepSeek-V4-Pro on LongBench-Pro
prompts with 16K/32K chunks. It has no end-to-end number above 1.26x. The
accepted acceptance line for this branch is therefore: kernel-level about
1.45x geomean on skewed real routing (reached: 1.1-1.55x per layer on DSv4
Flash GSM8K routing, 1.6-1.9x on the late MiMo 2.5 layers), end to end
1.15-1.25x on long-prompt datasets.

The in-server trace (previous section) located the end-to-end gap: the plan
and the weight copies it triggers run serially before the kernel and cost as
much as the kernel saves. Three changes close the gap in the order of their
expected effect.

### Deferred plan (`SGLANG_MEGAMOE_DYNAMIC_EPLB_DEFERRED=1`)

After a layer's first plan, a prefill forward only remaps onto the plan made
from its previous prefill forward and launches the kernel. It then records an
event after the kernel, and on one side stream shared by all layers waits for
that event, plans from a private int64 copy of this forward's logical
routing (the shared symmetric buffer moves on to the next layer), and copies
the weights the new plan needs. Nothing of this is on the critical path: the
plan and copies for layer L run while layers L+1... of the same step compute.

Hazards and how they are avoided:
- The copy writes this rank's redundant slots of layer L, which the kernel of
  layer L may still be reading: the side stream waits for the kernel's event.
  Copies pull from the resident ranks' resident slots, which are never written.
- The planner pushes histograms into every rank's workspace and spins on
  per-rank generation flags. Rank A's plan for chunk c+1 of layer L is issued
  after A's layer-L kernel of chunk c+1, which is a collective with B's, which
  B issues only after waiting for its own side-stream plan of chunk c. So no
  rank can be a plan ahead of another on the same workspace.
- Every forward of the layer, large or small, first makes the main stream
  wait for the layer's pending plan, so decode steps and the next chunk see a
  consistent placement. The last decoder layer drains all pending plans at
  the end of each step, so CUDA-graph decode replays (which run no Python)
  never overlap a weight copy.
- `REPLAN_EVERY=K` still applies: the deferred plan is issued every K-th
  prefill forward.

The plan is one chunk stale. The trace showed plan loads nearly identical
between consecutive chunks of the same request mix, and the remap-only path
is correct for any placement, so the cost of staleness is a slightly less
exact balance, not an error.

### Tokens per rank and datasets

SGLang TP8 sequence-shards the MoE input, so a 16384-token chunk is about
2K tokens per rank, the low end of the paper's range, where it reports 1.29x
rising to 1.77x at 64K per rank. The arms script now has `tp8c64k`
(65536-token chunks, about 8K tokens per rank, `--max-prefill-tokens`
raised to match, symmetric buffer cap unchanged at 16384). LongBench (English
tasks, 256 prompts of 8K-48K characters, converted by
`tests/eplb_tools/longbench_to_sharegpt.py`) is the long-prompt dataset,
served prefill-only (4 output tokens) at concurrency 32; GSM8K and ShareGPT
remain as the short-prompt and near-uniform controls.

### MiMo 2.5 end to end

MiMo-V2.5-Pro (384 experts, top-8, hidden 6144, 69 MoE layers) loads its
experts through the MXFP4 quantization method rather than the FP8 method
DeepSeek V4 uses, so the installer now also patches `mxfp4.py` (SM90 with the
megamoe backend builds the SM90 MXFP4 MegaMoE weights and the balancer arena)
and `MiMoV2MoE.forward_normal` (dispatch through `should_use_mega_moe` like
`DeepseekV2MoE`). The layer communicator already scatters and gathers tokens
for any a2a backend, so the sequence-sharded MoE input is the same as for
DeepSeek. The adapter accepts `*_scale` as well as `*_scale_inv` parameter
names and a missing routed scaling factor. The arms script takes
`MODEL_KIND=mimo25` (adds `--attention-backend fa3 --context-length 32768`,
kernel gate on the MiMo shape).

Runs submitted on 2026-10-14: DSv4 Flash fixed vs deferred at 16k and 64k
chunks, and MiMo 2.5 fixed vs deferred at 16k chunks, both with GSM8K,
ShareGPT-long and LongBench. Results go to the end-to-end section when they
land.

### First deferred-mode and MiMo runs (2026-10-14)

Both on healthy nodes (the probe reads 780-810 TFLOPS per GPU), TP8 = EP8,
16384-token chunks, interleaved fixed / deferred passes, input tok/s of the
steady-state passes (pass 1 discarded for GSM8K, which is cold), GSM8K
accuracy on 300 questions.

DeepSeek V4 Flash, fixed vs deferred (4 redundant slots):

| | GSM8K | ShareGPT-long | ShareGPT-long gen64 | LongBench | accuracy |
|---|---:|---:|---:|---:|---|
| fixed | 9893 | 46791 / 47474 | 41480 / 41663 | 44580 / 44682 | 286 / 286 |
| deferred | 10305 / 10505 | 47114 / 47155 | 41121 / 41091 | 44321 / 44289 | 291 / 287 |

In-server trace (62 forwards per arm at 1900-2050 tokens per rank; this
trace still started before the FP8 cast): fixed 1719 / 1741 us median per
MoE layer; deferred 1954 / 1894 us = 328 us before the kernel + 1629 / 1572
us kernel. The plan lines show the imbalance the kernel sees being removed
(for example max rank load 341 -> 208 blocks, mean 208, 7-11 replicas) and
0-1 weight copies per rank per layer, yet the layer is 10 % slower end to
end because of the 328 us in front of the kernel, which is far more than a
remap. The trace has since been split into wait / prepare / kernel and
moved after the cast (commit 87e5b10) to attribute it; the rerun is job
4997843. The 64k-chunk arms did not start: SGLang requires the MegaMoE
token cap to cover the whole chunk (it does not account for the sequence
sharding), so `tp8c64k` now sets the cap to 65536.

MiMo 2.5 Pro, same arms (first run with the MXFP4 MegaMoE path):

| | GSM8K | ShareGPT-long | ShareGPT-long gen64 | LongBench | accuracy |
|---|---:|---:|---:|---:|---|
| fixed | 12086 | 33250 / 33245 | 28453 / 28468 | 32490 / 32456 | 1 / 300 |
| deferred | 11482 / 12031 | 38669 / 38761 | 32599 / 32549 | 37353 / 37341 | 1 / 300 |

Per MoE layer (94 forwards per arm at 2048 tokens per rank): fixed 4703 /
4738 us median, deferred 3123 / 3148 us (321 us before the kernel + 2805 /
2827 us kernel), i.e. the balanced layer is 1.5x faster and the server is
16 % faster on ShareGPT-long and 15 % on LongBench. The plan lines show the
MiMo imbalance (max rank load 594 -> 285 blocks on a late layer). These
numbers are not claimable yet: GSM8K accuracy is 1/300 in both arms, so the
MiMo serving path is wrong somewhere common to both (expert weight layout
through the MXFP4 quant method, the attention backend, or prompt format).
Job 4997842 runs the same prompts through the MegaMoE path and the stock
expert path with two attention backends to locate it.

### MiMo 2.5 root cause: checkpoint scales below the kernel's E8M0 window (2026-10-15)

The correctness probe (same three prompts and 60 GSM8K questions per
configuration) placed the fault: the MegaMoE path produced the same garbage
with the fa3 and the default attention backend (0/60), while the stock MXFP4
Marlin runner was right with both (56/60, 57/60). The image's stock triton MoE
runner cannot run packed MXFP4 experts on SM90 ("Hidden size mismatch"), so
Marlin is the stock baseline for this checkpoint. A first probe ran the
image's own SGLang and DeepGEMM because the patched trees were not on the
path; every arm failed at load time.

`tests/eplb_tools/check_ckpt_experts.py` then ran one layer's real experts
through the adapter's transform and the kernel at EP1 against a dequantized
reference, with the byte-layout alternatives alongside:

| weights | adapter before the fix | bytes unpermuted | high nibble first | [up; gate] | after the fix |
|---|---:|---:|---:|---:|---:|
| MiMo 2.5 layer 5, 16 experts (codes 114-125) | 0.833 | -0.03 | -0.06 | 0.62 | **0.9999** |
| MiMo 2.5 layer 40, 48 experts (codes 118-125) | 0.933 | -0.00 | 0.00 | 0.65 | **0.9997** |
| DeepSeek V4 Flash layer 5, 16 experts (codes 119-123) | 0.965 | 0.00 | -0.00 | 0.59 | **0.9996** |

(per-token cosine, mean). The nibble order and the gate/up order were right;
the output was simply too small, and more so where the scales were smaller.
`quantization/mxfp4_fused_scale.cuh` explains it: the SM90 kernel dequantizes
E2M1 into E4M3 by folding the group's E8M0 scale into the E4M3 exponent, which
is exact only while every magnitude of the group is a normal E4M3 number,
0.5 * 2^k >= 2^-6 and 6 * 2^k <= 448, i.e. k in [-5, 6], codes 122..133.
Below that window groups flush to zero. The kernel bench quantizes
`randn * 0.05` weights, whose scales land at code 122, the edge of the window,
so the gate never saw it; checkpoints quantized against absolute magnitudes
sit below it (MiMo 2.5 codes 114-125, with 2^-13 groups; DeepSeek V4 Flash
119-123). DeepSeek V4 Flash was therefore also losing its smallest groups
(cosine 0.965) while still scoring 286/300 on GSM8K.

Fix (commit bda740e): `fold_e8m0_into_kernel_window` shifts every expert's
codes so that its largest code is 133 and returns the shift as the kernel's
per-expert fp32 global scale, which both GEMM epilogues already apply
(`l1_global_scales`, `l2_global_scales`; the EPLB arena carries them per slot
and the copy kernel moves them with the weights). Per-expert code spans are
3-10 on both checkpoints, inside the 12-code window, so no group is lost
(the redundant slot reports the lost fraction; it is 0 here). The adapter applies the
fold for both models; for scales already inside the window it is the identity
up to the global factor. The MiMo arms were resubmitted with the fix; the
DeepSeek V4 Flash arms that were already running picked the fixed adapter up
mid-job (each arm launches a new server), which does not change their timing
but means their accuracy columns mix both versions; a clean rerun follows.

### Serving results with the scale fix, 16k chunks (2026-10-15)

All arms below run the folded scales (commit bda740e). TP8 = EP8, 16384-token
chunks (about 2000 tokens per rank), input tok/s of the steady-state passes,
GSM8K accuracy on 300 questions. The adapter trace starts after the FP8 cast
and splits the balancer's own time into wait (for the deferred plan), prepare
(plan + copy + remap, or remap only) and kernel; medians over 63-94 full-chunk
forwards per arm.

DeepSeek V4 Flash:

| arm | GSM8K | ShareGPT-long | SG-long gen64 | LongBench | accuracy | layer (us) |
|---|---:|---:|---:|---:|---|---|
| fixed (3 passes) | 10377 / 8704 | 47210 / 47147 | 41363 / 41684 | 45015 / 44770 | 288 / 287 | kernel 1572-1619 |
| deferred 4 slots (3 passes) | 9808 / 9726 | 46915 / 47087 | 41064 / 41157 | 44289 / 44390 | 288 / 289 | 33 wait + 85 remap + kernel 1587-1630 |

MiMo 2.5 Pro:

| arm | GSM8K | ShareGPT-long | SG-long gen64 | LongBench | accuracy | layer (us) |
|---|---:|---:|---:|---:|---|---|
| fixed (2 passes) | 11674 / 11774 | 39094 / 39019 | 32421 / 32281 | 37436 / 37452 | 296 / 292 | kernel 2856 / 2890 |
| synchronous plan, 4 slots | 11975 | 38963 | 32038 | 37865 | 292 | 351 prepare + kernel 2561-2584 |
| deferred 4 slots (3 passes) | 12095 / 11694 / 11546 | 38235 / 38308 / 38318 | 31693 / 31656 / 31628 | 36740 / 36728 / 36708 | server died at the eval | 33 wait + 88 remap + kernel 2808-2838 |
| deferred, no CUDA graph | 6262 | 36190 | 22903 | 34959 | 294 | 33 + 86 + kernel 2817 |

What the numbers say:

* With correct numerics the earlier MiMo "+15-16 %" is gone; it was an
  artifact of the zeroed weights. On these datasets the static-placement-only imbalance
  that reaches the planner is 1.16x (MiMo) and 1.32x (DeepSeek) at the
  median, p90 1.37x / 1.69x, far below the GSM8K-only traces (kappa up to 14)
  because a 16k chunk mixes many prompts and the sequence sharding gives
  every rank the same token count.
* A fresh plan does speed up the MiMo kernel: synchronous plan 2561-2584 us
  vs fixed 2856-2890 us (1.11x), but its 351 us of plan + copy + remap on the
  critical path returns the layer to parity (2938 vs 2890 us) and the server
  to within 1 %.
* The deferred plan is one chunk stale, and that costs almost the whole
  kernel gain: deferred kernel 2808-2838 us (1.02x over fixed) against
  2561-2584 us with a fresh plan. Consecutive chunks route differently enough
  that last chunk's placement does not fit this chunk. On DeepSeek the
  deferred kernel is not faster at all (1587-1630 vs 1572-1619 us). The
  balancer's remaining cost in deferred mode is 118 us per layer (wait 33 +
  remap 85, the latter host-side launch overhead measured by the events).
* The deferred MiMo arm dies at the first request after LongBench when the
  full benchmark sequence (GSM8K, ShareGPT-long, 64-token generation,
  LongBench, eval) has run: TMA descriptor init error 719 after an earlier
  launch failure (reproduced twice). It does not
  happen with LongBench + eval alone, with the synchronous plan,
  or with CUDA graphs disabled, so it is an interaction between the
  side-stream plan/copy and the decode CUDA-graph replays that the
  generation phase exercises. The synchronous arm is correct on both models
  (DeepSeek 288/300 earlier, MiMo 292/300).
* 64k chunks do not fit: the 65536-token symmetric buffer OOMs at decode
  graph capture on both arms. 32k chunks (CAP 32768, static fraction 0.72,
  decode graph batch 128) are the largest; the first 32k attempt failed on an
  ambiguous flag name (`--cuda-graph-max-bs-decode`), resubmitted, fixed vs synchronous plan.

Decision: the synchronous plan is the balancer's serving mode; deferred mode
stays behind its flag as experimental until the graph interaction is found.
The path to a prefill gain is larger per-rank batches (32k chunks) where the
kernel saving grows and the 350 us plan cost does not.

## Final design and results (2026-10-20)

Production path, prefill only, DeepSeek V4 Flash at EP8 unless stated:

* **Static expert placement** (`tests/eplb_tools/build_static_placement.py`,
  applied through SGLang's `--init-expert-location` with static dispatch):
  one choice per layer of which rank owns which expert, from the first 25 %
  of a calibration window of the routing. Zero cost per call.
* **Plan + in-kernel weight copy + remap**: with redundant slots the planner runs once per
  large chunk (one CTA, 37-40 us in-kernel at 2048 tokens per rank), the
  MegaMoE kernel itself copies the changed redundant slots over NVLink while it
  computes the resident blocks and gates the dependent tiles on per-slot
  readiness counters (the default; `DG_EPLB_IN_KERNEL_COPY=0` restores the separate
  copy kernel as a measurement mode), and the remap rewrites the
  routing to physical slots. The per-expert global scales are read after the
  GEMM main loop for redundant slots (reading them at task start raced with
  the in-kernel weight copy); resident slots read theirs at task start.
* **Removed**: the deferred plan (one chunk stale, 1.02-1.06x kernel on
  ShareGPT and a CUDA-graph interaction), the fused-plan prototype (the
  planner inside the kernel's first phase; 137-569 us per layer slower than
  the single-launch plan at EP4 because its hand-off tail serialises), and
  the early plan on a side stream (see the 2026-10-22 section: it halves the
  prepare segment and gives the same back in the kernel and the shared
  experts).

Kernel level, real GSM8K routing replayed outside the server, EP8, 4 redundant slot
slots, rank-maximum time, geomean over four layers: kernel-only 1.42x at
1024 and 1.50x at 4096 tokens per rank; with the synchronous plan, copy and
remap on the critical path 1.26x and 1.44x. Inside the server the per-layer
MoE time under the static expert placement is 1.21x.

End to end (SGLang `bench_serving`, prefill-only 1000 few-shot prompts of
about 1k tokens, input token throughput, means over the interleaved passes):

| chunk | dataset, concurrency | fixed | static expert placement | permutation + 4 slots | 4 slots |
|---|---|---:|---:|---:|---:|
| 16k | GSM8K c128 | 45,228 | +5.3 % | +5.0 % | +6.4 % |
| 16k | GSM8K c256 | 45,708 | +5.3 % | +6.1 % | +6.1 % |
| 16k | MMLU c128 | 44,059 | +1.4 % | +2.8 % | +2.9 % |
| 16k | MMLU c256 | 46,026 | +0.8 % | +2.2 % | +2.1 % |
| 32k | GSM8K c128 | 46,670 | +5.5 % | +7.8 % | +8.2 % |
| 32k | GSM8K c256 | 45,234 | +4.0 % | +6.2 % | +7.1 % |
| 32k | MMLU c128 | 45,263 | +2.0 % | +4.1 % | +3.8 % |
| 32k | MMLU c256 | 43,504 | +1.1 % | +2.7 % | +3.1 % |

Mean TTFT moves the same way (-2 % to -7 %). GSM8K 300-question accuracy
285-291 / 300 on every arm. The gain is bounded by the MoE share of the
prefill step (19-30 % here): the kernel ceiling of 1.66x corresponds to
about +12 % end to end.

Decode was measured separately with a replica router on static redundant slot-slot
placement (METRO-style). The kernel gains 4.5-7.8 % but the per-layer route
and count exchange costs 11.5-14 us, which over 40 layers exceeds the saving;
TPOT ends 1-6 % worse at 8-64 concurrent requests. That line is not in this
branch; the only route to a positive result is replica selection inside the
MegaMoE dispatch phase.

## Where the remaining prefill gap is (2026-10-22)

Question: today's +5 to +8 % end to end against the derived ceiling of about
+12 %. Every term below was measured on the final tree; each is either
recovered or attributed to a measured cause.

### Inside the server (TP8, 16k chunks, GSM8K 1k, 278 prefill forwards per arm)

The adapter trace times every 50th prefill forward on rank 0 with CUDA events
after the FP8 cast; the forwards carry the same ordinals and token counts in
every arm (median 1848 tokens per rank), so the arms pair by forward.

| arm | prepare us | kernel us | total us | busiest rank, blocks |
|---|---:|---:|---:|---|
| fixed placement | 0 | 1670 (mean 1698) | 1670 | static-placement-only 234 (contiguous: 278) |
| permutation + 4 slots | 70 | 1216 | 1287 | planned 190 vs mean 190 |
| same, plan on a side stream | 47 | 1235 | 1284 | planned 191 vs mean 190 |

Paired kernel ratio fixed / balanced 1.357x (median; mean 1.389x), total
1.283x. The planner reaches the block ideal (busiest rank equals the mean in
the median forward, at most 20 blocks above it), so neither more redundant slot
slots nor a better planner buys anything here; the host planner study on
the 1k traces agrees (S = 4: 0.3-0.6 % above ideal, the exact optimum
0.1-0.2 %; S = 2: 2-5 %; S = 1: 11-14 %).

A torch-profiler window of the same server gives the MoE share directly:
the MegaMoE kernel is 27.2 % of GPU busy time in the fixed arm (mean large
launch 1773 us) and 20.6 % + 2.7 % planner in the balanced arm. On that
share the per-layer times reproduce the measured gain (+6.9 % predicted,
+6.8 % measured), and the kernel's own ceiling on this workload is about
+10 %: the fixed arm's time is 94 us + 5.40 us per busiest-rank block
(R^2 1.00), which at the balanced load (190 blocks) is 1119 us. The balanced
path costs 1287 us, 168 us more. That excess decomposes as follows.

### Term by term

1. **Prepare, 70 us per layer (5.4 % of the balanced layer): planner wait
   and launch gap, not planner work.** The planner's own phase record, read
   from the workspace in the server (TP8, 16k chunks, 158 traced large
   forwards, median / p90 in us): total 38.8 / 71.2 = histogram 1.6 +
   publish 2.5 + peer wait 15.3 / 46.6 + setup 1.7 + greedy 5.6 + prune 4.5
   + slot assignment 1.8 + table publish 0.9 + remap 3.2. The compute sum
   is 23 us, the same as in the quiet kernel bench (21-25 us at EP8), so a
   faster greedy would return at most 6-10 us per layer, under 0.5 % of the
   layer, and was not built. The rest of the segment is the wait for the
   slowest rank's histogram (the ranks arrive skewed by host jitter; the
   fixed arm pays the same skew inside its kernel's first barrier) and the
   gap between the two launches. The host is not the cause: in the recorded
   profiler traces the MegaMoE kernel starts a median 5.6 ms after its
   launch call returned, cudaLaunchKernel costs 4 us, and there is no
   per-layer host-device synchronisation in either arm. Two alternatives
   were measured and closed: the plan on a side stream after top-k (prepare
   47 instead of 70 us, kernel +19 us, total 0.997x, end to end GSM8K c128
   46,752 vs 48,303 and MMLU 45,739 vs 45,329; at 32k 50,176 vs 50,143) and
   the kernel translating logical experts to slots itself so that no remap
   launch precedes it (the dispatch warps reserve per-expert ordinals and
   rewrite the routing in place; valid remap and the remap kernel's per-slot
   counts in 28 cells, output within the remap path's own run-to-run spread;
   kernel +0 to +10 us, plan alone 23-25 us against plan+remap 23-25 us in
   one launch; per layer 2420 vs 2418 us at TP4, end to end within noise).
   The translation path is kept on the `megamoe_mxfp4_prefill_ceiling`
   branch as the record; neither is in this tree.
2. **The in-kernel-copy instantiation, about 35 us per layer.** With zero copies
   pending the EPLB kernel's L1 task takes 31.6 us against 30.3 us in the
   fixed kernel (+4.5 %), L2 15.45 against 15.3 (+1 % after the fix below).
   Not the extra physical slots (the EPLB kernel with the in-kernel weight copy compiled
   out runs the same routing at 1.00-1.02x of the fixed kernel), not the
   global scales (fixed + scales = fixed), not the loader's per-task readiness
   check, not the dispatch warps' copy code, not the coherent scale load:
   compiling each of them out leaves L1 at 31.2-31.9 us. ptxas reports 168
   registers, no spills and no stack for both instantiations; the fused one
   is 5496 against 5024 SASS instructions, all of it in the dispatch warps'
   copy path. The L2 part was recovered: the per-expert global scale is now
   read at task start for resident slots (never a copy target) and after the main
   loop only for redundant slots, which took the L2 task from 15.9 to 15.45 us;
   pinning the load into the first k-block with an asm barrier made every
   task 2 % slower and was dropped. The L1 part stays attributed to code
   placement of the larger binary; it is worth about +0.9 % end to end.
3. **Rank finish spread at equal blocks, 10 us per layer.** With the planned
   loads equal within 1-3 blocks the ranks' GEMM phases end 20 us apart in
   the median cell (36 us at p90, out of 1640 us at EP4). A fit of the
   per-rank finish on blocks, distinct experts and partial last blocks
   explains 16 % of it (7.3 us per block, 0 per expert, a partial block costs
   1.11 of a full one, not less); the busiest rank's excess over the mean rank
   is 10.5 us median, 22 us p90, of which the block count explains 1.8 us.
   No per-task cost model in the planner can reach that, so the term is
   attributed to the kernel's launch-to-launch scheduling noise.
4. **Everything outside the MoE layer, 73 % of the step.** Attention 19 %,
   dense GEMMs 18.5 %, collectives 17 %, other 17 % of GPU busy time in the
   fixed arm; not touched by the balancer and the reason a 1.36x kernel is a
   +7 % step.

### Loss table, TP8, 16k chunks, per MoE layer

| | us | share of the fixed layer |
|---|---:|---:|
| fixed placement, measured | 1670 | 100 % |
| ideal with this kernel at the balanced load (94 + 5.40 x 190) | 1119 | 67 % |
| balanced, measured: prepare 70 + kernel 1216 | 1287 | 77 % |
| of which planner compute (greedy 5.6, prune 4.5, remap 3.2, the rest 7) | 23 | 1.4 % |
| of which peer wait and launch gap | 47 | 2.8 % |
| of which in-kernel-copy instantiation (L1 task +4.5 %, L2 +1 %) | 35 | 2.1 % |
| of which rank finish spread at equal blocks | 10 | 0.6 % |
| of which unexplained kernel excess over the fixed-arm cost model | 52 | 3.1 % |

The balanced arm realises 1.283x of the 1.49x the block count allows; the
MoE share of the step (27 %) turns that into the measured +6.8 % against a
+10 % ceiling for this kernel on this workload.

### Tools added for this study

`tests/test_mxfp4_mega_moe_sm90_eplb.py --phase-stamps 1` prints, per bench
cell, the in-kernel phase stamps and per-task times of the fixed kernel, the
fixed kernel with global scales, the EPLB kernel with empty redundant slots on
the resident rank routing (the slot cost), the EPLB kernel with the in-kernel weight copy
compiled out, and the planned EPLB kernel, plus a per-rank table (blocks,
distinct experts, partial last blocks, GEMM finish) and the host cost of
each launch; `--translate 1` checks the in-kernel translation against the
remap path (valid remap, equal per-slot counts, output within the remap
kernel's own run-to-run spread) and times both. The adapter trace splits
the host time into decision, prepare, argument glue and kernel call.

### End to end after this study

TP8, 16k chunks (input tok/s, GSM8K 1k c128 / c256, MMLU 1k c128 / c256):
fixed 45,216 / 45,797 / 44,061 / 45,776; permutation + 4 slots 48,303 /
48,929 / 45,329 / 47,113 (+6.8 / +6.8 / +2.9 / +2.9 %). 32k chunks: fixed
46,529 / 40,807 / 45,236 / 43,386; permutation + 4 slots 50,143 / 43,016 /
46,887 / 44,452 (+7.8 / +5.4 / +3.6 / +2.5 %; the c256 cells at 32k are
unstable with the smaller static memory fraction). TP4 (EP4), 16k chunks:
fixed 36,718 / 37,037 / 35,913 / 37,020; 4 slots 37,712 / 38,865 / 36,985 /
37,532 (+2.7 / +4.9 / +3.0 / +1.4 %); with the in-kernel translation 38,600 /
38,953 / 36,857 / 37,883 (+5.1 / +5.2 / +2.6 / +2.3 %), within the noise of
the remap path on the GPU timeline.
