# METRO (moe_load_balancer) vs MegaMoE dynamic EPLB (DeepGEMM): comparison and planner benchmark

Date: 2026-10-12. Target: SM90 (Hopper), EP8 unless stated.
Sources: B = `moe_load_balancer` branch `feat/metro-fused2` (`policies/l2/metro.py`, `adapters/sglang/metro/runtime.py`, `docs/metro-user-guide.md`, `tests/test_metro.py`, `tests/dist_metro_fused.py`, `tests/dist_metro_contention.py`), METRO paper arXiv 2512.09277; A = `DeepGEMM` branch `megamoe_mxfp4_dynamic_eplb` (`docs/superpowers/specs/2026-10-10-mxfp4-megamoe-dynamic-eplb-design.md`, `deep_gemm/mega/eplb_planner.py`, `deep_gemm/mega/eplb.py`, `deep_gemm/include/deep_gemm/impls/smxx_mega_moe_eplb.cuh`), MegaFlux arXiv 2610.00671 App. A.1.

## 1. Verdict

The two planners solve different problems and should not be ranked against each other on one axis. B (METRO) is a pure **routing** policy for memory-bound decode: with a fixed EPLB placement that already holds `R` redundant experts, it picks one replica per active logical expert so that the busiest rank reads the fewest expert weights; it moves no bytes, keeps no cross-step state, exchanges one 256-float all-reduce (or a 32-byte bitmap) per layer, and its single-warp Triton kernel costs 4-7 us per layer at EP8 (measured here on SM90, R=32..64, M<=64 per rank). A is a **placement plus routing** policy for compute-bound prefill: every step it recomputes which experts the helper slots hold, copies changed weights over NVLink (about 170 us per 20 MiB slot), and splits each expert's token *blocks* across its copies so that the busiest rank runs the fewest GEMM blocks; its one-CTA planner costs 6-13 us of compute at EP1 and 110-290 us at EP8 including the wait for the slowest peer. On real routing traces at EP8 the measurements line up with the designs: at decode sizes (8-16 tokens per rank) both reach the same maximum of activated experts per rank (within 0.5 expert of each other and within 1-2 of the ideal mean) but A needs 4-6 slot copies per step to get there, which is 0.7-1 ms of weight traffic per layer that a 0.3 ms decode layer cannot pay, while B pays nothing. At prefill sizes (256-1024 tokens per rank) A brings the busiest rank's block count to within 0.5 block of the mean (for example 89.9 -> 70.5 blocks on the skewed MiMo layer) whereas METRO, which routes all tokens of an expert to one rank, leaves the block count where static EPLB put it or makes it worse (72.6 -> 85.9 at R=32). Use B for decode and A for prefill; a combined design is sketched in section 4.

## 2. Side-by-side

| | A: MegaMoE dynamic EPLB (DeepGEMM) | B: METRO L2 router (moe_load_balancer) |
|---|---|---|
| Objective per step | Minimise the maximum number of **token blocks** `sum_e ceil(tokens_e / block_m)` on any rank: a GEMM-wave (compute) cost model | Minimise the maximum number of **activated logical experts** on any rank: a weight-read (HBM bytes) cost model. Paper: min-exp-routing ILP; Lemma 1 says one replica per expert is enough |
| Decision granularity | Per expert and per block: an expert's blocks are split across its copies (owner first, then replicas, by cumulative block ranges) | Per expert, all or nothing: every token of logical `l` goes to exactly one physical copy per step |
| What changes | **Placement** (which expert each helper slot holds) **and** routing (block ranges) | **Routing only**; the placement (EPLB with `R` redundant experts) is static until the L1 rebalance rebuilds it |
| Weights move at runtime | Yes: a changed helper slot fetches its MXFP4 planes (packed L1/L2, tile-major scales, global scales, 20.3 MiB at 6144/2048) from the owner over NVLink, about 170 us per slot, only when a slot's expert changes (slot cache) | No. Only the chosen replica changes |
| Memory for redundancy | `S` helper slots per rank, reassignable every step (tests: S=4, 32 extra slots at EP8) | `R` redundant physical experts fixed by EPLB (validated R=8..96 at EP8); same HBM cost per slot, but one slot can only ever serve the expert EPLB put there |
| Persistent state | Symmetric workspace: `slot_expert[R][S]`, `copies[E][R]` (rank, cumulative blocks), per-expert base offsets, copy list and count, generation flags; symmetric weight arena with `H+S` slots per rank | `MetroTables` per layer and placement (`physical_by_rank [L,G]`, `rank_mask`, `default_physical`, `single_rank`, `rep_logical/rep_mask/rep_phys`), an int32 workspace, cached by `placement_version`; p2p mode adds a symmetric `2*G*words` int64 buffer and a device epoch counter |
| Exchanged between ranks, when | Every step inside `plan`: the rank's full histogram `hist[E]` (int32, 1-1.5 KiB) is written into **every** peer's workspace, then a release flag; all ranks spin on all flags (30 s budget, then trap) | Every decode step, every MoE layer: default = one `float32[L]` EP all-reduce (1 KiB) between the count and route kernels; p2p = `L/32` int64 words (bitmap of active experts) pushed to every peer's slot, then spin on the epoch; OR of bitmaps replaces the count because only `count > 0` matters |
| Kernels per MoE layer | 3 launches: `plan` (1 CTA, warp 0 runs the serial greedy after the exchange), `copy_weights` (grid over changed slots, empty when nothing changed), `remap` (grid-stride over `tokens*topk`, rewrites `topk_idx` in the symmetric buffer to physical slot ids) | 2 launches + 1 collective (`metro_count_logical`, 1 program / 2 warps; custom all-reduce; `metro_route`, 1 program / 1 warp) or 1 launch (`metro_route_fused`). `metro_route` writes the physical ids in the requested dtype (int64 for DeepEP) in the same launch |
| Planner algorithm | Deterministic block-granular greedy (MegaFlux A.1, load only): `tau = ceil(sum_e b_e / R)`; up to `S+2` rounds of donor -> receiver block moves (overloaded owners by decreasing overload, home experts by decreasing blocks, receivers that already hold the expert first, fresh copies gated by `open_gate_blocks`); replica-drop refinement; single-block moves between the extremes; slot assignment that keeps cached slots | Paper Algorithm 1, serialised: active experts held by one rank add a fixed load there; active replicated experts are visited in ascending logical id and each goes to the eligible rank with the lowest load (ties to the lowest rank). Optional stale second pass places inactive experts with a fractional weight (0..8 eighths) |
| Serial work per call | O(rounds x donors x home experts) warp-serial steps, each an `R`-lane argmin, plus O(E*R) setup and two refinement passes; grows with experts per rank and with ranks | O(#active replicated experts) steps, each a 32-lane argmin over ranks (packed `(load << 5) | rank` key); the greedy is independent of the token count |
| Routing remap | Token `(t,k)` of expert `e` gets ordinal `base_rank(e) + local position`; block `ordinal / block_m`; the copy whose cumulative range contains the block; physical slot `rank*(H+S)+local_slot` | `out[t,k] = chosen_phys[topk_ids[t,k]]`; rows past `num_token_non_padded` and negative ids become -1 |
| Cross-rank consistency | Same histogram matrix -> same plan (ties broken by index); no collective, but every rank must issue the same number of plans | Same global count or same OR of bitmaps -> same assignment; under DP attention every rank must pass the same count; p2p degrades to partial bitmaps after `spin_limit` (balance only, never correctness) |
| Target backend | DeepGEMM SM90 MXFP4 fused MegaMoE kernel (one persistent kernel: dispatch, GEMMs, combine, on `R*(H+S)` physical slots); the kernel body is unchanged | SGLang with DeepEP (`--moe-a2a-backend deepep --deepep-mode auto`) and DeepGEMM masked grouped GEMM; runs only in pure-decode steps inside the CUDA graph; `--enable-eplb` is rejected on this branch |
| Regime | Prefill, M >= 256 per rank: compute bound, block count is the right cost | Decode, M = 1..64 per rank: memory bound, number of distinct expert weights read is the right cost |
| Own measured cost per layer | plan 6-13 us at EP1 (48 experts, pure compute); 110-290 us at EP8 including the wait for the slowest peer's launch; copy about 1 us steady state, 0.5-1.1 ms on a cold change; remap 2-8 us | route 3.6-6.7 us + count 1.3-1.8 us in graph (R=32..128, M=4..64 per rank) + 256-float all-reduce; 1.00 ms per 58-layer step at bs/rank 4 (assign 0.38, all-reduce 0.38, count 0.24); fused kernel 7.7 us in isolation but 2-2.5x slower in situ because its single warp shares SMs with the shared-expert GEMM |
| Own measured gain | Kernel 1.5-1.7x, end to end 1.04-1.52x at EP8 under heavy synthetic skew (half of all slots on three experts of one rank), M = 64..4096 | TPOT -3.5 / -5.5 / -1.3 / +2.7 % vs no redundancy at bs/rank 1 / 4 / 16 / 64 (DeepSeek-V3 FP8, EP8, R=64); -8 to -11 % vs static routing at the same R; DeepEP time 6.85 -> 4.07 ms per step |
| Limits | Planner serialised with the layer; the slowest rank's launch gates everyone; one greedy candidate; EP <= 32; cold copies 0.5-1 ms | EP <= 32; gain inverts at bs/rank >= 64 because the extra physical experts push DeepGEMM's masked GEMM to BLOCK_N=192 (3.5-5 % slower); stale mode balances poorly (65 % active-set overlap step to step); redundancy takes HBM from the KV cache |

### Differences that decide cost per step

1. **Synchronisation, not bytes.** Both exchange about 1 KiB per rank per layer. B hides the wait inside a tuned custom all-reduce (0.38 ms per 58 layers, about 6.5 us per layer); A spins inside the same kernel that then plans, so any launch skew between ranks becomes planner time (the 110-290 us at EP8 vs 6-13 us at EP1 in A's own measurements).
2. **Serial work.** B's greedy runs about one warp step per active replicated expert (measured: 1.8 us + about 0.08 us per replicated expert at EP8). A's greedy walks donors x home experts x rounds plus two refinement passes; the pure-Python references show the ratio: 0.6-5.5 ms (A) vs 0.07-0.18 ms (B) per call on the same inputs, 8-25x.
3. **Data movement.** B never moves weights. A moves 20 MiB per changed slot; on the real traces the steady state still changes 1.5-2.5 slots per step (summed over ranks) at prefill sizes and 3.5-5.6 at decode sizes, because the active set changes from batch to batch. A prefill layer of 1.5-2.6 ms absorbs one 0.17 ms copy; a decode layer does not.
4. **Token-count dependence.** A's remap is a grid-stride kernel and its planner is O(1) in tokens after the histogram. B's route loop is one warp walking `tokens*topk` entries 256 at a time: measured 2 us at M=1 and 12-20 us at M=1024 per rank (count kernel: 1.1 us -> 14-20 us). B's kernels are shaped for a few hundred entries per layer.

## 3. Planner benchmarks

All numbers below were measured in this study. B's Triton kernels ran on one SM90 GPU (CUDA-graph replay of 58 back-to-back calls, 30 replays, per-call time; the eager launch floor of 55 us per Triton call is excluded because the production path is graph-captured). Balance figures are means over 16 sampled batches per setting. Parity of `metro_route` against the CPU oracle `metro_assign_reference` held in all 126 GPU configurations.

### 3.1 B: `metro_route` + `metro_count_logical` latency vs ranks, redundant experts, tokens (random placement, random routing, graph replay, us per layer)

L = 256 logical experts, top-8:

| G | R (replicated) | M=1 route / count | M=8 | M=64 | M=256 | M=1024 |
|---:|---:|---|---|---|---|---|
| 8 | 0 (0) | 1.8 / 1.1 | 2.0 / 1.3 | 2.3 / 1.7 | 4.2 / 4.1 | 12.0 / 13.7 |
| 8 | 32 (28) | 4.0 / 1.3 | 4.0 / 1.3 | 4.4 / 1.8 | 6.4 / 4.2 | 14.2 / 13.7 |
| 8 | 64 (49) | 6.2 / 1.3 | 6.2 / 1.3 | 6.5 / 1.8 | 8.5 / 4.2 | 16.2 / 13.7 |
| 8 | 128 (87) | 6.8 / 1.3 | 9.2 / 1.3 | 9.5 / 1.8 | 11.4 / 4.2 | 19.1 / 13.7 |
| 16 | 32 (29) | 4.5 / 1.2 | 4.5 / 1.2 | 4.9 / 1.8 | 6.8 / 4.2 | 14.7 / 13.7 |
| 16 | 64 (51) | 7.0 / 1.2 | 7.0 / 1.2 | 7.3 / 1.8 | 9.3 / 4.2 | 17.1 / 13.7 |
| 16 | 128 (92) | 11.4 / 1.3 | 10.8 / 1.3 | 11.1 / 1.7 | 13.1 / 4.1 | 20.8 / 13.7 |
| 32 | 32 (29) | 3.4 / 1.2 | 3.4 / 1.2 | 3.8 / 1.7 | 5.8 / 4.2 | 13.6 / 13.7 |
| 32 | 64 (52) | 5.3 / 1.3 | 5.3 / 1.3 | 5.7 / 1.8 | 7.6 / 4.2 | 15.4 / 13.8 |
| 32 | 128 (96) | 7.2 / 1.2 | 7.7 / 1.2 | 8.0 / 1.7 | 10.0 / 4.2 | 17.7 / 13.8 |

L = 384 logical experts (MiMo shape), top-8:

| G | R (replicated) | M=1 | M=8 | M=64 | M=256 | M=1024 |
|---:|---:|---|---|---|---|---|
| 8 | 0 (0) | 2.1 / 1.5 | 2.1 / 1.5 | 2.5 / 2.1 | 4.4 / 5.6 | 12.1 / 19.6 |
| 8 | 32 (24) | 3.9 / 1.5 | 3.9 / 1.5 | 4.3 / 2.2 | 6.2 / 5.6 | 13.9 / 19.6 |
| 8 | 64 (51) | 6.3 / 1.5 | 6.3 / 1.5 | 6.7 / 2.2 | 8.7 / 5.6 | 16.6 / 19.6 |
| 8 | 128 (96) | 7.7 / 1.5 | 10.0 / 1.5 | 10.4 / 2.2 | 12.4 / 5.6 | 20.4 / 19.6 |
| 16 | 64 (54) | 7.2 / 1.5 | 7.2 / 1.5 | 7.6 / 2.1 | 9.6 / 5.6 | 17.6 / 19.5 |
| 32 | 64 (55) | 5.8 / 1.6 | 5.8 / 1.6 | 6.1 / 2.1 | 8.1 / 5.6 | 16.2 / 19.6 |
| 32 | 128 (102) | 9.4 / 1.5 | 9.4 / 1.5 | 9.7 / 2.1 | 11.7 / 5.6 | 19.4 / 19.6 |

Scaling with the number of experts (G=8, R=E/8, M=8, route / count us): E=64: 2.3 / 1.0; 128: 3.2 / 1.2; 256: 4.0 / 1.3; 384: 5.4 / 1.5; 512: 6.9 / 1.5; 1024: 13.1 / 2.4.

Reading: the route kernel is `1.8 us + ~0.08 us per active replicated expert + ~0.0012 us per (token, top-k) entry`; the rank count only changes the per-step argmin width (G=32 is not slower than G=8). The count kernel is `1.1 us + ~0.0015 us per entry`. Both are flat for decode (M <= 64) and grow linearly for prefill: at M=1024 per rank the pair costs 26-40 us per layer, more than A's whole planner at EP1. These are single-GPU numbers; the all-reduce (about 6.5 us per layer in B's own profile) and the wait for the slowest rank come on top in the EP8 system. B's own in-graph figures (3.6-6.7 us route at R=32..128, bs/rank 4..64) agree with this table.

### 3.2 A and B on the same inputs: real routing traces, EP8

Setup: three layers per trace (first, middle, last); for each layer the EPLB placement for B is computed from the first half of the trace (DeepSeek EPLB global policy, `rebalance_experts`), batches are sampled from the second half (M tokens per rank, 8 ranks, 16 batches per setting). A uses its own contiguous home layout (`owner = e // H`) with S=4 helper slots and `block_m` from DeepGEMM's heuristic for that M (shown). "A base" is A's layout with no helper slots; "EPLB-R0 static" is the EPLB placement without redundancy; "EPLB-R32 static" is the same memory budget as A (32 extra slots) under sglang-style static dispatch (local copy if the sender holds one, else round-robin); "METRO" is Algorithm 1 on the EP-summed counts. Every cell is the busiest rank's value, averaged over the batches and the three layers. "ideal" is the mean over ranks after METRO (the lower bound for the max). Block counts for B use the same `block_m` as A.

| trace | M/rank | block_m | A base: max active / blocks / tokens | A plan (S=4): max active / blocks / tokens | A slot copies per step (sum over ranks) | EPLB-R0 static: active / blocks | EPLB-R32 static: active / blocks | EPLB-R32 METRO: active / blocks / tokens | EPLB-R64 METRO: active / blocks | ideal mean active / blocks |
|---|---:|---:|---|---|---:|---|---|---|---|---|
| dsv4_flash_gsm8k | 8 | 16 | 18.4 / 19.0 / 76 | 15.2 / 15.2 / 72 | 4.6 | 18.0 / 18.1 | 21.1 / 21.1 | 15.8 / 16.6 / 83 | 15.6 / 16.6 | 14.5 / 14.8 |
| dsv4_flash_gsm8k | 16 | 16 | 22.0 / 24.3 / 149 | 20.0 / 20.1 / 131 | 3.6 | 21.7 / 22.8 | 24.9 / 25.0 | 19.5 / 23.2 / 167 | 19.8 / 23.3 | 18.2 / 19.6 |
| dsv4_flash_gsm8k | 32 | 16 | 24.9 / 33.5 / 298 | 24.2 / 27.2 / 236 | 2.8 | 24.8 / 29.8 | 28.1 / 31.6 | 22.7 / 34.7 / 335 | 23.0 / 34.4 | 21.5 / 26.6 |
| dsv4_flash_gsm8k | 64 | 32 | 27.4 / 36.3 / 589 | 27.0 / 30.0 / 461 | 2.3 | 27.3 / 32.4 | 30.6 / 33.7 | 25.4 / 37.4 / 660 | 25.8 / 36.5 | 24.5 / 29.5 |
| dsv4_flash_gsm8k | 256 | 96 | 30.3 / 44.8 / 2346 | 30.6 / 36.7 / 1744 | 2.0 | 30.2 / 38.4 | 33.8 / 41.1 | 29.1 / 46.0 / 2544 | 29.7 / 45.8 | 28.3 / 36.3 |
| dsv4_flash_gsm8k | 1024 | 192 | 31.7 / 70.2 / 9427 | 32.5 / 53.0 / 6713 | 1.5 | 31.1 / 56.0 | 35.1 / 57.8 | 30.7 / 71.5 / 9781 | 31.2 / 69.0 | 30.0 / 52.6 |
| dsv4_flash_sharegpt | 8 | 16 | 25.1 / 25.1 / 61 | 22.0 / 22.0 / 59 | 4.7 | 24.9 / 24.9 | 27.4 / 27.4 | 22.9 / 22.9 / 59 | 22.3 / 22.3 | 21.4 / 21.4 |
| dsv4_flash_sharegpt | 16 | 16 | 29.6 / 29.8 / 121 | 27.2 / 27.2 / 115 | 3.5 | 29.1 / 29.2 | 32.8 / 32.8 | 28.0 / 28.1 / 117 | 27.5 / 27.9 | 26.7 / 26.7 |
| dsv4_flash_sharegpt | 32 | 16 | 31.5 / 34.0 / 238 | 31.1 / 31.6 / 219 | 2.6 | 31.1 / 33.2 | 35.0 / 35.8 | 30.6 / 33.4 / 233 | 30.5 / 34.1 | 29.7 / 31.1 |
| dsv4_flash_sharegpt | 64 | 32 | 32.0 / 34.7 / 470 | 31.7 / 32.6 / 443 | 1.6 | 31.6 / 33.5 | 35.6 / 36.0 | 31.8 / 34.2 / 463 | 31.7 / 34.4 | 30.8 / 32.1 |
| dsv4_flash_sharegpt | 256 | 96 | 32.0 / 38.1 / 1887 | 32.7 / 35.3 / 1735 | 1.5 | 32.0 / 36.7 | 36.0 / 38.4 | 32.8 / 37.6 / 1818 | 32.7 / 39.4 | 31.6 / 34.8 |
| dsv4_flash_sharegpt | 1024 | 192 | 32.0 / 55.5 / 7517 | 33.4 / 48.7 / 6542 | 2.1 | 32.0 / 51.5 | 36.0 / 53.2 | 33.0 / 54.6 / 7320 | 33.0 / 56.9 | 31.9 / 48.3 |
| mimo25_gsm8k | 8 | 16 | 28.6 / 29.5 / 100 | 24.8 / 24.8 / 84 | 5.6 | 27.8 / 27.9 | 30.8 / 30.8 | 25.8 / 26.1 / 96 | 25.1 / 25.6 | 24.0 / 24.2 |
| mimo25_gsm8k | 16 | 16 | 34.3 / 37.2 / 194 | 31.5 / 31.7 / 163 | 4.1 | 33.7 / 34.6 | 37.1 / 37.2 | 32.1 / 34.4 / 191 | 31.3 / 33.9 | 30.1 / 31.2 |
| mimo25_gsm8k | 32 | 16 | 38.0 / 47.7 / 387 | 37.2 / 40.0 / 302 | 2.7 | 38.2 / 42.8 | 41.5 / 44.8 | 36.6 / 46.4 / 380 | 35.9 / 46.5 | 34.9 / 39.5 |
| mimo25_gsm8k | 64 | 32 | 40.6 / 50.3 / 779 | 40.2 / 43.1 / 612 | 2.7 | 40.6 / 45.3 | 44.0 / 47.3 | 39.2 / 50.2 / 766 | 39.1 / 48.9 | 38.1 / 42.8 |
| mimo25_gsm8k | 256 | 96 | 44.0 / 58.8 / 3099 | 44.6 / 50.4 / 2401 | 1.8 | 43.9 / 52.0 | 47.3 / 53.7 | 43.1 / 58.4 / 2876 | 43.0 / 58.6 | 42.1 / 50.0 |
| mimo25_gsm8k | 1024 | 192 | 46.0 / 89.9 / 12407 | 47.1 / 70.5 / 8934 | 1.5 | 46.1 / 72.6 | 49.8 / 75.0 | 45.7 / 85.9 / 11202 | 45.4 / 90.5 | 44.5 / 70.1 |

The skewed layers dominate the averages: MiMo layer 34 has a top expert at 35x the mean load and DeepSeek V4 Flash layer 0 at 34x; the ShareGPT layers are nearly flat (top expert 5-6x). Per-layer numbers are in the raw JSON produced by the benchmark.

What the table says:

* **Decode (M = 8-16 per rank), activated-expert metric.** A with 4 helper slots and METRO on EPLB+32 reach the same busiest-rank active count (A 15.2 / 22.0 / 24.8 vs METRO 15.8 / 22.9 / 25.8 at M=8; both within 1-2 experts of the ideal mean, and 3-4 experts below the no-redundancy baseline, i.e. 12-17 % fewer weight reads on the critical rank). At this block size (16 tokens) nearly every active expert is one block, so A's block objective degenerates into METRO's activated-expert objective. The difference is the price: A changes 3.5-5.6 helper slots per step (summed over ranks) because the active set changes from batch to batch; at 170 us per slot that is 0.6-1.0 ms of NVLink copies per layer, which is larger than the whole decode layer. METRO copies nothing.
* **Static redundancy without a dynamic router hurts decode**: EPLB-R32 static has 2-3 more active experts on the busiest rank than no redundancy at all (21.1 vs 18.0, 30.8 vs 27.8). This reproduces B's user-guide finding (static@64 is +4.8 to +15 % TPOT) and is the whole reason METRO exists.
* **Prefill (M = 256-1024 per rank), block metric.** A brings the busiest rank to within 0.3-1 block of the ideal mean (53.0 vs 52.6; 70.5 vs 70.1; 48.7 vs 48.3) from 70-90 blocks without helpers, i.e. 1.14-1.28x less work on the critical rank, while touching only 1.5-2.1 slots per step (summed over ranks). METRO on the same memory budget leaves the block count at or above static EPLB (71.5 vs 56.0 on DeepSeek V4 Flash GSM8K, 85.9 vs 72.6 on MiMo): because it sends all tokens of a hot expert to one rank, it undoes the token balance that EPLB's static dispatch had. More redundancy does not fix this (R=64: 69.0 / 90.5). A's activated-expert count goes up by about one at prefill (46.0 -> 47.1), which does not matter when the layer is compute bound.
* **Max tokens** follows blocks: at prefill A cuts the busiest rank's tokens by 28-30 % on the skewed traces (9427 -> 6713; 12407 -> 8934), METRO changes them by -4 to +4 %.

### 3.3 Synthetic skew (EP8, three hot experts on rank 0 take half of all slots)

| shape | M/rank | block_m | A base: active / blocks / tokens | A plan: active / blocks / tokens | EPLB-R32 static: active / blocks / tokens | EPLB-R32 METRO: active / blocks / tokens |
|---|---:|---:|---|---|---|---|
| 256 experts top-6 | 8 | 16 | 20.8 / 29.2 / 217 | 18.6 / 18.6 / 100 | 24.1 / 24.1 / 61 | 21.1 / 22.2 / 152 |
| 256 experts top-6 | 64 | 32 | 32.0 / 78.8 / 1720 | 36.0 / 44.2 / 1077 | 35.0 / 40.6 / 440 | 32.0 / 49.2 / 751 |
| 256 experts top-6 | 1024 | 192 | 32.0 / 160.1 / 27671 | 36.0 / 95.1 / 17012 | 35.0 / 54.1 / 7079 | 32.0 / 75.1 / 11331 |
| 384 experts top-8 | 8 | 16 | 29.1 / 38.8 / 285 | 26.2 / 26.2 / 139 | 32.2 / 32.5 / 78 | 29.2 / 31.3 / 212 |
| 384 experts top-8 | 64 | 32 | 48.0 / 111.1 / 2304 | 52.0 / 70.4 / 1479 | 51.0 / 58.7 / 581 | 48.0 / 74.8 / 1239 |
| 384 experts top-8 | 1024 | 192 | 48.0 / 218.3 / 36858 | 52.0 / 136.3 / 23083 | 51.0 / 77.6 / 9470 | 48.0 / 105.2 / 15067 |

Under this extreme skew the EPLB placement (computed from the same distribution, so it is an oracle here) does most of the work by giving the three hot experts up to 8 copies each; METRO then chooses one copy per expert and *re-concentrates* the tokens (max tokens 61 -> 152, 7079 -> 11331). A, with only 4 helper slots and one block granularity, splits a hot expert over up to 5 ranks and halves the busiest rank's blocks (160 -> 95, 218 -> 136); with S=4 it cannot reach the oracle placement's 54-78 blocks, which shows that A's slot budget, not its planner, is the limit under heavy skew. Uniform random routing (not shown) saturates every rank's local experts at M >= 32 and neither planner has anything to do.

### 3.4 Planner compute on identical inputs (CPU references, per call, median)

| | EP8, 256 experts | EP8, 384 experts | EP16 | EP32 |
|---|---:|---:|---:|---:|
| A `plan_reference` (pure Python) | 0.6-2.1 ms (5.4 ms under heavy skew) | 1.0-5.5 ms | 1.0-2.0 ms | 1.8-3.9 ms |
| B `metro_assign_reference` (pure Python, R=32) | 0.07-0.11 ms | 0.10-0.14 ms | 0.12-0.15 ms | 0.15-0.18 ms |

The device implementations are what matter, and the two cannot be run side by side here (A's planner is JIT-compiled into DeepGEMM and needs its symmetric workspace; B's is Triton). A's own measurements put its device planner at 6-13 us at EP1 and 110-290 us at EP8 (dominated by waiting for the slowest peer's launch); B's single-warp route kernel measured 4-7 us at EP8 for decode batches in this study. The 8-25x gap of the Python references is a fair proxy for the algorithmic work: A does rounds of donor/receiver search and two refinement passes over an `E x R` block matrix; B does one pass over the active replicated experts.

## 4. Recommendation

**Decode: B.** Its objective is the right one for a memory-bound layer (the busiest rank's weight bytes), its per-layer cost is 4-7 us of kernel plus one small all-reduce, it is CUDA-graph safe, and it needs no weight movement. On the real traces it removes 3-4 activated experts from the busiest rank at EP8 with 32 redundant slots (12-17 %), which is exactly what the DeepEP time reduction in B's end-to-end numbers reflects. A at decode sizes reaches the same balance but only by re-copying 20 MiB slots several times per step; even if the planner's 110-290 us were hidden, the copies alone exceed the layer time.

**Prefill: A.** The layer is compute bound; the cost is `max_rank sum ceil(tokens/block_m)`, and only a planner that can *split* an expert's tokens across copies can balance it. A brings the busiest rank to within one block of the mean on every trace layer with four helper slots and 1.5-2 slot copies per step across all ranks; METRO makes prefill balance worse than static EPLB because it concentrates each expert on one rank. A's remaining costs (planner serialised with the layer, cold copies) are a few percent of a 1.5-10 ms prefill layer and are already on the follow-up list (overlap with the previous layer, readiness gating in the kernel).

**Combination.** The two are complementary and share the same abstractions (a logical top-k table, a per-rank set of physical slots, a deterministic planner replicated on every rank), so one runtime can switch objective by batch size:

1. *One slot model.* Keep A's helper slots as the physical redundancy. At prefill, the planner reassigns them per step (A as is). For decode, freeze them: run A's planner once on a window of counts (or use EPLB's replication on the same window) to pick which experts the helper slots hold, copy once, and leave them for many steps. The helper slots then play the role of METRO's `R` redundant experts, but their contents can be re-chosen between windows without a server restart, which METRO's static placement cannot do.
2. *Two objectives, one kernel path.* Per layer: compute the local histogram once (A's plan kernel already does this; B's count kernel is the same thing). If `M * topk / E` is below the block size (every active expert is at most one block, the regime where A's block greedy degenerates into METRO's greedy anyway), run METRO's one-pass activated-expert greedy over the frozen slot table (`O(active replicated experts)`, a few us) and skip the copy stage entirely; otherwise run A's block-granular greedy with copies allowed. The remap stage is identical in both cases (logical id -> physical slot by cumulative block range; METRO is the special case of one copy getting all blocks). This brings METRO's objective to the fused MegaMoE kernel, where B's own result says the gain is largest (bs/rank 1-4), and keeps A's objective where it pays (M >= 256).
3. *Exchange.* Replace A's full-histogram push with B's bitmap exchange in the decode branch (one 64-bit word per 32 experts, OR instead of sum): METRO only needs `count > 0`. Keep the full histogram for the prefill branch, which needs the token counts to compute blocks. Both already use symmetric memory with a generation or epoch tag, so this is a mode switch rather than a new transport.
4. *Why not the reverse.* Giving METRO block splitting would turn it into A (it would have to move weights or accept that one replica per expert cannot balance tokens); giving A's planner to decode would keep paying for copies. The split by `M * topk / E` is the natural boundary and is already the boundary DeepGEMM uses to choose `block_m`.

What this does not solve: B's observation that extra physical experts slow DeepGEMM's masked GEMM at large batch (BLOCK_N choice) applies to the fused MegaMoE kernel as well, where A measured two empty helper slots at about 1 % at EP1; the combination should keep the helper-slot count small (4 per rank) and rely on reassignment rather than on a large static `R`.

## 5. Caveats

* **No 8-GPU measurement of B's fused exchange in this study.** `tests/dist_metro_fused.py` and `tests/dist_metro_contention.py` were submitted as an 8-GPU batch job on the shared cluster and had not been scheduled when this report was written; the per-layer numbers for `metro_route_fused` vs count + one-shot all-reduce + route, and the exposed latency under a concurrent GEMM, are therefore taken from B's user guide (7.7 vs 14.8 us in isolation; +0.19 / +0.57 / +4.9 us exposed for the default path at bs/rank 4 / 16 / 64) rather than reproduced. The job remains queued and its log can be appended later.
* **A's device planner was not re-timed here.** The 6-13 us (EP1) and 110-290 us (EP8) figures are from A's design document. The A-vs-B planner cost comparison on identical inputs is therefore the pure-Python reference ratio plus B's measured kernel, not two device kernels side by side.
* **B's kernels were timed on one GPU without the all-reduce.** The table in 3.1 isolates count and route; B's own profile adds about 6.5 us per layer for the custom all-reduce at EP8.
* **EPLB placement for B is an oracle.** It is computed from the first half of the same trace layer the batches are drawn from. A real deployment rebalances from a stale window, so B's balance in 3.2 is an upper bound; A does not depend on any placement history, so its numbers are not affected.
* **Trace sampling.** Decode batches were formed by sampling independent rows from the trace layer (8-64 per rank x 8 ranks), not by replaying sequences; the first/middle/last layer of each trace were used; the sharegpt trace layers are nearly flat, the gsm8k layers have 34-35x hot experts. "Blocks" for B's placements use A's `block_m` heuristic and count per physical slot, which is the cost model of the MegaMoE kernel, not of DeepEP + masked grouped GEMM.
* **A's slot copies per step are summed over ranks**; the critical-path cost is the per-rank maximum, which is smaller but not measured here. The spec's 170 us per slot is for the 6144/2048 MiMo shape; the DeepSeek V4 Flash shape (4096/2048) is smaller.
* **Not an end-to-end comparison.** Neither system was run with its MoE kernel in this study; the conclusions on regimes rest on the two projects' own end-to-end numbers plus the planner-level measurements above.
