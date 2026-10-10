# Dynamic expert load balancing for the SM90 MXFP4 MegaMoE kernel

Branch: `megamoe_mxfp4_dynamic_eplb`, cut from `megamoe_mxfp4_dev_m`.

## Problem

The fused MXFP4 MegaMoE kernel executes every expert on the rank that owns
its weights. Under routing skew the rank hosting the hottest experts decides
the layer latency while the others idle. Both references (TensorRT-LLM
DynamicEPLB, MegaFlux) solve this the same way: give every rank a few spare
*helper slots*, replicate hot experts into them at runtime, and send a share of
those experts' token blocks to the replicas, leaving router outputs unchanged.

## Contract

* **Physical slots.** A rank holds `H = E / R` home experts in local slots
  `[0, H)` and `S` helper slots in `[H, H + S)`. Physical slot id of
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
  2. `copy_weights`: copies changed helper slots' MXFP4 planes (packed L1/L2,
     tile-major scales, global scales) from the owner over NVLink. Weights
     live in a symmetric arena so peers can read them. Unchanged slots are
     not copied (slot cache).
  3. `remap`: rewrites the symmetric-buffer `topk_idx` to physical slot ids.
     Token `(t, k)` of expert `e` gets global ordinal
     `base_rank(e) + local_position`, block `ordinal / block_m`, and the
     copy whose cumulative block range contains it.
  4. `mxfp4_mega_moe` runs unchanged on physical slots.
* **Correctness invariant.** Any physical slot holding expert `e`'s weights
  produces the same result as the owner, so the plan only affects balance.
  Reference checks therefore compare against the logical-expert reference.

## Planner (device, one warp does the sequential part)

Inputs: `c[r][e]` token counts, previous `slot_expert`, block size `beta`
(= the kernel's `BLOCK_M` for this batch), replica open gate `lambda_open`
(blocks a new, uncached replica must receive to be worth a weight copy).

1. `b_e = ceil(sum_r c[r][e] / beta)`, `q[e][h(e)] = b_e`, loads
   `l_r = sum_e q[e][r]`, target `tau = ceil(sum_e b_e / R)`.
2. Up to `S + 2` rounds: visit overloaded owners by decreasing overload. For
   each home expert by decreasing remaining blocks, move
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

In-kernel readiness gating of helper slots (overlapping the weight copy with
home-expert tasks) and multi-candidate planner search. Both are additive.
