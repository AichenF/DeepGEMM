"""SM90 MXFP4 MegaMoE forward path for SGLang, with optional dynamic EPLB.

Drop-in companion to ``mega_moe_sm90.py``: FP4 experts (DeepSeek V4 style
packed E2M1 + E8M0 scales, or MiMo's MXFP4 store) go through DeepGEMM's fused
SM90 MXFP4 MegaMoE kernel. ``install_mxfp4_megamoe.py`` wires both the
DeepSeek V2/V4 and the MiMo V2 MoE modules to ``run_sm90_mxfp4_mega_routed``. ``SGLANG_MEGAMOE_DYNAMIC_EPLB_SLOTS=S`` (S > 0)
turns on dynamic expert load balancing with S helper slots per rank; 0 or
unset is fixed placement on the same kernel, so the two arms differ only in
the balancer. ``SGLANG_MEGAMOE_DYNAMIC_EPLB_MIN_TOKENS`` (default 512) is the
smallest local token count that triggers a new plan.
``SGLANG_MEGAMOE_DYNAMIC_EPLB_EARLY_PLAN=1`` (default off) launches that plan
on a side stream right after top-k and runs the shared experts while it
computes, so the plan is still made from the current chunk's routing but no
longer idles the GPU before the MegaMoE kernel (prefill only; see
``_EPLB_EARLY_PLAN`` below).

Owner permutation (PREFILL_headroom_study.md section 10, zero per-call
cost): start the server with ``--init-expert-location <file>`` built by
``tests/eplb_tools/build_owner_permutation.py`` (a per-layer
``physical_to_logical_map`` with no redundancy). SGLang's weight loader then
puts logical expert ``perm[p]`` into physical slot ``p`` on rank ``p // H``,
and with ``--ep-dispatch-algorithm static`` (its default once a location is
given) top-k returns physical ids, so the kernel, the planner and the remap
see the permuted layout without any change here. The adapter reads the loaded
location once per layer (``_owner_permutation``) to log it and to map files
that name logical experts (the static placement file) before
``set_replicas``; ``SGLANG_MEGAMOE_OWNER_PERM=<companion json>`` makes it
check that the loaded location equals the file the script built and fail
otherwise. Nothing of this runs per forward, so CUDA graphs are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

_EPLB_SLOTS = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_SLOTS", "0"))
_EPLB_OPEN_GATE = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_OPEN_GATE_BLOCKS", "0"))
# Below this many local tokens (decode steps, small prefill chunks) the planner
# is skipped: tokens are remapped onto the placement from the last plan, so the
# per-layer planner cost is not paid where imbalance cannot be exploited. All
# ranks must take the same branch; without DP attention every rank sees the
# same token count.
_EPLB_MIN_TOKENS = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_MIN_TOKENS", "512"))
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_REPLAN_EVERY=K`` (default 1): a layer plans on
# every K-th large forward and remaps the others onto its last plan. Routing
# statistics change little between consecutive prefill chunks of the same
# request mix, so a stale plan still balances while the plan and the weight
# copies are paid once per K chunks. The count is per layer and advances only
# on forwards all ranks agree are large, so all ranks take the same branch.
_EPLB_REPLAN_EVERY = max(1, int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_REPLAN_EVERY", "1")))
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_EARLY_PLAN=1``: take the planner off the
# critical path without making the plan stale. ``forward_mega_moe`` (patched
# by ``install_mxfp4_megamoe.py``) hands the forward to
# ``forward_sm90_mxfp4_mega_moe``, which runs the gate and top-k first,
# launches this chunk's plan on a side stream right after top-k (the plan
# kernel is histogram, exchange with the peers, greedy and tables in one
# launch, so the whole kernel moves), then runs the shared experts on the
# main stream; the main stream waits for the plan event just before the
# remap and the MegaMoE launch. In the stock order the shared experts run
# before the gate, so nothing of size sits between top-k and the kernel;
# the reorder creates the window. The plan is computed from the current
# chunk's routing, exactly as in the synchronous mode, but its 76-110 us
# (mostly the in-kernel wait for the slowest rank's launch) overlap the
# shared-expert GEMMs. Prefill only: CUDA-graph capture and replays keep the
# plain path. The plan kernel keeps its 30 s peer-wait trap; a late rank only delays its
# peers' side streams, never a main stream a peer waits on, so no cycle.
_EPLB_EARLY_PLAN = os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_EARLY_PLAN", "0") == "1"
_EPLB_EARLY_STREAM = None
# SGLang / DeepGEMM symbols bound on the first forward (importing them at
# module import time would create cycles; importing per call costs a few
# microseconds each on the per-layer host path).
_per_token_cast_to_fp8 = None
_get_moe_ep_group = None
_get_dp_global_num_tokens = None
_get_is_capture_mode = None
_ExpertLocationDispatchInfo = None
_get_global_expert_location_metadata = None
_envs = None
# ``SGLANG_MEGAMOE_OWNER_PERM=<json>``: the companion file written by
# ``build_owner_permutation.py`` next to the ``--init-expert-location`` file;
# the loaded location of every layer must equal its row (see the docstring).
_OWNER_PERM_FILE = os.environ.get("SGLANG_MEGAMOE_OWNER_PERM", "")
_OWNER_PERM_DATA = None


def _bind_lazy() -> None:
    global _per_token_cast_to_fp8, _get_moe_ep_group, _get_dp_global_num_tokens
    global _get_is_capture_mode, _ExpertLocationDispatchInfo, _get_global_expert_location_metadata, _envs
    from deep_gemm.utils import per_token_cast_to_fp8
    from sglang.srt.distributed.parallel_state import get_moe_ep_group
    from sglang.srt.environ import envs
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
    from sglang.srt.model_executor.runner import get_is_capture_mode
    _per_token_cast_to_fp8 = per_token_cast_to_fp8
    _get_moe_ep_group = get_moe_ep_group
    _get_dp_global_num_tokens = get_dp_global_num_tokens
    _get_is_capture_mode = get_is_capture_mode
    _ExpertLocationDispatchInfo = ExpertLocationDispatchInfo
    _get_global_expert_location_metadata = get_global_expert_location_metadata
    _envs = envs


def _owner_permutation(moe):
    """The owner permutation of this layer as SGLang loaded it: ``perm[p]`` is
    the logical expert in physical slot ``p`` (``None`` for the identity).
    Read once per layer from the global expert-location metadata, so it is
    exactly what the weight loader applied and what the static top-k dispatch
    emits; with ``SGLANG_MEGAMOE_OWNER_PERM`` the row must equal the companion
    file of ``build_owner_permutation.py`` (identity for layers it omits)."""
    global _OWNER_PERM_DATA
    cached = getattr(moe.experts, "_mega_moe_owner_perm", False)
    if cached is not False:
        return cached
    if _get_global_expert_location_metadata is None:
        _bind_lazy()
    layer_id = getattr(moe, "layer_id", None)
    num_experts = int(moe.experts.num_experts)
    identity = list(range(num_experts))
    meta = _get_global_expert_location_metadata()
    perm = None
    if meta is not None and layer_id is not None and int(layer_id) < meta.num_layers:
        layer_id = int(layer_id)
        row = [int(x) for x in meta.physical_to_logical_map_cpu[layer_id].tolist()]
        assert len(row) == num_experts, (
            f"layer {layer_id}: {len(row)} physical experts for {num_experts} logical ones; the MXFP4 "
            "MegaMoE path takes a pure owner permutation (its helper slots are not SGLang redundant experts)")
        if row != identity:
            assert sorted(row) == identity, f"layer {layer_id}: expert location is not a permutation"
            perm = row
    if _OWNER_PERM_FILE and layer_id is not None:
        if _OWNER_PERM_DATA is None:
            with open(_OWNER_PERM_FILE) as f:
                _OWNER_PERM_DATA = json.load(f)
        expected = [int(x) for x in _OWNER_PERM_DATA["layers"].get(str(layer_id), identity)]
        if (perm if perm is not None else identity) != expected:
            raise RuntimeError(
                f"layer {layer_id}: the expert location SGLang loaded differs from SGLANG_MEGAMOE_OWNER_PERM="
                f"{_OWNER_PERM_FILE}; pass the matching file to --init-expert-location")
    moe.experts._mega_moe_owner_perm = perm
    return perm


def _log_owner_placement(moe, group, perm) -> None:
    """One line per layer on rank 0 at the first forward: how many experts
    the permutation gives to another rank than the contiguous placement."""
    if perm is None or group.rank() != 0:
        return
    num_experts = len(perm)
    home = num_experts // group.size()
    moved = sum(1 for p, e in enumerate(perm) if p // home != e // home)
    logging.getLogger(__name__).info(
        "megamoe-eplb layer %s: owner permutation active (SGLang expert location), %d of %d experts "
        "owned by another rank than in the contiguous placement; rank 0 holds logical experts %s",
        moe.layer_id, moved, num_experts, perm[:home])


def _layer_consts(moe) -> tuple:
    """Per-layer constants of the forward, read from the config once instead
    of on every call: (hidden, intermediate, num_experts, top_k,
    routed_scaling, activation clamp)."""
    c = getattr(moe.experts, "_mxfp4_mega_consts", None)
    if c is None:
        routed_scaling = 1.0 if moe.experts.should_fuse_routed_scaling_factor_in_topk \
            else float(getattr(moe, "routed_scaling_factor", None) or 1.0)
        c = moe.experts._mxfp4_mega_consts = (
            int(moe.config.hidden_size), int(moe.config.moe_intermediate_size),
            int(moe.experts.num_experts), int(moe.config.num_experts_per_tok),
            routed_scaling, getattr(moe.config, "swiglu_limit", None))
    return c
# Static helper-slot placement. ``SGLANG_MEGAMOE_STATIC_PLACEMENT=<file>`` is
# a ``{layer_id: counts[E]}`` load histogram (token counts per logical
# expert): at a layer's first forward its helper slots are filled once by
# EPLB-style replication by load (``static_replica_placement``); layers
# absent from the file keep empty helper slots. With
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_STATIC=1`` the prefill planner never runs, so
# the placement stays what the file gave: a fixed redundant placement on the
# same kernel, which is how an external static balancer is compared.
_STATIC_PLACEMENT = os.environ.get("SGLANG_MEGAMOE_STATIC_PLACEMENT", "")
_EPLB_STATIC = os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_STATIC", "0") == "1"
_STATIC_HIST = None
_MXFP4_SYMM_BUFFER: dict = {}
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE=N`` (N > 0): rank 0 times every N-th
# layer forward with at least ``_TRACE_MIN_TOKENS`` local tokens (CUDA events
# around the whole MoE call, read back one trace period later so nothing
# stalls) and logs it, with the planned/remap-only counters and the rank
# loads of the plan that forward used. Works with and without the balancer,
# so the two arms can be compared layer by layer inside the server.
_EPLB_TRACE = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE", "0"))
_EPLB_STATS = {"plan": 0, "plan_tokens": 0, "remap": 0, "remap_tokens": 0, "large": 0}
_TRACE_MIN_TOKENS = 512
_TRACE_PENDING = None


def _plan_summary(copies: torch.Tensor, slot_expert: torch.Tensor, home_experts: int) -> str:
    """Rank loads in token blocks for one plan, read from the planner's copy
    table: owner-only placement versus the planned placement."""
    c = copies.tolist()
    num_ranks = len(c[0])
    fixed = [0] * num_ranks
    planned = [0] * num_ranks
    for e, row in enumerate(c):
        prev = 0
        for rank, cum in row:
            if rank < 0:
                break
            if cum >= 0x7FFFFFFF:
                return "no plan yet"
            planned[rank] += cum - prev
            prev = cum
        fixed[e // home_experts] += prev
    replicas = int((slot_expert >= 0).sum().item())
    return (f"replicas {replicas}, max rank load {max(fixed)} -> {max(planned)} blocks "
            f"(mean {sum(planned) / num_ranks:.0f})")


def _trace_flush(eplb) -> None:
    global _TRACE_PENDING
    if _TRACE_PENDING is None:
        return
    (ordinal, tokens, mode, start, mid, end, copies, slots, copy_count,
     host_us, early_events) = _TRACE_PENDING
    _TRACE_PENDING = None
    end.synchronize()
    msg = (f"[megamoe-trace] large forward #{ordinal}: {tokens} local tokens, {mode}, "
           f"{start.elapsed_time(end) * 1000:.0f} us")
    if eplb is not None:
        st = _EPLB_STATS
        msg += (f" (prepare {start.elapsed_time(mid) * 1000:.0f}"
                f" + kernel {mid.elapsed_time(end) * 1000:.0f})"
                f" | host prepare {host_us[0]:.0f} us, kernel launch {host_us[1]:.0f} us")
        if early_events is not None:
            # Early plan: GPU spans of the reordered forward. ``top-k->shared``
            # is the shared experts (the overlap window), ``shared->copies`` the
            # FP8 cast and the buffer copies, ``side plan`` the plan kernel on
            # the side stream including its peer wait, issued that long after
            # top-k; the main stream's remaining wait is inside ``prepare``.
            ev_topk, ev_shared, s0, s1 = early_events
            msg += (f" | early: top-k->shared {ev_topk.elapsed_time(ev_shared) * 1000:.0f} us, "
                    f"shared->copies {ev_shared.elapsed_time(start) * 1000:.0f} us, ")
            if s0 is not None:
                msg += (f"side plan {s0.elapsed_time(s1) * 1000:.0f} us issued "
                        f"{ev_topk.elapsed_time(s0) * 1000:.0f} us after top-k")
            else:
                msg += "no plan this forward"
        msg += (f" | planned {st['plan']} (mean {st['plan_tokens'] / max(st['plan'], 1):.0f} tokens), "
                f"remap-only {st['remap']} (mean {st['remap_tokens'] / max(st['remap'], 1):.0f} tokens)"
                f" | its plan: {_plan_summary(copies, slots, eplb.home_experts)}, "
                f"{int(copy_count.item())} weight copies on rank 0")
    else:
        msg += f" | host kernel launch {host_us[1]:.0f} us"
    logging.getLogger(__name__).info(msg)


def _trace_decision(num_tokens: int, group) -> bool:
    """Whether rank 0 times this forward (every ``_EPLB_TRACE``-th large one).
    Counts the large forwards, so call it once per forward."""
    if _EPLB_TRACE and num_tokens >= _TRACE_MIN_TOKENS and group.rank() == 0:
        _EPLB_STATS["large"] += 1
        return _EPLB_STATS["large"] % _EPLB_TRACE == 0
    return False


def is_sm90_mxfp4_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "mxfp4_mega_moe") and getattr(experts, "_mega_moe_sm90_mxfp4_weights", False)


def _ocp_to_marlin_packed(packed: torch.Tensor) -> torch.Tensor:
    """OCP MXFP4 bytes (element 2j in the low nibble of byte j) to the
    chunk-of-8 permutation DeepGEMM's MXFP4 quantizer emits (byte b of a
    chunk holds elements b+4 low and b high)."""
    assert packed.dtype == torch.uint8
    *outer, half_k = packed.shape
    k = half_k * 2
    nibbles = torch.empty((*outer, k), dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = packed & 0x0F
    nibbles[..., 1::2] = (packed >> 4) & 0x0F
    chunks = nibbles.view(*outer, k // 8, 8)
    return (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).reshape(*outer, half_k).contiguous()


def build_sm90_mxfp4_mega_moe_experts_weights(experts) -> None:
    """Transform the layer's packed FP4 experts into the fused kernel layout.

    Expects ``w13_weight`` / ``w2_weight`` as packed uint8 ``[E, N, K/2]`` and
    ``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` as E8M0 ``[E, N, K/32]``
    (uint8, int8 view or float8_e8m0fnu)."""
    if getattr(experts, "_mega_moe_weights_built", False):
        return
    import deep_gemm

    def as_u8(t):
        return t.data.view(torch.uint8) if t.data.dtype != torch.uint8 else t.data

    def as_e8m0_u8(t):
        """Group scales as E8M0 bytes. The loader may keep them as floating
        point values (power-of-two UE8M0 scales widened to fp32)."""
        t = t.data
        if t.dtype in (torch.uint8, torch.int8) or str(t.dtype).endswith("e8m0fnu"):
            return t.view(torch.uint8).contiguous()
        f = t.float()
        e = torch.log2(f.clamp_min(2.0 ** -127)).round()
        if not torch.equal(torch.ldexp(torch.ones_like(e), e.to(torch.int32)), f):
            import logging
            logging.getLogger(__name__).warning("MXFP4 group scales are not exact powers of two; rounding to E8M0")
        return (e + 127).clamp(0, 255).to(torch.uint8).contiguous()

    w13 = _ocp_to_marlin_packed(as_u8(experts.w13_weight))
    w2 = _ocp_to_marlin_packed(as_u8(experts.w2_weight))
    # DeepSeek V4 checkpoints load through the FP8 method (``*_scale_inv``),
    # MiMo through the MXFP4 method (``*_scale``); the tensors are the same.
    def scale_param(name):
        p = getattr(experts, name + "_scale_inv", None)
        return p if p is not None else getattr(experts, name + "_scale")

    s13 = as_e8m0_u8(scale_param("w13_weight"))
    s2 = as_e8m0_u8(scale_param("w2_weight"))
    assert w13.shape[2] * 2 == s13.shape[2] * 32 and w2.shape[2] * 2 == s2.shape[2] * 32, \
        f"scale shapes do not match a 32-element MXFP4 group: {tuple(w13.shape)} {tuple(s13.shape)}"
    # The kernel folds each group scale into an E4M3 exponent, which is exact
    # only for E8M0 codes 122..133. Shift every expert's codes into that window
    # and carry the shift as the kernel's per-expert global scale.
    from deep_gemm.quantization_mxfp4 import fold_e8m0_into_kernel_window
    s13, gs1, lost1 = fold_e8m0_into_kernel_window(s13)
    s2, gs2, lost2 = fold_e8m0_into_kernel_window(s2)
    if max(lost1, lost2) > 0:
        import logging
        logging.getLogger(__name__).warning(
            "MXFP4 groups below the kernel's E8M0 window flush to zero: %.4f (l1) %.4f (l2)", lost1, lost2)
    l1, l2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90((w13, s13), (w2, s2))
    experts.mega_num_local_experts = int(w13.shape[0])
    hidden = int(w13.shape[2] * 2)          # packed [E, 2I, K/2]
    intermediate = int(w13.shape[1] // 2)
    # The packed checkpoint tensors are no longer needed in their original layout.
    experts.w13_weight.data = torch.empty(0, dtype=torch.uint8, device=w13.device)
    experts.w2_weight.data = torch.empty(0, dtype=torch.uint8, device=w2.device)
    del w13, w2, s13, s2
    experts._mega_moe_sm90_mxfp4_weights = True
    experts._mega_moe_weights_built = True
    experts._mega_moe_eplb = None
    if _EPLB_SLOTS > 0:
        # Allocate the balancer at load time so the KV-cache sizing sees it and
        # nothing is allocated while CUDA graphs are being captured. The arena
        # takes over the home experts; the transformed originals are released.
        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        group = get_moe_ep_group().device_group
        num_topk = int(getattr(experts, "top_k")) + int(getattr(experts, "num_fused_shared_experts", 0) or 0)
        cap = int(os.environ.get("SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK", "4096"))
        eplb = _new_eplb(group, int(experts.num_experts), cap, num_topk, hidden, intermediate)
        eplb.set_home_weights(l1, l2, gs1, gs2)
        del l1, l2
        torch.cuda.empty_cache()
        experts._mega_moe_eplb = eplb
        return
    experts.mega_l1_weights = l1
    experts.mega_l2_weights = l2
    experts.mega_l1_global_scales = gs1
    experts.mega_l2_global_scales = gs2


def _get_symm_buffer(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden):
    import deep_gemm
    key = (id(group), num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
    buf = _MXFP4_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
            use_fp8_dispatch=True, activation="swiglu")
        _MXFP4_SYMM_BUFFER[key] = buf
    return buf


def _new_eplb(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden):
    import deep_gemm
    # One symmetric buffer sized for the physical slots, shared by all layers.
    num_slots = num_experts + _EPLB_SLOTS * group.size()
    symm = _get_symm_buffer(group, num_slots, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
    return deep_gemm.DynamicEPLB(group, num_experts, _EPLB_SLOTS, num_max_tokens_per_rank, num_topk,
                                 hidden, intermediate_hidden, open_gate_blocks=_EPLB_OPEN_GATE, symm=symm)


def _install_static_placement(moe, eplb) -> None:
    """Fill the helper slots once from this layer's load histogram
    (``SGLANG_MEGAMOE_STATIC_PLACEMENT``); layers absent from the file keep
    empty helper slots."""
    global _STATIC_HIST
    from deep_gemm.mega.eplb_planner import apply_owner_permutation, static_replica_placement
    if _STATIC_HIST is None:
        _STATIC_HIST = torch.load(_STATIC_PLACEMENT, map_location="cpu")
    layer_id = getattr(moe, "layer_id", None)
    hist = _STATIC_HIST.get(int(layer_id)) if layer_id is not None else None
    if hist is None:
        table = [[-1] * eplb.num_helper_slots for _ in range(eplb.num_ranks)]
        how = "no history"
    else:
        # The file names logical experts; the balancer's slots and
        # ``set_replicas`` use the kernel's physical numbering, so under an
        # owner permutation the counts are mapped to physical order first.
        counts = hist["counts"] if isinstance(hist, dict) else hist
        counts = torch.as_tensor(counts).float().tolist()
        perm = _owner_permutation(moe)
        if perm is not None:
            counts = apply_owner_permutation(counts, perm)
        table = static_replica_placement(counts, eplb.num_ranks, eplb.num_helper_slots)
        how = "static placement from load counts"
        if perm is not None:
            how += " (physical ids through the owner permutation)"
    copied = eplb.set_replicas(table)
    if eplb.rank == 0:
        replicated = sum(1 for r in table for e in r if e >= 0)
        logging.getLogger(__name__).info(
            "megamoe-eplb layer %s: %s, %d replicas in %d helper slots (rank 0 copied %d)",
            layer_id, how, replicated, eplb.num_ranks * eplb.num_helper_slots, copied)


def _get_eplb(moe, group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden):
    eplb = getattr(moe.experts, "_mega_moe_eplb", None)
    if eplb is None:
        eplb = _new_eplb(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
        eplb.set_home_weights(moe.experts.mega_l1_weights, moe.experts.mega_l2_weights)
        moe.experts.mega_l1_weights = None
        moe.experts.mega_l2_weights = None
        torch.cuda.empty_cache()
        moe.experts._mega_moe_eplb = eplb
    if not getattr(eplb, "_layer_ready", False):
        # First forward of this layer. SGLang warms every graph batch size up
        # eagerly before capturing it, so this runs outside any capture; the
        # layer id is only known here, not at weight-load time.
        if _STATIC_PLACEMENT:
            _install_static_placement(moe, eplb)
        # Realign the ranks' CPUs before this layer's first MegaMoE launch. The
        # kernel waits for its peers in an NVLink barrier with a 30 s budget and
        # traps beyond it (reported as an unspecified launch failure); SGLang's
        # DeepGEMM pre-compile walk runs on the node's first rank only and, at
        # 32k chunks, keeps that rank away for about 80 s right here. The fixed
        # path gets the same alignment for free from its lazy symmetric-buffer
        # rendezvous; with helper slots everything was allocated at load time.
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        eplb._layer_ready = True
    assert eplb.num_topk == num_topk and eplb.num_max_tokens_per_rank >= num_max_tokens_per_rank, \
        f"EPLB sized for topk={eplb.num_topk}, cap={eplb.num_max_tokens_per_rank}; forward needs {num_topk}, {num_max_tokens_per_rank}"
    return eplb


def _decision_tokens(num_tokens: int, group) -> int:
    """The token count every rank agrees on. The planner is a collective:
    every rank must take the same branch and use the same block size, so
    both derive from this. SGLang publishes the batch's token counts: one
    entry per DP rank with DP attention, otherwise one global entry while
    the MoE input is sequence-sharded across the EP ranks."""
    global_num_tokens = _get_dp_global_num_tokens()
    if not global_num_tokens:
        return num_tokens
    if len(global_num_tokens) > 1:
        return max(global_num_tokens)
    return (global_num_tokens[0] + group.size() - 1) // group.size()


def _decide(eplb, decision_tokens: int, capturing: bool) -> tuple:
    """Whether this forward plans. Advances the layer's large-forward
    counter, so it is called exactly once per forward."""
    replan = False
    if decision_tokens >= _EPLB_MIN_TOKENS and not capturing and not _EPLB_STATIC:
        large = getattr(eplb, "_large_forwards", 0)
        eplb._large_forwards = large + 1
        if eplb.generation == 0:
            replan = True
        else:
            replan = large % _EPLB_REPLAN_EVERY == 0
    return replan


class _EarlyPlan:
    """What ``forward_sm90_mxfp4_mega_moe`` decided and launched before the
    shared experts, handed to ``run_sm90_mxfp4_mega_routed``."""
    __slots__ = ("eplb", "decision", "decision_tokens", "block_m", "plan_done",
                 "topk_written", "trace_this", "ev_topk", "ev_shared", "side_events")

    def __init__(self, eplb, decision, decision_tokens, trace_this):
        self.eplb = eplb
        self.decision = decision
        self.decision_tokens = decision_tokens
        self.block_m = None
        self.plan_done = None
        self.topk_written = False
        self.trace_this = trace_this
        self.ev_topk = None
        self.ev_shared = None
        self.side_events = (None, None)


def _begin_early_plan(moe, group, topk_ids, num_tokens: int, cap: int) -> _EarlyPlan:
    """Right after top-k: decide this forward's balancer action and, when it
    is a fresh plan, launch the plan kernel on the side stream so that the
    shared experts overlap it. The side stream writes the logical routing
    into the shared buffer itself (ordered after top-k by an event, and
    after the previous layer's kernel by stream order), plans from it, and
    records ``plan_done``; the main stream waits for that event only just
    before the remap."""
    global _EPLB_EARLY_STREAM
    hidden, intermediate, num_experts, top_k_default, _, _ = _layer_consts(moe)
    top_k = topk_ids.shape[1] if num_tokens > 0 else top_k_default
    eplb = _get_eplb(moe, group, num_experts, cap, top_k, hidden, intermediate)
    capturing = torch.cuda.is_current_stream_capturing()
    decision_tokens = _decision_tokens(num_tokens, group)
    early = _EarlyPlan(eplb, _decide(eplb, decision_tokens, capturing), decision_tokens,
                       _trace_decision(num_tokens, group))
    if early.trace_this:
        early.ev_topk = torch.cuda.Event(enable_timing=True)
        early.ev_topk.record()
    if not early.decision[0] or capturing:
        return early
    if _EPLB_EARLY_STREAM is None:
        _EPLB_EARLY_STREAM = torch.cuda.Stream()
    side = _EPLB_EARLY_STREAM
    block_m = eplb.block_m(decision_tokens)
    topk_done = torch.cuda.Event()
    topk_done.record()
    with torch.cuda.stream(side):
        side.wait_event(topk_done)
        if num_tokens > 0:
            eplb.topk_view(num_tokens).copy_(topk_ids)
        if early.trace_this:
            s0 = torch.cuda.Event(enable_timing=True)
            s0.record()
        eplb.plan(num_tokens, block_m)
        if early.trace_this:
            s1 = torch.cuda.Event(enable_timing=True)
            s1.record()
            early.side_events = (s0, s1)
        done = torch.cuda.Event()
        done.record()
    early.block_m = block_m
    early.plan_done = done
    early.topk_written = True
    return early


def wants_early_plan(moe) -> bool:
    """Asked by the patched ``forward_mega_moe`` before its stock body: True
    hands the forward to ``forward_sm90_mxfp4_mega_moe``. False whenever the
    knob is off, no balancer is configured, the layer is not on the MXFP4
    path, or a CUDA graph is being captured (the decode path stays as it is).
    With the knob off this is one module-flag test."""
    if not _EPLB_EARLY_PLAN or _EPLB_SLOTS <= 0:
        return False
    if not getattr(moe.experts, "_mega_moe_sm90_mxfp4_weights", False):
        return False
    if _get_is_capture_mode is None:
        _bind_lazy()
    return not _get_is_capture_mode()


def forward_sm90_mxfp4_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch=None,
    input_ids_global: torch.Tensor = None,
) -> torch.Tensor:
    """``forward_mega_moe`` with the early plan: gate and top-k first, the
    plan launched on the side stream, then the shared experts on the main
    stream, then the routed experts (which wait for the plan just before the
    remap). Same gate, top-k and shared-expert calls as the stock body, in a
    different order; the result is the same sum."""
    if _per_token_cast_to_fp8 is None:
        _bind_lazy()
    num_tokens = hidden_states.shape[0]
    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(forward_batch.num_token_non_padded if forward_batch is not None else None),
            expert_location_dispatch_info=_ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None
    cap = _envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    assert num_tokens <= cap, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK={cap}")
    group = _get_moe_ep_group().device_group
    early = _begin_early_plan(moe, group, topk_ids, num_tokens, cap)
    shared_output = moe._forward_shared_experts(hidden_states)
    if early.trace_this:
        early.ev_shared = torch.cuda.Event(enable_timing=True)
        early.ev_shared.record()
    y = run_sm90_mxfp4_mega_routed(moe, hidden_states, topk_ids, topk_weights, num_tokens, cap, early=early)
    if shared_output is not None:
        y.add_(shared_output)
    return y


def run_sm90_mxfp4_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    early: _EarlyPlan = None,
) -> torch.Tensor:
    import deep_gemm
    if _per_token_cast_to_fp8 is None:
        _bind_lazy()

    group = _get_moe_ep_group().device_group
    hidden, intermediate, num_experts, top_k_default, routed_scaling, clamp = _layer_consts(moe)
    top_k = topk_ids.shape[1] if num_tokens > 0 else top_k_default
    if not getattr(moe.experts, "_mega_moe_placement_logged", False):
        # First forward of this layer (SGLang's eager warm-up, never under a
        # capture): resolve the owner permutation SGLang loaded, check it
        # against SGLANG_MEGAMOE_OWNER_PERM if given, and log it once. Host
        # work only; afterwards this is one attribute test per forward.
        _log_owner_placement(moe, group, _owner_permutation(moe))
        moe.experts._mega_moe_placement_logged = True

    eplb = None
    if _EPLB_SLOTS > 0:
        eplb = early.eplb if early is not None else \
            _get_eplb(moe, group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)
        buf = eplb.symm
    else:
        buf = _get_symm_buffer(group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)

    trace_this = early.trace_this if early is not None else _trace_decision(num_tokens, group)
    mode = "kernel, fixed placement"

    if num_tokens > 0:
        x_fp8, x_sf = _per_token_cast_to_fp8(hidden_states.to(torch.bfloat16), use_ue8m0=False, gran_k=128)
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_sf)
        # ``copy_`` converts the routing ids (int32) and weights to the
        # buffer's dtypes itself, so no intermediate ``.to`` tensors are made.
        if early is None or not early.topk_written:
            buf.topk_idx[:num_tokens].copy_(topk_ids)
        if routed_scaling == 1.0:
            buf.topk_weights[:num_tokens].copy_(topk_weights)
        else:
            buf.topk_weights[:num_tokens].copy_(topk_weights * routed_scaling)
    # The trace starts after the FP8 cast and the buffer copies, which both
    # arms pay alike, so "prepare" is the balancer's own work (plan + copy,
    # or remap) and "kernel" is the MegaMoE call.
    if trace_this:
        _trace_flush(eplb)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        t_host0 = time.perf_counter()

    y = torch.empty((max(num_tokens, 1), hidden), dtype=torch.bfloat16, device=hidden_states.device)
    if eplb is not None:
        # Inside a CUDA-graph capture the forward must stay a pure replayable
        # kernel sequence: no planner (a collective with spin-waits whose
        # generation number would be baked into the graph), no side stream,
        # no event waits. Captured forwards remap onto whatever plan is
        # current at replay time.
        capturing = torch.cuda.is_current_stream_capturing()
        if early is not None:
            replan = early.decision
            decision_tokens = early.decision_tokens
        else:
            decision_tokens = _decision_tokens(num_tokens, group)
            replan = _decide(eplb, decision_tokens, capturing)
        if replan:
            if early is not None and early.plan_done is not None:
                # The plan of this chunk has been running on the side stream
                # since top-k; only the remap is left on the critical path.
                torch.cuda.current_stream().wait_event(early.plan_done)
                eplb.remap(num_tokens, early.block_m, after_plan=True)
                mode = "early plan | remap+kernel"
            else:
                eplb.prepare(num_tokens, eplb.block_m(decision_tokens))
                mode = "plan+copy+remap+kernel"
            _EPLB_STATS["plan"] += 1
            _EPLB_STATS["plan_tokens"] += num_tokens
        elif num_tokens > 0:
            eplb.remap(num_tokens, eplb.block_m(num_tokens))
            _EPLB_STATS["remap"] += 1
            _EPLB_STATS["remap_tokens"] += num_tokens
            mode = "remap+kernel, last plan"
        if trace_this:
            mid = torch.cuda.Event(enable_timing=True)
            mid.record()
            t_host1 = time.perf_counter()
        # With ``fuse_copy`` (the default; ``DG_EPLB_FUSE_COPY=0`` turns it
        # off) the planner's copy list is applied by the MegaMoE kernel itself
        # (its loaders fetch the changed helper slots and gate the dependent
        # tiles on per-slot readiness counters), so ``prepare`` skipped the
        # copy kernel.
        deep_gemm.mxfp4_mega_moe(
            y, eplb.l1_weights, eplb.l2_weights, buf,
            cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
            l1_global_scales=eplb.l1_global_scales, l2_global_scales=eplb.l2_global_scales,
            activation_clamp=clamp, fast_math=True,
            eplb_fused_copy=eplb.fused_copy_args() if eplb.fuse_copy else None)
    else:
        if trace_this:
            t_host1 = time.perf_counter()
        deep_gemm.mxfp4_mega_moe(
            y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf,
            l1_global_scales=moe.experts.mega_l1_global_scales,
            l2_global_scales=moe.experts.mega_l2_global_scales,
            activation_clamp=clamp, fast_math=True)
    if trace_this:
        global _TRACE_PENDING
        t_host2 = time.perf_counter()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        early_events = None
        if early is not None and early.ev_topk is not None and early.ev_shared is not None:
            early_events = (early.ev_topk, early.ev_shared) + tuple(early.side_events)
        _TRACE_PENDING = (_EPLB_STATS["large"], num_tokens, mode, start,
                          mid if eplb is not None else None, end,
                          eplb.copies.clone() if eplb is not None else None,
                          eplb.slot_expert.clone() if eplb is not None else None,
                          eplb.copy_count.clone() if eplb is not None else None,
                          ((t_host1 - t_host0) * 1e6, (t_host2 - t_host1) * 1e6),
                          early_events)
    return y[:num_tokens]
