"""SM90 MXFP4 MegaMoE forward path for SGLang, with optional dynamic EPLB.

Drop-in companion to ``mega_moe_sm90.py``: FP4 experts (DeepSeek V4 style
packed E2M1 + E8M0 scales, or MiMo's MXFP4 store) go through DeepGEMM's fused
SM90 MXFP4 MegaMoE kernel. ``install_mxfp4_megamoe.py`` wires both the
DeepSeek V2/V4 and the MiMo V2 MoE modules to ``run_sm90_mxfp4_mega_routed``. ``SGLANG_MEGAMOE_DYNAMIC_EPLB_SLOTS=S`` (S > 0)
turns on dynamic expert load balancing with S helper slots per rank; 0 or
unset is fixed placement on the same kernel, so the two arms differ only in
the balancer. ``SGLANG_MEGAMOE_DYNAMIC_EPLB_MIN_TOKENS`` (default 512) is the
smallest local token count that triggers a new plan.
"""

from __future__ import annotations

import logging
import os
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
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_DEFERRED=1``: after its first plan a layer
# never plans on the critical path. A large forward remaps onto the plan made
# from the previous large forward of the same layer, launches the kernel, and
# then plans and copies weights for the next one on a side stream, behind the
# remaining layers of the step (the planner keeps a private copy of the
# logical routing because the shared buffer moves on). Every forward of the
# layer first waits for its pending plan, and the last decoder layer drains
# all pending plans so CUDA-graph decode steps never overlap a weight copy.
# ``REPLAN_EVERY`` still sets how often the deferred plan is issued.
_EPLB_DEFERRED = os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_DEFERRED", "0") == "1"
_EPLB_SIDE_STREAM = None
_EPLB_PENDING: list = []
_MXFP4_SYMM_BUFFER: dict = {}
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE=N`` (N > 0): rank 0 times every N-th
# layer forward with at least ``_TRACE_MIN_TOKENS`` local tokens (CUDA events
# around the whole MoE call, read back one trace period later so nothing
# stalls) and logs it, with the planned/remap-only counters and the rank
# loads of the plan that forward used. Works with and without the balancer,
# so the two arms can be compared layer by layer inside the server.
_EPLB_TRACE = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE", "0"))
_EPLB_STATS = {"plan": 0, "plan_tokens": 0, "remap": 0, "remap_tokens": 0, "large": 0, "deferred": 0}
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
    ordinal, tokens, mode, start, mid, end, copies, slots, copy_count = _TRACE_PENDING
    _TRACE_PENDING = None
    end.synchronize()
    msg = (f"[megamoe-trace] large forward #{ordinal}: {tokens} local tokens, {mode}, "
           f"{start.elapsed_time(end) * 1000:.0f} us")
    if eplb is not None:
        st = _EPLB_STATS
        msg += (f" (prepare {start.elapsed_time(mid) * 1000:.0f} + kernel {mid.elapsed_time(end) * 1000:.0f})"
                f" | planned {st['plan']} (mean {st['plan_tokens'] / max(st['plan'], 1):.0f} tokens), "
                f"remap-only {st['remap']} (mean {st['remap_tokens'] / max(st['remap'], 1):.0f} tokens), "
                f"deferred plans {st['deferred']}"
                f" | its plan: {_plan_summary(copies, slots, eplb.home_experts)}, "
                f"{int(copy_count.item())} weight copies on rank 0")
    logging.getLogger(__name__).info(msg)


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
        eplb.set_home_weights(l1, l2)
        del l1, l2
        torch.cuda.empty_cache()
        experts._mega_moe_eplb = eplb
        return
    experts.mega_l1_weights = l1
    experts.mega_l2_weights = l2


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


def _wait_deferred(eplb) -> None:
    ev = getattr(eplb, "_deferred_event", None)
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
        eplb._deferred_event = None


def _drain_deferred() -> None:
    for eplb in _EPLB_PENDING:
        _wait_deferred(eplb)
    _EPLB_PENDING.clear()


def _launch_deferred(eplb, num_tokens: int, block_m: int) -> None:
    """Plan and copy weights for the next large forward of this layer on the
    side stream, once the kernel that still reads the helper slots is done."""
    global _EPLB_SIDE_STREAM
    if _EPLB_SIDE_STREAM is None:
        _EPLB_SIDE_STREAM = torch.cuda.Stream()
    kernel_done = torch.cuda.Event()
    kernel_done.record()
    with torch.cuda.stream(_EPLB_SIDE_STREAM):
        _EPLB_SIDE_STREAM.wait_event(kernel_done)
        eplb.plan(num_tokens, block_m, topk_idx=eplb._deferred_topk)
        eplb.copy_weights()
        done = torch.cuda.Event()
        done.record()
    eplb._deferred_event = done
    _EPLB_PENDING.append(eplb)
    _EPLB_STATS["deferred"] += 1


def _get_eplb(moe, group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden):
    eplb = getattr(moe.experts, "_mega_moe_eplb", None)
    if eplb is None:
        eplb = _new_eplb(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
        eplb.set_home_weights(moe.experts.mega_l1_weights, moe.experts.mega_l2_weights)
        moe.experts.mega_l1_weights = None
        moe.experts.mega_l2_weights = None
        torch.cuda.empty_cache()
        moe.experts._mega_moe_eplb = eplb
    assert eplb.num_topk == num_topk and eplb.num_max_tokens_per_rank >= num_max_tokens_per_rank, \
        f"EPLB sized for topk={eplb.num_topk}, cap={eplb.num_max_tokens_per_rank}; forward needs {num_topk}, {num_max_tokens_per_rank}"
    return eplb


def run_sm90_mxfp4_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_max_tokens_per_rank: int,
) -> torch.Tensor:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group = get_moe_ep_group().device_group
    hidden = moe.config.hidden_size
    intermediate = moe.config.moe_intermediate_size
    num_experts = moe.experts.num_experts
    top_k = topk_ids.shape[1] if num_tokens > 0 else moe.config.num_experts_per_tok
    routed_scaling = 1.0 if moe.experts.should_fuse_routed_scaling_factor_in_topk \
        else float(getattr(moe, "routed_scaling_factor", None) or 1.0)
    clamp = getattr(moe.config, "swiglu_limit", None)

    eplb = None
    if _EPLB_SLOTS > 0:
        eplb = _get_eplb(moe, group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)
        buf = eplb.symm
    else:
        buf = _get_symm_buffer(group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)

    trace_this = False
    if _EPLB_TRACE and num_tokens >= _TRACE_MIN_TOKENS and group.rank() == 0:
        _EPLB_STATS["large"] += 1
        trace_this = _EPLB_STATS["large"] % _EPLB_TRACE == 0
    if trace_this:
        _trace_flush(eplb)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
    mode = "kernel, fixed placement"

    if num_tokens > 0:
        x_fp8, x_sf = per_token_cast_to_fp8(hidden_states.to(torch.bfloat16), use_ue8m0=False, gran_k=128)
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_sf)
        buf.topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
        buf.topk_weights[:num_tokens].copy_((topk_weights * routed_scaling).to(torch.float32))

    y = torch.empty((max(num_tokens, 1), hidden), dtype=torch.bfloat16, device=hidden_states.device)
    if eplb is not None:
        # The planner is a collective: every rank must take the same branch
        # and use the same block size, so both derive from a token count all
        # ranks agree on. SGLang publishes the batch's token counts: one entry
        # per DP rank with DP attention, otherwise one global entry while the
        # MoE input is sequence-sharded across the EP ranks.
        from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
        global_num_tokens = get_dp_global_num_tokens()
        if not global_num_tokens:
            decision_tokens = num_tokens
        elif len(global_num_tokens) > 1:
            decision_tokens = max(global_num_tokens)
        else:
            decision_tokens = (global_num_tokens[0] + group.size() - 1) // group.size()
        # Inside a CUDA-graph capture the forward must stay a pure replayable
        # kernel sequence: no planner (a collective with spin-waits whose
        # generation number would be baked into the graph), no side stream,
        # no event waits. Captured forwards remap onto whatever plan is
        # current at replay time; the deferred plans are drained before any
        # capture by the last layer of the preceding eager step.
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            _wait_deferred(eplb)
        replan = schedule = False
        if decision_tokens >= _EPLB_MIN_TOKENS and not capturing:
            large = getattr(eplb, "_large_forwards", 0)
            eplb._large_forwards = large + 1
            if eplb.generation == 0:
                replan = True
            elif _EPLB_DEFERRED:
                schedule = large % _EPLB_REPLAN_EVERY == 0
            else:
                replan = large % _EPLB_REPLAN_EVERY == 0
        if replan:
            eplb.prepare(num_tokens, eplb.block_m(decision_tokens))
            _EPLB_STATS["plan"] += 1
            _EPLB_STATS["plan_tokens"] += num_tokens
            mode = "plan+copy+remap+kernel"
        else:
            if schedule:
                if getattr(eplb, "_deferred_topk", None) is None:
                    eplb._deferred_topk = torch.empty((eplb.num_max_tokens_per_rank, top_k),
                                                      dtype=torch.int64, device=hidden_states.device)
                eplb._deferred_topk[:num_tokens].copy_(topk_ids.to(torch.int64))
            if num_tokens > 0:
                eplb.remap(num_tokens, eplb.block_m(num_tokens))
                _EPLB_STATS["remap"] += 1
                _EPLB_STATS["remap_tokens"] += num_tokens
                mode = "remap+kernel, deferred plan" if _EPLB_DEFERRED else "remap+kernel, last plan"
        if trace_this:
            mid = torch.cuda.Event(enable_timing=True)
            mid.record()
        deep_gemm.mxfp4_mega_moe(
            y, eplb.l1_weights, eplb.l2_weights, buf,
            cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
            activation_clamp=clamp, fast_math=True)
    else:
        deep_gemm.mxfp4_mega_moe(
            y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf,
            activation_clamp=clamp, fast_math=True)
    if trace_this:
        global _TRACE_PENDING
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        _TRACE_PENDING = (_EPLB_STATS["large"], num_tokens, mode, start,
                          mid if eplb is not None else None, end,
                          eplb.copies.clone() if eplb is not None else None,
                          eplb.slot_expert.clone() if eplb is not None else None,
                          eplb.copy_count.clone() if eplb is not None else None)
    if eplb is not None and _EPLB_DEFERRED and not capturing:
        if schedule:
            _launch_deferred(eplb, num_tokens, eplb.block_m(decision_tokens))
        if getattr(moe, "layer_id", None) == moe.config.num_hidden_layers - 1:
            _drain_deferred()
    return y[:num_tokens]
