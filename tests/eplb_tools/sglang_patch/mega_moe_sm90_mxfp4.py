"""SM90 MXFP4 MegaMoE forward path for SGLang, with optional dynamic EPLB.

Drop-in companion to ``mega_moe_sm90.py``: FP4 experts (DeepSeek V4 style
packed E2M1 + E8M0 scales, or MiMo's MXFP4 store) go through DeepGEMM's fused
SM90 MXFP4 MegaMoE kernel. ``SGLANG_MEGAMOE_DYNAMIC_EPLB_SLOTS=S`` (S > 0)
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
_MXFP4_SYMM_BUFFER: dict = {}
# ``SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE=N`` (N > 0): rank 0 reports every N planned
# layer forwards how many forwards planned vs. remapped onto the last placement,
# their mean local token counts and the replica count of the latest plan.
_EPLB_TRACE = int(os.environ.get("SGLANG_MEGAMOE_DYNAMIC_EPLB_TRACE", "0"))
_EPLB_STATS = {"plan": 0, "plan_tokens": 0, "remap": 0, "remap_tokens": 0}


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
    s13 = as_e8m0_u8(experts.w13_weight_scale_inv)
    s2 = as_e8m0_u8(experts.w2_weight_scale_inv)
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
    routed_scaling = 1.0 if moe.experts.should_fuse_routed_scaling_factor_in_topk else float(moe.routed_scaling_factor)
    clamp = getattr(moe.config, "swiglu_limit", None)

    eplb = None
    if _EPLB_SLOTS > 0:
        eplb = _get_eplb(moe, group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)
        buf = eplb.symm
    else:
        buf = _get_symm_buffer(group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate)

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
        if decision_tokens >= _EPLB_MIN_TOKENS:
            eplb.prepare(num_tokens, eplb.block_m(decision_tokens))
            _EPLB_STATS["plan"] += 1
            _EPLB_STATS["plan_tokens"] += num_tokens
            if _EPLB_TRACE and _EPLB_STATS["plan"] % _EPLB_TRACE == 0 and group.rank() == 0:
                st = _EPLB_STATS
                logging.getLogger(__name__).info(
                    "[megamoe-eplb] planned %d layer forwards (mean %.0f tokens), remap-only %d (mean %.0f tokens), "
                    "latest plan: %d weight copies on rank 0", st["plan"], st["plan_tokens"] / st["plan"],
                    st["remap"], st["remap_tokens"] / max(st["remap"], 1), int(eplb.copy_count.item()))
        elif num_tokens > 0:
            eplb.remap(num_tokens, eplb.block_m(num_tokens))
            _EPLB_STATS["remap"] += 1
            _EPLB_STATS["remap_tokens"] += num_tokens
        deep_gemm.mxfp4_mega_moe(
            y, eplb.l1_weights, eplb.l2_weights, buf,
            cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
            activation_clamp=clamp, fast_math=True)
    else:
        deep_gemm.mxfp4_mega_moe(
            y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf,
            activation_clamp=clamp, fast_math=True)
    return y[:num_tokens]
