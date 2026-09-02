#!/usr/bin/env python3
"""Single-rank numerical test for the route-aware V4 Flash WGMMA kernels."""

from __future__ import annotations

import argparse

import torch

from humming import ops
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

import v4_flash_tp_wgmma as kernel


H = 4096
E = 256
TOP_K = 6
FP4 = None


def make_routes(m: int, pattern: str, device: torch.device):
    if pattern == "balanced":
        ids = torch.arange(m * TOP_K, dtype=torch.int32).view(m, TOP_K) % E
    else:
        ids = torch.arange(TOP_K, dtype=torch.int32).repeat(m, 1)
    weights = torch.arange(1, TOP_K + 1, dtype=torch.float32).repeat(m, 1)
    weights /= weights.sum(dim=1, keepdim=True)
    return ids.to(device), weights.to(device)


def dequant_braided(packed: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    global FP4
    n, half_k = packed.shape
    k = half_k * 2
    groups = k // 32
    pair = packed.view(n, groups, 16)
    nibble = torch.cat((pair & 0xF, pair >> 4), dim=-1).reshape(n, k)
    magnitude = FP4[(nibble & 7).long()]
    value = torch.where((nibble & 8).bool(), -magnitude, magnitude)
    scale = torch.exp2((exponent.int() - 127).float()).repeat_interleave(32, dim=1)
    return value * scale


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.double().flatten(), b.double().flatten(), dim=0
        ).item()
    )


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        (torch.linalg.vector_norm(a.double() - b.double())
         / torch.linalg.vector_norm(b.double()).clamp_min(1e-40)).item()
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--pattern", choices=("balanced", "skew"), default="balanced")
    parser.add_argument("--intermediate", type=int, choices=(256, 512), default=512)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(20260902)
    torch.cuda.manual_seed(20260902)
    global FP4
    FP4 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device
    )

    intermediate = args.intermediate
    n13 = 2 * intermediate
    # Random packed values are already in the braided physical layout consumed
    # by the inherited RS-WGMMA core.  Narrow scale exponents keep references
    # finite while exercising multiple E8M0 values.
    w13 = torch.randint(0, 256, (E, n13, H // 2), dtype=torch.uint8, device=device)
    s13 = torch.randint(125, 129, (E, n13, H // 32), dtype=torch.uint8, device=device)
    w2 = torch.randint(
        0, 256, (E, H, intermediate // 2), dtype=torch.uint8, device=device
    )
    s2 = torch.randint(
        125, 129, (E, H, intermediate // 32), dtype=torch.uint8, device=device
    )
    w13_reference = w13
    w2_reference = w2
    if kernel.MODE2_BRAID:
        # Preserve the ordinary braided values for the independent torch
        # reference, then convert only the kernel operands offline.
        w13_reference = w13.clone()
        w2_reference = w2.clone()
        kernel.braid_mode2_(w13)
        kernel.braid_mode2_(w2)

    topk_ids, topk_weights = make_routes(args.m, args.pattern, device)
    sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
        topk_ids, block_size=8, num_experts=E, ignore_invalid_expert=True
    )
    routes = args.m * TOP_K
    x = torch.randn((args.m, H), dtype=torch.bfloat16, device=device) * 0.1
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qx, x_scale = ops.quant_input(
        inputs=x,
        outputs=qx,
        dtype="float8e4m3",
        group_size=128,
        m_major_scale=False,
        scale_dtype="float32",
    )

    partials = torch.empty(
        (kernel.W13_MAX_SPLITS, routes, n13), dtype=torch.float32, device=device
    )
    activation = torch.empty(
        (routes, intermediate), dtype=torch.bfloat16, device=device
    )
    lut = kernel.make_e2m1_e8m0_lut(device)
    kernel.run_w13(
        w13,
        s13,
        qx.view(torch.uint8),
        x_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        partials,
        lut,
        intermediate,
    )
    qact = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    if kernel.FUSED_ACT_QUANT:
        act_scale = torch.empty(
            (routes, intermediate // 128), dtype=torch.float32, device=device
        )
        kernel.reduce_swiglu_quant(
            partials,
            activation,
            qact.view(torch.uint8),
            act_scale,
            intermediate,
        )
    else:
        kernel.reduce_swiglu(
            partials, activation, intermediate
        )
        qact, act_scale = ops.quant_input(
            inputs=activation,
            outputs=qact,
            dtype="float8e4m3",
            group_size=128,
            m_major_scale=False,
            scale_dtype="float32",
        )
    local = torch.zeros((args.m, H), dtype=torch.float32, device=device)
    down = (
        torch.empty((routes, H), dtype=torch.bfloat16, device=device)
        if kernel.W2_ROUTE_OUTPUT
        else None
    )
    w2_output = down if down is not None else local
    kernel.run_w2(
        w2,
        s2,
        qact.view(torch.uint8),
        act_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        topk_weights,
        w2_output,
        lut,
        intermediate,
    )
    output = torch.empty((args.m, H), dtype=torch.bfloat16, device=device)
    if kernel.W2_ROUTE_OUTPUT:
        assert down is not None
        moe_fused_mul_sum(
            inputs=down.view(args.m, TOP_K, H),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            is_ep=False,
            routed_scaling_factor=1.5,
            outputs=output,
        )
        local = output.float()
    else:
        kernel.cast_bf16(local, output)
    torch.cuda.synchronize()

    # Full all-route reference.  Only active experts are dequantized, one at a
    # time, to keep the temporary footprint small.
    x_dequant = qx.float() * x_scale.repeat_interleave(128, dim=1)
    gate_up_ref = torch.empty((routes, n13), dtype=torch.float32, device=device)
    flat_ids = topk_ids.flatten()
    for expert in torch.unique(flat_ids).tolist():
        route_index = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        token_index = torch.div(route_index, TOP_K, rounding_mode="floor")
        weight = dequant_braided(w13_reference[expert], s13[expert])
        gate_up_ref[route_index] = x_dequant[token_index] @ weight.t()
        del weight

    gate_bf = gate_up_ref[:, :intermediate].bfloat16().float()
    up_bf = gate_up_ref[:, intermediate:].bfloat16().float()
    activation_ref = (
        (torch.nn.functional.silu(gate_bf) * up_bf).bfloat16()
    )

    act_dequant = qact.float() * act_scale.repeat_interleave(128, dim=1)
    down_ref = torch.empty((routes, H), dtype=torch.float32, device=device)
    for expert in torch.unique(flat_ids).tolist():
        route_index = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
        weight = dequant_braided(w2_reference[expert], s2[expert])
        down_ref[route_index] = act_dequant[route_index] @ weight.t()
        del weight
    local_ref = (
        down_ref.view(args.m, TOP_K, H)
        * topk_weights[:, :, None]
    ).sum(dim=1) * 1.5

    selected_split_k = kernel.select_w13_split_k(routes)
    w13_actual = partials[:selected_split_k].sum(dim=0)
    print(
        "V4_WGMMA_CHECK "
        f"M={args.m} pattern={args.pattern} Is={intermediate} "
        f"active={torch.unique(flat_ids).numel()} padded={num_tokens_padded.item()} "
        f"split_k={selected_split_k} mode2={kernel.MODE2_BRAID} "
        f"fused_act_quant={kernel.FUSED_ACT_QUANT} "
        f"w2_global_lut={kernel.W2_GLOBAL_LUT}"
    )
    print(
        "V4_WGMMA_W13 "
        f"cos={cosine(w13_actual, gate_up_ref):.9f} "
        f"rel_l2={rel_l2(w13_actual, gate_up_ref):.9f}"
    )
    print(
        "V4_WGMMA_ACT "
        f"cos={cosine(activation, activation_ref):.9f} "
        f"rel_l2={rel_l2(activation, activation_ref):.9f}"
    )
    print(
        "V4_WGMMA_W2 "
        f"cos={cosine(local, local_ref):.9f} "
        f"rel_l2={rel_l2(local, local_ref):.9f} "
        f"finite={bool(torch.isfinite(output).all())} "
        f"absmax={float(output.float().abs().max().item()):.9g} "
        f"nonzero={int(torch.count_nonzero(output).item())}"
    )
    if cosine(local, local_ref) < 0.99 or not torch.isfinite(output).all():
        raise SystemExit("V4_WGMMA_WRONG")
    print("V4_WGMMA_OK")


if __name__ == "__main__":
    main()
