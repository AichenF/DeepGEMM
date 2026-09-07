#!/usr/bin/env python3
"""Single-GPU local-body smoke for the native V4 Flash MegaMoE kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

if os.environ.get("V4_NATIVE_TEST_H20_EXACT", "0") == "1":
    import v4_flash_tp_h20_exact_megamoe as native
else:
    import v4_flash_tp_native_megamoe as native


def dequant_marlin_weight(
    packed: torch.Tensor, exponent: torch.Tensor
) -> torch.Tensor:
    rows, half_k = packed.shape
    chunks = packed.view(rows, half_k // 4, 4)
    nibble = torch.cat((chunks >> 4, chunks & 0x0F), dim=-1).reshape(
        rows, half_k * 2
    )
    fp4 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    magnitude = fp4[(nibble & 7).long()]
    value = torch.where((nibble & 8).bool(), -magnitude, magnitude)
    scale = torch.exp2((exponent.int() - 127).float()).repeat_interleave(
        32, dim=1
    )
    return value * scale


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.double().flatten(), reference.double().flatten(), dim=0
        ).item()
    )


def rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(actual.double() - reference.double())
            / torch.linalg.vector_norm(reference.double()).clamp_min(1e-40)
        ).item()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="synchronize and exit immediately after the captured native launch",
    )
    parser.add_argument(
        "--phase-stamps",
        action="store_true",
        help="print local-body globaltimer stamps (requires V4_NATIVE_PHASE_STAMPS=1)",
    )
    args = parser.parse_args()
    if args.m not in (8, 16, 32, 64, 128):
        parser.error("--m must be one of 8,16,32,64,128")
    if args.phase_stamps and not native.NATIVE_PHASE_STAMPS:
        parser.error("--phase-stamps requires V4_NATIVE_PHASE_STAMPS=1")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    intermediate = 2048 // args.tp
    n13 = 2 * intermediate
    torch.manual_seed(20260904)
    w13 = torch.randint(
        0, 256, (256, n13, 2048), dtype=torch.uint8, device=device
    )
    s13 = torch.randint(
        125, 129, (256, n13, 128), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0,
        256,
        (256, 4096, intermediate // 2),
        dtype=torch.uint8,
        device=device,
    )
    s2 = torch.randint(
        125,
        129,
        (256, 4096, intermediate // 32),
        dtype=torch.uint8,
        device=device,
    )
    native_w13, native_w2, native_g13, native_g2, native_s13, native_s2 = (
        native.transform_weights(w13, s13, w2, s2)
    )

    workspace = native.allocate_workspace(intermediate, device)
    qx = (torch.randn((args.m, 4096), device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    x_scale = (
        1.0
        + torch.arange(args.m, dtype=torch.float32, device=device)[:, None]
        * 0.01
        + torch.arange(32, dtype=torch.float32, device=device)[None, :]
        * 0.0001
    )
    topk_ids = (
        torch.arange(args.m * 6, dtype=torch.int64, device=device)
        .view(args.m, 6)
        .remainder_(256)
    )
    topk_weights = torch.arange(
        1, 7, dtype=torch.float32, device=device
    ).repeat(args.m, 1)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    workspace.load_inputs(qx, x_scale, topk_ids, topk_weights)
    output = torch.empty(
        (args.m, 4096), dtype=torch.bfloat16, device=device
    )
    cold_l2 = None
    if args.phase_stamps:
        # The diagnostic is a timing measurement too: evict the H20 L2 with a
        # separate allocation excluded from the stamped business kernel.
        cold_l2 = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        cold_l2.zero_()
    phase_storage = native.run_local(
        workspace,
        native_w13,
        native_w2,
        native_s13,
        native_s2,
        native_g13,
        native_g2,
        output,
        args.m,
    )
    torch.cuda.synchronize()

    if args.phase_stamps:
        num_ctas = 156 if native.NATIVE_TWO_CTA_PER_SM else 78
        stamps = phase_storage.view(torch.int64)[: 4 + 2 * num_ctas].cpu()
        entry = int(stamps[0])
        route = int(stamps[1])
        all_gemm = int(stamps[2])
        combine = int(stamps[3])
        w13_stamps = stamps[4 : 4 + num_ctas]
        w2_stamps = stamps[4 + num_ctas : 4 + 2 * num_ctas]
        w13_nonzero = w13_stamps[w13_stamps != 0]
        w2_nonzero = w2_stamps[w2_stamps != 0]
        if not entry or not route or not all_gemm or not combine:
            raise RuntimeError(f"missing phase boundary stamp: {stamps[:4].tolist()}")
        if not w13_nonzero.numel() or not w2_nonzero.numel():
            raise RuntimeError("missing per-CTA W13/W2 completion stamps")
        w13_last = int(w13_nonzero.max())
        w2_last = int(w2_nonzero.max())
        if not (entry <= route <= all_gemm <= combine):
            raise RuntimeError(
                "phase boundary stamps are not monotonic: "
                f"{[entry, route, all_gemm, combine]}"
            )
        to_us = lambda end: (end - entry) / 1000.0
        print(
            "NATIVE_PHASE_STAMPS "
            + json.dumps(
                {
                    "m": args.m,
                    "num_ctas": num_ctas,
                    "route_publish_us": to_us(route),
                    "w13_last_us": to_us(w13_last),
                    "w2_last_us": to_us(w2_last),
                    "all_gemm_us": to_us(all_gemm),
                    "combine_only_us": (combine - all_gemm) / 1000.0,
                    "local_body_us": to_us(combine),
                    "w13_ctas": int(w13_nonzero.numel()),
                    "w2_ctas": int(w2_nonzero.numel()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Nsight Compute kernel replay restores mutable workspace allocations to
    # their pre-launch contents after the final pass.  The detailed numerical
    # audit below therefore cannot consume route/pool metadata after replay.
    if args.profile_only:
        print(
            "NATIVE_LOCAL_PROFILE_ONLY "
            + json.dumps({"m": args.m, "synchronized": True}),
            flush=True,
        )
        return

    # Reconstruct the expert-major padded BM8 pool row for every route.  The
    # old `route * BLOCK_M` shortcut was valid only while routes were unique
    # by expert and made the M128 post-kernel diagnostic index out of bounds.
    route_indices = torch.arange(args.m * 6, device=device)
    flat_experts = topk_ids.flatten()
    expert_counts = torch.bincount(
        flat_experts, minlength=native.NUM_EXPERTS
    )
    padded_counts = (
        (expert_counts + native.BLOCK_M - 1) // native.BLOCK_M
    ) * native.BLOCK_M
    expert_bases = torch.cumsum(padded_counts, dim=0) - padded_counts
    route_order = torch.arange(flat_experts.numel(), device=device)
    expert_ordinals = (
        (flat_experts[:, None] == flat_experts[None, :])
        & (route_order[None, :] < route_order[:, None])
    ).sum(dim=1)
    pool_rows = expert_bases[flat_experts] + expert_ordinals
    src_tokens = torch.div(route_indices, native.TOP_K, rounding_mode="floor")
    src_topk = route_indices.remainder(native.TOP_K)
    pooled_x = workspace.l1_acts.index_select(0, pool_rows)
    expected_x = qx.index_select(0, src_tokens)
    pooled_sf = workspace.l1_acts_sf[:, pool_rows].T.contiguous()
    expected_sf = x_scale.index_select(0, src_tokens)
    if native.NATIVE_FOLD_W13_GLOBAL_SCALE:
        expected_sf = expected_sf * native_g13.index_select(
            0, flat_experts
        )[:, None]
    pooled_weights = workspace.l1_topk_weights.index_select(0, pool_rows)
    expected_weights = topk_weights[src_tokens, src_topk]
    x_mismatch_bytes = int(
        (pooled_x.view(torch.uint8) != expected_x.view(torch.uint8)).sum()
    )
    sf_max_abs = float((pooled_sf - expected_sf).abs().max())
    weight_max_abs = float((pooled_weights - expected_weights).abs().max())

    x0 = qx[0].float() * x_scale[0].repeat_interleave(128)
    w13_expert0 = dequant_marlin_weight(w13[0], s13[0])
    gate_up = x0 @ w13_expert0.T
    gate_fp32, up_fp32 = gate_up.chunk(2)
    swiglu_fp32 = torch.nn.functional.silu(gate_fp32) * up_fp32
    gate_bf16 = gate_fp32.bfloat16().float()
    up_bf16 = up_fp32.bfloat16().float()
    swiglu_bf16 = (
        torch.nn.functional.silu(gate_bf16) * up_bf16
    ).bfloat16().float()
    native_l2_scale = workspace.l2_acts_sf[: intermediate // 128, 0]
    native_l2 = workspace.l2_acts[0].float() * native_l2_scale.repeat_interleave(
        128
    )
    native_vs_fp32 = {
        "cosine": cosine(native_l2, swiglu_fp32),
        "rel_l2": rel_l2(native_l2, swiglu_fp32),
    }
    native_vs_bf16 = {
        "cosine": cosine(native_l2, swiglu_bf16),
        "rel_l2": rel_l2(native_l2, swiglu_bf16),
    }
    route0_weight = topk_weights[0, 0]
    native_vs_weighted_bf16 = {
        "cosine": cosine(native_l2, swiglu_bf16 * route0_weight),
        "rel_l2": rel_l2(native_l2, swiglu_bf16 * route0_weight),
    }
    print(
        "NATIVE_LOCAL_RESULT "
        + json.dumps(
            {
                "m": args.m,
                "tp": args.tp,
                "intermediate_per_rank": intermediate,
                "finite": bool(torch.isfinite(output).all()),
                "max_abs": float(output.float().abs().max()),
                "l1_x_mismatch_bytes": x_mismatch_bytes,
                "l1_sf_max_abs": sf_max_abs,
                "l1_weight_max_abs": weight_max_abs,
                "output_sha256": hashlib.sha256(
                    output.view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
                "native_l2_vs_torch_fp32": native_vs_fp32,
                "native_l2_vs_torch_bf16": native_vs_bf16,
                "native_l2_vs_torch_weighted_bf16": (
                    native_vs_weighted_bf16
                ),
                "workspace_bytes": workspace.storage.numel(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
