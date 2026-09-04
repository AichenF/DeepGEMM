#!/usr/bin/env python3
"""Single-GPU local-body smoke for the native V4 Flash MegaMoE kernel."""

from __future__ import annotations

import argparse
import json

import torch

import v4_flash_tp_native_megamoe as native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    args = parser.parse_args()
    if args.m not in (8, 16, 32, 64, 128):
        parser.error("--m must be one of 8,16,32,64,128")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(20260904)
    w13 = torch.randint(
        0, 256, (256, 1024, 2048), dtype=torch.uint8, device=device
    )
    s13 = torch.randint(
        125, 129, (256, 1024, 128), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 256, (256, 4096, 256), dtype=torch.uint8, device=device
    )
    s2 = torch.randint(
        125, 129, (256, 4096, 16), dtype=torch.uint8, device=device
    )
    native_w13, native_w2 = native.transform_weights(w13, s13, w2, s2)

    workspace = native.allocate_workspace(512, device)
    qx = (torch.randn((args.m, 4096), device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    x_scale = torch.ones((args.m, 32), dtype=torch.float32, device=device)
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
    native.run_local(workspace, native_w13, native_w2, output, args.m)
    torch.cuda.synchronize()
    print(
        "NATIVE_LOCAL_RESULT "
        + json.dumps(
            {
                "m": args.m,
                "finite": bool(torch.isfinite(output).all()),
                "max_abs": float(output.float().abs().max()),
                "workspace_bytes": workspace.storage.numel(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
