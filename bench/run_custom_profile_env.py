#!/usr/bin/env python3
"""Run custom-only V4 profiling fixtures without importing Humming kernels."""

from __future__ import annotations

import runpy
import sys
import types

import torch


def quant_input(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline group-128 E4M3 quantization for profiler input construction."""
    groups = inputs.float().view(inputs.shape[0], -1, 128)
    scales = groups.abs().amax(dim=-1).clamp_min_(1.0e-12).div_(448.0)
    quantized = groups.div(scales.unsqueeze(-1)).clamp_(-448.0, 448.0)
    outputs.copy_(quantized.reshape_as(inputs).to(outputs.dtype))
    return outputs, scales.contiguous()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_custom_profile_env.py TARGET [ARGS...]")

    sys.path.insert(0, "/home/xutingz/fac")
    import v4_bench_env_runner as environment

    environment.install_pinned_sglang()

    humming = types.ModuleType("humming")
    humming_ops = types.ModuleType("humming.ops")
    humming_ops.quant_input = quant_input
    humming.ops = humming_ops
    sys.modules["humming"] = humming
    sys.modules["humming.ops"] = humming_ops

    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[:0] = [".", "bench"]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
