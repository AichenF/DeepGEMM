"""Eight-rank correctness coverage for the SM120 routed-MoE fast path.

The distributed test is opt-in because one session reserves the full two-slot
GIN transport. Run it on eight SM120 GPUs with::

    RUN_SM120_ROUTED_MOE_E2E=1 torchrun --nproc-per-node=8 \
        -m pytest -q tests/test_routed_moe_sm120_e2e.py

The numeric oracle starts from deterministic input and weight encodings and
does not read kernel intermediates.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import pytest
import torch

from deep_gemm.testing.routed_moe_sm120 import (
    ACTIVATION_CLAMP,
    CONTRACT,
    HIDDEN,
    INTERMEDIATE,
    K_GROUP,
    LOCAL_EXPERTS,
    MAX_ROWS,
    ROUTE_WEIGHTS,
    TASK_ROWS,
    TOP_K,
    WORLD_SIZE,
    ModelContract,
    deterministic_fp4_value,
    deterministic_input_encoding,
    deterministic_route_expert,
    deterministic_weight_exponents,
    epoch_slots,
    make_dsv4_w4a8_inputs,
    make_dsv4_w4a8_weights,
)


def source_order_combine(partials: Any) -> Any:
    """Accumulate route slots in their original top-k order, then round BF16."""

    if partials.ndim != 3 or partials.shape[1] != TOP_K:
        raise ValueError(f"partials must have shape [tokens, {TOP_K}, hidden]")
    combined = torch.zeros(
        (partials.shape[0], partials.shape[2]),
        dtype=torch.float32,
        device=partials.device,
    )
    for route_slot in range(TOP_K):
        combined += partials[:, route_slot].float()
    return combined.to(torch.bfloat16)


def _physical_gate_up(gate: Any, up: Any) -> Any:
    if gate.shape != up.shape or gate.shape[-1] != INTERMEDIATE:
        raise ValueError("W1 must contain one gate and one up projection")
    output = torch.empty(
        (*gate.shape[:-1], 2 * INTERMEDIATE),
        dtype=gate.dtype,
        device=gate.device,
    )
    logical = torch.arange(INTERMEDIATE, device=gate.device)
    physical_gate = (logical // 8) * 16 + logical % 8
    output[..., physical_gate] = gate
    output[..., physical_gate + 8] = up
    return output


def _split_physical_gate_up(w1_bf16: Any) -> tuple[Any, Any]:
    if w1_bf16.shape[-1] != 2 * INTERMEDIATE:
        raise ValueError("W1 output width must be 2 * intermediate")
    logical = torch.arange(INTERMEDIATE, device=w1_bf16.device)
    physical_gate = (logical // 8) * 16 + logical % 8
    return w1_bf16[..., physical_gate], w1_bf16[..., physical_gate + 8]


def _requantize_k32(weighted: Any) -> tuple[Any, Any, Any]:
    if weighted.ndim != 2 or weighted.shape[1] != INTERMEDIATE:
        raise ValueError("SwiGLU result must have the down-projection width")
    grouped = weighted.reshape(weighted.shape[0], INTERMEDIATE // K_GROUP, K_GROUP)
    amax = grouped.abs().amax(dim=2)
    raw_scale = (amax * (1.0 / 448.0)).contiguous()
    bits = raw_scale.view(torch.int32)
    exponent = (
        ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
    ).clamp(max=254).to(torch.uint8)
    inverse = ((254 - exponent.to(torch.int32)) << 23).view(torch.float32)
    fp8 = (grouped * inverse[:, :, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float32),
        exponent.to(torch.int32) - 127,
    )
    dequantized = fp8.view(torch.float8_e4m3fn).float() * scale[:, :, None]
    return fp8.reshape(weighted.shape), exponent, dequantized.reshape(weighted.shape)


def _decode_input(source_rank: int, token: int, device: Any) -> Any:
    codes, exponents = deterministic_input_encoding(source_rank, token, device)
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32),
        exponents.to(torch.int32) - 127,
    )
    return codes.view(torch.float8_e4m3fn).float() * scales[:, None]


def _expected_route(
    source_rank: int,
    token: int,
    route_slot: int,
    global_expert: int,
    device: Any,
    *,
    round_swiglu_output: bool = False,
) -> dict[str, Any]:
    """Compute one route solely from its input and deterministic weight recipe."""

    x = _decode_input(source_rank, token, device)
    x_group_sum = x.sum(dim=1)
    w1_scale_exp = deterministic_weight_exponents(global_expert, "w1", device)
    w1_scale = torch.ldexp(
        torch.ones_like(w1_scale_exp, dtype=torch.float32),
        w1_scale_exp.to(torch.int32) - 127,
    )
    common = (x_group_sum * w1_scale).sum()
    logical = torch.arange(INTERMEDIATE, device=device)
    gate = (common * deterministic_fp4_value(global_expert, "gate", logical)).to(
        torch.bfloat16
    )
    up = (common * deterministic_fp4_value(global_expert, "up", logical)).to(
        torch.bfloat16
    )
    w1_bf16 = _physical_gate_up(
        gate[None, :], up[None, :]
    )

    gate_f32, up_f32 = _split_physical_gate_up(w1_bf16)
    gate_f32 = gate_f32.float().clamp(max=ACTIVATION_CLAMP)
    up_f32 = up_f32.float().clamp(
        min=-ACTIVATION_CLAMP, max=ACTIVATION_CLAMP
    )
    silu = gate_f32 * (
        1.0 / (1.0 + torch.exp2(-gate_f32 * 1.4426950408889634))
    )
    weighted = silu * up_f32 * float(ROUTE_WEIGHTS[route_slot])
    if round_swiglu_output:
        weighted = weighted.to(torch.bfloat16).float()
    intermediate_fp8, intermediate_scale, intermediate = _requantize_k32(weighted)

    w2_scale_exp = deterministic_weight_exponents(global_expert, "w2", device)
    w2_scale = torch.ldexp(
        torch.ones_like(w2_scale_exp, dtype=torch.float32),
        w2_scale_exp.to(torch.int32) - 127,
    )
    down_weight = deterministic_fp4_value(global_expert, "down", logical)
    down = (
        intermediate.reshape(INTERMEDIATE // K_GROUP, K_GROUP)
        * w2_scale[:, None]
        * down_weight.reshape(INTERMEDIATE // K_GROUP, K_GROUP)
    ).sum()
    output = torch.arange(HIDDEN, device=device)
    output_sign = torch.where(
        ((output // 64 + global_expert) & 1) == 0,
        1.0,
        -1.0,
    )
    down = (down * output_sign).to(torch.bfloat16)
    return {
        "w1_bf16": w1_bf16[0],
        "intermediate_fp8": intermediate_fp8[0],
        "intermediate_scale": intermediate_scale[0],
        "w2_bf16": down,
    }


def _distributed_backend() -> tuple[Any, Any, int]:
    import deep_gemm

    if not torch.cuda.is_available() or not deep_gemm._C.has_sm120_routed_moe():
        pytest.skip("DeepGEMM was not built with the SM120 NCCL GIN path")
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip("run with torchrun --nproc-per-node=8")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    group = torch.distributed.group.WORLD
    torch.distributed.barrier(group=group, device_ids=[local_rank])
    control_group = torch.distributed.new_group(backend="gloo")
    return group, control_group, torch.distributed.get_rank(group)


def _expected_tokens(rank: int, tokens: tuple[int, ...], device: Any) -> Any:
    rows = []
    for token in tokens:
        routes = []
        for slot in range(TOP_K):
            expert = deterministic_route_expert(rank, token, slot)
            routes.append(
                _expected_route(rank, token, slot, expert, device)["w2_bf16"]
            )
        rows.append(torch.stack(routes))
    return source_order_combine(torch.stack(rows))


def _expected_local_work(rank: int, active_rows: int) -> tuple[int, int]:
    local_counts = [0] * LOCAL_EXPERTS
    for source in range(WORLD_SIZE):
        for token in range(active_rows):
            for slot in range(TOP_K):
                expert = deterministic_route_expert(source, token, slot)
                if expert // LOCAL_EXPERTS == rank:
                    local_counts[expert % LOCAL_EXPERTS] += 1
    expected_routes = sum(local_counts)
    expected_tasks = sum(
        (count + TASK_ROWS - 1) // TASK_ROWS for count in local_counts
    )
    return expected_routes, expected_tasks


def test_dsv4_contract_is_gate_up_down_w4a8_k32() -> None:
    assert CONTRACT == ModelContract()
    assert CONTRACT.w1 == "gate+up"
    assert CONTRACT.w2 == "down"
    assert K_GROUP == 32
    gate = torch.arange(INTERMEDIATE, dtype=torch.bfloat16).reshape(1, -1)
    up = -gate
    physical = _physical_gate_up(gate, up)
    observed_gate, observed_up = _split_physical_gate_up(physical)
    assert torch.equal(observed_gate, gate)
    assert torch.equal(observed_up, up)


def test_requant_zero_group_uses_canonical_ue8m0_scale() -> None:
    zero = torch.zeros((1, INTERMEDIATE), dtype=torch.float32)
    fp8, exponent, dequantized = _requantize_k32(zero)
    assert not fp8.any()
    assert not dequantized.any()
    assert not exponent.any()


def test_fixture_detects_extra_swiglu_bf16_rounding() -> None:
    canonical = _expected_route(0, 1, 0, 17, "cpu")
    rounded = _expected_route(
        0,
        1,
        0,
        17,
        "cpu",
        round_swiglu_output=True,
    )
    assert not torch.equal(canonical["intermediate_fp8"], rounded["intermediate_fp8"])
    assert not torch.equal(canonical["w2_bf16"], rounded["w2_bf16"])


def test_source_order_reduce_and_epoch_slot_reuse() -> None:
    partials = torch.zeros((1, TOP_K, 1), dtype=torch.bfloat16)
    partials[0, :, 0] = torch.tensor(
        [1.0e20, -1.0e20, 1.0, 0.5, 0.25, 0.125], dtype=torch.bfloat16
    )
    expected = torch.zeros((1, 1), dtype=torch.float32)
    for slot in range(TOP_K):
        expected += partials[:, slot].float()
    assert torch.equal(source_order_combine(partials), expected.to(torch.bfloat16))
    reverse_accumulator = torch.zeros_like(expected)
    for slot in reversed(range(TOP_K)):
        reverse_accumulator += partials[:, slot].float()
    assert not torch.equal(
        source_order_combine(partials), reverse_accumulator.to(torch.bfloat16)
    )
    assert epoch_slots([0, 1, 2]) == [0, 1, 0]


def test_oracle_does_not_read_kernel_stage_mirrors() -> None:
    source = inspect.getsource(_expected_route)
    assert "w1_d" not in source.lower()
    assert "w2_d" not in source.lower()


@pytest.mark.skipif(
    os.environ.get("RUN_SM120_ROUTED_MOE_E2E") != "1",
    reason="set RUN_SM120_ROUTED_MOE_E2E=1 for the full eight-rank allocation",
)
def test_public_api_eight_rank_correctness() -> None:
    group, control_group, rank = _distributed_backend()
    import deep_gemm

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    active_rows = int(os.environ.get("SM120_ROUTED_MOE_E2E_ROWS", "1"))
    epochs = int(os.environ.get("SM120_ROUTED_MOE_E2E_EPOCHS", "3"))
    if not 1 <= active_rows <= MAX_ROWS or not 1 <= epochs <= 8:
        raise ValueError("E2E rows must be in [1, 8192] and epochs in [1, 8]")
    sample_tokens = (
        tuple(range(active_rows))
        if active_rows <= 8
        else (0, 1, active_rows // 2, active_rows - 1)
    )
    inputs = make_dsv4_w4a8_inputs(rank, active_rows, device)
    weights = make_dsv4_w4a8_weights(rank, device)
    expected = _expected_tokens(rank, sample_tokens, device)
    session = None
    workspace = None
    try:
        torch.distributed.barrier(group=control_group)
        session = deep_gemm.SM120RoutedMoESession(group, device)
        torch.distributed.barrier(group=control_group)
        workspace = deep_gemm.SM120RoutedMoEWorkspace(device)
        observed_slots = []
        for epoch in range(epochs):
            workspace.output[:active_rows].fill_(float("nan"))
            output = deep_gemm.fp8_fp4_routed_moe_sm120(
                session,
                workspace,
                inputs.x,
                inputs.x_scales,
                inputs.topk_indices,
                inputs.topk_weights,
                weights.w1_up_gate,
                weights.w2_down,
            )
            torch.cuda.synchronize(device)
            observed_slots.append(epoch & 1)
            torch.testing.assert_close(
                output[list(sample_tokens)], expected, rtol=0, atol=0
            )
            diagnostics = workspace.diagnostics()
            expected_routes, expected_tasks = _expected_local_work(
                rank, active_rows
            )
            assert int(diagnostics["protocol_error"].item()) == 0
            assert int(diagnostics["total_valid_routes"].item()) == expected_routes
            assert int(diagnostics["total_m_tasks"].item()) == expected_tasks
        assert observed_slots == [epoch & 1 for epoch in range(epochs)]
        passed = torch.tensor(1, dtype=torch.int32)
        torch.distributed.all_reduce(
            passed, op=torch.distributed.ReduceOp.MIN, group=control_group
        )
        assert int(passed.item()) == 1
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if workspace is not None and (session is None or session.closed):
                workspace.close()
        torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group(group)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
