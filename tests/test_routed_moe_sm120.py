"""SM120 routed-MoE contract, adapter, and opt-in EP8 correctness tests.

Run the distributed case on eight SM120 GPUs with::

    RUN_SM120_ROUTED_MOE_E2E=1 torchrun --nproc-per-node=8 \
        -m pytest -q tests/test_routed_moe_sm120.py
"""

from __future__ import annotations

import copy
import inspect
import os
import threading
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import deep_gemm
from deep_gemm.mega import routed_moe_sm120 as adapter
from deep_gemm.mega.routed_moe_sm120 import _workspace_specs
from routed_moe_sm120_utils import (
    ACTIVATION_CLAMP,
    CONTRACT,
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    MAX_ROWS,
    TOP_K,
    WORLD_SIZE,
    ModelContract,
    _requantize_k32 as requantize_k32,
    deterministic_route_expert,
    deterministic_weight_exponents,
    epoch_slots,
    expected_local_work,
    expected_route,
    expected_tokens,
    make_dsv4_w4a8_inputs,
    make_dsv4_w4a8_weights,
    source_order_combine,
)


def test_sm120_public_exports_and_dsv4_w4a8_contract():
    assert deep_gemm.SM120RoutedMoESession is adapter.SM120RoutedMoESession
    assert deep_gemm.SM120RoutedMoEWorkspace is adapter.SM120RoutedMoEWorkspace
    assert deep_gemm.fp8_fp4_routed_moe_sm120 is adapter.fp8_fp4_routed_moe_sm120
    assert CONTRACT == ModelContract()
    assert CONTRACT.w1 == "gate+up"
    assert CONTRACT.w2 == "down"
    assert CONTRACT.activation == "MXFP8 E4M3, K32"
    assert CONTRACT.weight == "MXFP4 E2M1, K32"


def test_sm120_structured_fixture_matches_public_api_layout():
    inputs = make_dsv4_w4a8_inputs(rank=3, active_rows=2, device="cpu")
    weights = make_dsv4_w4a8_weights(rank=3, device="meta")

    assert inputs.x.shape == (2, HIDDEN)
    assert inputs.x.dtype == torch.uint8
    assert inputs.x_scales.shape == (2, HIDDEN // 128)
    assert inputs.x_scales.dtype == torch.int32
    assert inputs.topk_indices.shape == (2, TOP_K)
    assert inputs.topk_indices.dtype == torch.int64
    assert inputs.topk_weights.shape == (2, TOP_K)
    assert inputs.topk_weights.dtype == torch.float32
    assert inputs.topk_indices[1].tolist() == [
        deterministic_route_expert(3, 1, route_slot)
        for route_slot in range(TOP_K)
    ]
    assert weights.w1_weight.shape == (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2)
    assert weights.w1_scales.shape == (LOCAL_EXPERTS, HIDDEN // 128, 2 * INTERMEDIATE)
    assert weights.w2_weight.shape == (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2)
    assert weights.w2_scales.shape == (LOCAL_EXPERTS, INTERMEDIATE // 128, HIDDEN)
    assert weights.w1_weight.dtype == weights.w2_weight.dtype == torch.int8
    assert weights.w1_scales.dtype == weights.w2_scales.dtype == torch.int32


def test_sm120_testing_weight_recipe_rejects_unknown_projection():
    with pytest.raises(ValueError, match="unknown projection"):
        deterministic_weight_exponents(0, "side", "cpu")


def test_sm120_workspace_diagnostics_are_cloned():
    workspace = object.__new__(deep_gemm.SM120RoutedMoEWorkspace)
    workspace._closed = False
    names = (
        "protocol_error",
        "owner_record_counts",
        "source_route_counts",
        "result_ovf_cursor",
        "total_valid_routes",
        "total_m_tasks",
    )
    workspace._arguments = {
        name: torch.tensor([index], dtype=torch.int32)
        for index, name in enumerate(names)
    }

    diagnostics = workspace.diagnostics()

    assert diagnostics.keys() == workspace._arguments.keys()
    for name in names:
        assert diagnostics[name] is not workspace._arguments[name]
        assert torch.equal(diagnostics[name], workspace._arguments[name])


def test_sm120_layout_contract():
    layout = dict(deep_gemm._C.get_sm120_routed_moe_layout())

    assert layout == {
        "world_size": 8,
        "num_experts": 256,
        "experts_per_rank": 32,
        "num_topk": 6,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "max_rows": 8192,
        "task_rows": 128,
        "max_grid_ctas": 110,
        "activation_clamp": 10.0,
        "fast_math": True,
        "pool_rows": 397280,
        "max_tasks": 3104,
        "mailbox_entries": 8192,
        "phase_timestamp_count": 33,
        "codec_blocks_per_row": 64,
        "codec_encoded_row_bytes": 6272,
        "codec_raw_blocks_per_row": 64,
        "codec_raw_values_in_body": 48,
        "codec_raw_tail_bytes": 32,
        "codec_max_chunks_per_peer": 769,
        "result_route_pitch_bytes": 8320,
        "result_slot_bytes": 409141248,
        "dispatch_header_window_bytes": 4608,
        "dispatch_header_slot_bytes": 288,
        "dispatch_record_bytes": 4352,
        "dispatch_payload_window_bytes": 570425344,
        "result_window_bytes": 6546259968,
        "ack_window_bytes": 8,
        "gin_context_count": 2,
        "dispatch_chunks": 8,
        "gin_signal_count": 88,
        "world_barrier_count": 1,
    }


def test_sm120_workspace_contract_uses_semantic_pipeline_names():
    specs = _workspace_specs(dict(deep_gemm._C.get_sm120_routed_moe_layout()))

    assert {"pipeline_claim_cursor", "pipeline_tile_mailbox"} <= specs.keys()
    assert {
        "c56_claim_cursor",
        "c56_tile_mailbox",
        "result_owner_ready",
        "pull_request_scratch",
        "meta_token",
        "meta_slot",
        "expert_scatter_offsets",
        "task_max_source",
        "task_expert",
        "task_source_rank",
        "task_owner_rank",
        "task_m_local",
        "task_valid_m",
        "total_padded_rows",
    }.isdisjoint(specs)


def test_sm120_tensor_validation_is_fail_closed():
    tensor = torch.empty((2, 4), dtype=torch.float32)
    adapter._require_tensor(tensor, "tensor", tensor.device, torch.float32, (2, 4))

    with pytest.raises(ValueError, match="dtype"):
        adapter._require_tensor(tensor, "tensor", tensor.device, torch.int32, (2, 4))
    with pytest.raises(ValueError, match="shape"):
        adapter._require_tensor(tensor, "tensor", tensor.device, torch.float32, (4, 2))
    with pytest.raises(ValueError, match="contiguous"):
        adapter._require_tensor(
            tensor.transpose(0, 1),
            "tensor",
            tensor.device,
            torch.float32,
            (4, 2),
        )


def test_sm120_tensor_map_recipes_are_cached(monkeypatch):
    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace.layout = {
        "experts_per_rank": LOCAL_EXPERTS,
        "hidden": HIDDEN,
        "intermediate_hidden": INTERMEDIATE,
        "pool_rows": 397280,
        "task_rows": 128,
    }
    workspace._tensor_map_key = None
    workspace._tensor_maps = {}
    workspace._tensor_map_sources = ()
    workspace._pool_fp8 = torch.empty(0, dtype=torch.uint8, device="meta")
    workspace._pool_scales = torch.empty(0, dtype=torch.int32, device="meta")
    workspace._intermediate_fp8 = torch.empty(0, dtype=torch.uint8, device="meta")
    workspace._intermediate_scales = torch.empty(0, dtype=torch.int32, device="meta")
    workspace._w1_output = torch.empty(0, dtype=torch.bfloat16, device="meta")
    workspace._w2_output = torch.empty(0, dtype=torch.bfloat16, device="meta")
    weights = make_dsv4_w4a8_weights(rank=0, device="meta")
    calls = []

    def make_tensor_map(*arguments):
        calls.append(arguments)
        return len(calls)

    monkeypatch.setattr(adapter._C, "make_sm120_tma_2d", make_tensor_map)
    workspace._prepare_tensor_maps(
        weights.w1_weight,
        weights.w1_scales,
        weights.w2_weight,
        weights.w2_scales,
    )
    workspace._prepare_tensor_maps(
        weights.w1_weight,
        weights.w1_scales,
        weights.w2_weight,
        weights.w2_scales,
    )

    assert len(calls) == 10
    assert calls[1][1:] == ("fp4", 4096, 4096 * 32, 2048, 128, 128, 128)
    assert calls[3][1:] == ("int32", 4096, 32 * 32, 4096 * 4, 128, 1, 0)
    assert calls[6][1:] == ("fp4", 2048, 4096 * 32, 1024, 128, 128, 128)
    assert calls[8][1:] == ("int32", 4096, 16 * 32, 4096 * 4, 128, 1, 0)
    assert workspace._tensor_map_sources == (
        weights.w1_weight,
        weights.w1_scales,
        weights.w2_weight,
        weights.w2_scales,
    )


def test_sm120_fast_path_contract():
    can_use = deep_gemm._C.can_use_sm120_routed_moe_fast_path
    target = {
        "num_ranks": 8,
        "num_experts": 256,
        "num_topk": 6,
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_external_shared_experts": 1,
        "num_tokens": 2048,
        "activation_clamp": 10.0,
        "fast_math": True,
    }

    if not deep_gemm._C.has_sm120_routed_moe():
        assert not can_use(**target)
        return
    assert can_use(**target)
    for field, unsupported in (
        ("num_ranks", 4),
        ("num_experts", 128),
        ("num_topk", 8),
        ("hidden", 7168),
        ("intermediate_hidden", 4096),
        ("num_external_shared_experts", 0),
        ("num_tokens", 8193),
        ("activation_clamp", 7.0),
        ("fast_math", False),
    ):
        candidate = {**target, field: unsupported}
        assert not can_use(**candidate)


@pytest.mark.skipif(
    not deep_gemm._C.has_sm120_routed_moe(),
    reason="DeepGEMM was built without NCCL GIN support",
)
def test_sm120_launcher_rejects_legacy_raw_resource_interface():
    launch = deep_gemm._C.sm120_fp8_fp4_routed_moe
    with pytest.raises(TypeError):
        launch(
            {},
            rank=0,
            world_size=8,
            active_rows=1,
            epoch=0,
            grid_ctas=8,
            activation_clamp=10.0,
            fast_math=True,
        )


def _tma_binding():
    binding = getattr(deep_gemm._C, "make_sm120_tma_2d", None)
    if binding is None:
        pytest.skip("DeepGEMM was built without NCCL GIN support")
    return binding


def _make_tma(tensor, **overrides):
    arguments = {
        "data_type": "uint8",
        "inner": 256,
        "outer": 1,
        "outer_stride_bytes": 256,
        "box_inner": 32,
        "box_outer": 1,
        "swizzle_bytes": 32,
    }
    arguments.update(overrides)
    return _tma_binding()(tensor, **arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"inner": 0}, "dimensions"),
        ({"box_inner": 0}, "box dimensions"),
        ({"box_inner": 257}, "box dimensions"),
        ({"box_inner": 32, "inner": 16}, "exceed tensor dimensions"),
        ({"outer_stride_bytes": 15}, "outer stride"),
        ({"outer_stride_bytes": 128}, "smaller than one row"),
        ({"box_inner": 8, "swizzle_bytes": 0}, "multiple of 16"),
        ({"box_inner": 64, "swizzle_bytes": 32}, "exceeds swizzle width"),
    ),
)
def test_sm120_tma_rejects_invalid_geometry_without_cuda(overrides, message):
    tensor = torch.empty(256, dtype=torch.uint8)
    with pytest.raises(ValueError, match=message):
        _make_tma(tensor, **overrides)


def test_sm120_tma_rejects_dtype_mismatch_without_cuda():
    tensor = torch.empty(256, dtype=torch.int32)
    with pytest.raises(ValueError, match="does not match tensor dtype"):
        _make_tma(tensor)


def test_sm120_tma_rejects_extent_beyond_tensor_without_cuda():
    tensor = torch.empty(255, dtype=torch.uint8)
    with pytest.raises(ValueError, match="extent exceeds tensor storage"):
        _make_tma(tensor)


def test_sm120_tma_rejects_noncontiguous_tensor_without_cuda():
    tensor = torch.empty((16, 16), dtype=torch.uint8).transpose(0, 1)
    with pytest.raises(ValueError, match="must be contiguous"):
        _make_tma(tensor)


def test_sm120_tma_rejects_misaligned_base_without_cuda():
    tensor = torch.empty(257, dtype=torch.uint8)[1:]
    with pytest.raises(ValueError, match="insufficiently aligned"):
        _make_tma(tensor)


def test_sm120_tma_rejects_invalid_fp4_contract_without_cuda():
    tensor = torch.empty(128, dtype=torch.int8)
    with pytest.raises(ValueError, match="FP4 tensor maps require"):
        _make_tma(
            tensor,
            data_type="fp4",
            inner=64,
            outer_stride_bytes=32,
            box_inner=64,
            swizzle_bytes=0,
        )


def test_sm120_tma_rejects_fp4_extent_beyond_tensor_without_cuda():
    tensor = torch.empty(63, dtype=torch.int8)
    with pytest.raises(ValueError, match="extent exceeds tensor storage"):
        _make_tma(
            tensor,
            data_type="fp4",
            inner=128,
            outer_stride_bytes=64,
            box_inner=128,
            swizzle_bytes=128,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_sm120_tma_builds_valid_cuda_carrier():
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("TMA requires compute capability 9.0 or newer")
    tensor = torch.empty(256, dtype=torch.uint8, device="cuda")
    carrier = _make_tma(tensor)
    assert carrier.dtype == torch.uint8
    assert carrier.device == tensor.device
    assert carrier.numel() == 128


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_sm120_tma_restores_current_device():
    target_device = 1
    major, _ = torch.cuda.get_device_capability(target_device)
    if major < 9:
        pytest.skip("TMA requires compute capability 9.0 or newer")
    torch.cuda.set_device(0)
    tensor = torch.empty(256, dtype=torch.uint8, device=target_device)
    carrier = _make_tma(tensor)
    assert carrier.device.index == target_device
    assert torch.cuda.current_device() == 0


def _mock_public_launch_state():
    session = object.__new__(adapter.SM120RoutedMoESession)
    session.device = torch.device("meta")
    session.rank = 0
    session.world_size = WORLD_SIZE
    session._native = SimpleNamespace(closed=False)
    session._launch_lock = threading.RLock()
    session._bound_workspace = None
    session._next_epoch = 0
    session._pending_launch = None

    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace.device = session.device
    workspace.layout = {
        "world_size": WORLD_SIZE,
        "experts_per_rank": LOCAL_EXPERTS,
        "num_topk": TOP_K,
        "hidden": HIDDEN,
        "intermediate_hidden": INTERMEDIATE,
        "max_rows": MAX_ROWS,
        "max_grid_ctas": 110,
        "activation_clamp": ACTIVATION_CLAMP,
        "fast_math": True,
    }
    workspace.enable_phase_trace = False
    workspace._closed = False
    workspace._bound_session = None
    workspace._epoch = 0
    workspace._arguments = {"workspace_marker": object()}
    workspace._tensor_maps = {"W1_A": object()}
    workspace.output = torch.empty((MAX_ROWS, HIDDEN), dtype=torch.bfloat16, device="meta")
    workspace._prepare_tensor_maps = lambda *args: None

    inputs = (
        torch.empty((1, HIDDEN), dtype=torch.uint8, device="meta"),
        torch.empty((1, HIDDEN // 128), dtype=torch.int32, device="meta"),
        torch.empty((1, TOP_K), dtype=torch.int64, device="meta"),
        torch.empty((1, TOP_K), dtype=torch.float32, device="meta"),
    )
    weights = (
        (
            torch.empty(
                (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
                dtype=torch.int8,
                device="meta",
            ),
            torch.empty(
                (LOCAL_EXPERTS, HIDDEN // 128, 2 * INTERMEDIATE),
                dtype=torch.int32,
                device="meta",
            ),
        ),
        (
            torch.empty(
                (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2),
                dtype=torch.int8,
                device="meta",
            ),
            torch.empty(
                (LOCAL_EXPERTS, INTERMEDIATE // 128, HIDDEN),
                dtype=torch.int32,
                device="meta",
            ),
        ),
    )
    return session, workspace, inputs, weights


def _launch_mock(session, workspace, inputs, weights):
    return deep_gemm.fp8_fp4_routed_moe_sm120(
        session,
        workspace,
        *inputs,
        *weights,
    )


def test_sm120_public_launch_binds_one_workspace_and_advances_epoch(monkeypatch):
    session, workspace, inputs, weights = _mock_public_launch_state()
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)
    monkeypatch.setattr(
        adapter._C,
        "sm120_fp8_fp4_routed_moe",
        lambda **kwargs: calls.append(kwargs),
    )

    output = _launch_mock(session, workspace, inputs, weights)
    _launch_mock(session, workspace, inputs, weights)

    assert output.shape == (1, HIDDEN)
    assert [call["epoch"] for call in calls] == [0, 1]
    assert calls[0]["session"] is session._native
    assert calls[0]["arguments"]["workspace_marker"] is workspace._arguments[
        "workspace_marker"
    ]
    assert calls[0]["arguments"]["W1_A"] is workspace._tensor_maps["W1_A"]
    assert calls[0]["arguments"]["topk_idx_i32"].dtype == torch.int32
    assert calls[0]["arguments"]["x_fp8_i32"].dtype == torch.int32
    assert "rank" not in calls[0]["arguments"]
    assert "world_size" not in calls[0]["arguments"]
    assert session._bound_workspace is workspace
    assert workspace._bound_session() is session
    assert session._next_epoch == workspace._epoch == 2

    other_workspace = copy.copy(workspace)
    other_workspace._closed = False
    other_workspace._bound_session = None
    other_workspace._epoch = 0
    with pytest.raises(RuntimeError, match="already bound to another workspace"):
        _launch_mock(session, other_workspace, inputs, weights)
    assert len(calls) == 2
    assert other_workspace._epoch == 0

    other_session = object.__new__(adapter.SM120RoutedMoESession)
    other_session._bound_workspace = None
    other_session._next_epoch = 0
    with pytest.raises(RuntimeError, match="workspace is already bound to a session"):
        other_session._require_workspace(workspace)


def test_sm120_public_launch_is_fail_closed(monkeypatch):
    session, workspace, inputs, weights = _mock_public_launch_state()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)
    monkeypatch.setattr(adapter._C, "sm120_fp8_fp4_routed_moe", lambda **kwargs: None)
    _launch_mock(session, workspace, inputs, weights)
    pending_launch = session._pending_launch

    workspace._epoch = 7
    with pytest.raises(RuntimeError, match="epoch does not match"):
        _launch_mock(session, workspace, inputs, weights)
    workspace._epoch = 1

    def fail_launch(**kwargs):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(adapter._C, "sm120_fp8_fp4_routed_moe", fail_launch)
    with pytest.raises(RuntimeError, match="launch failed"):
        _launch_mock(session, workspace, inputs, weights)
    assert workspace._epoch == 1
    assert session._pending_launch is pending_launch


def test_sm120_public_api_rejects_invalid_types_and_workspace_close(monkeypatch):
    session, workspace, inputs, weights = _mock_public_launch_state()
    with pytest.raises(TypeError, match="session"):
        _launch_mock(object(), workspace, inputs, weights)
    with pytest.raises(TypeError, match="workspace"):
        _launch_mock(session, object(), inputs, weights)

    workspace._bound_session = weakref.ref(session)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: pytest.fail("workspace synchronized before checking its session"),
    )
    with pytest.raises(RuntimeError, match="close the bound .* session"):
        workspace.close()
    assert not workspace.closed


def test_sm120_session_close_drains_the_last_epoch(monkeypatch):
    calls = []

    class NativeSession:
        closed = False

        def quiesce(self, **arguments):
            calls.append(("quiesce", arguments))

        def close(self):
            calls.append(("close", None))
            self.closed = True

    session = object.__new__(adapter.SM120RoutedMoESession)
    session.device = torch.device("cuda", 0)
    session.group = object()
    session._native = NativeSession()
    session._launch_lock = threading.RLock()
    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace._epoch = 1
    session._bound_workspace = workspace
    session._next_epoch = 1
    session._pending_launch = (workspace, {"workspace": object()}, 2048, 110, 0)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: calls.append(("sync", device)),
    )
    monkeypatch.setattr(
        adapter.dist,
        "barrier",
        lambda **arguments: calls.append(("barrier", arguments)),
    )

    session.close()

    assert [name for name, _ in calls] == ["quiesce", "sync", "barrier", "close"]
    assert calls[0][1]["active_rows"] == 2048
    assert calls[0][1]["grid_ctas"] == 110
    assert session._pending_launch is None
    assert session._bound_workspace is None
    assert session.closed


def test_sm120_oracle_preserves_precision_boundaries_and_source_order():
    zero = torch.zeros((1, INTERMEDIATE), dtype=torch.float32)
    fp8, exponent, dequantized = requantize_k32(zero)
    assert not fp8.any()
    assert not exponent.any()
    assert not dequantized.any()

    canonical = expected_route(0, 1, 0, 17, "cpu")
    rounded = expected_route(
        0,
        1,
        0,
        17,
        "cpu",
        round_swiglu_output=True,
    )
    assert not torch.equal(canonical["intermediate_fp8"], rounded["intermediate_fp8"])
    assert not torch.equal(canonical["w2_bf16"], rounded["w2_bf16"])

    partials = torch.zeros((1, TOP_K, 1), dtype=torch.bfloat16)
    partials[0, :, 0] = torch.tensor(
        [1.0e20, -1.0e20, 1.0, 0.5, 0.25, 0.125],
        dtype=torch.bfloat16,
    )
    forward = source_order_combine(partials)
    reverse = torch.zeros((1, 1), dtype=torch.float32)
    for slot in reversed(range(TOP_K)):
        reverse += partials[:, slot].float()
    assert not torch.equal(forward, reverse.to(torch.bfloat16))
    assert epoch_slots([0, 1, 2]) == [0, 1, 0]

    source = inspect.getsource(expected_route).lower()
    assert "w1_d" not in source
    assert "w2_d" not in source


def _distributed_backend() -> tuple[Any, Any, int]:
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


@pytest.mark.skipif(
    os.environ.get("RUN_SM120_ROUTED_MOE_E2E") != "1",
    reason="set RUN_SM120_ROUTED_MOE_E2E=1 for the full eight-rank allocation",
)
def test_sm120_public_api_ep8_bit_exact_multi_epoch():
    group, control_group, rank = _distributed_backend()
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
    expected = expected_tokens(rank, sample_tokens, device)
    session = None
    workspace = None
    try:
        torch.distributed.barrier(group=control_group)
        session = deep_gemm.SM120RoutedMoESession(group, device)
        properties = session.properties
        assert properties["rank"] == rank
        assert properties["world_size"] == WORLD_SIZE
        assert properties["nccl_version"] == 23007
        assert properties["gin_connection_count"] >= 1
        assert len(properties["gin_net_device_types"]) == properties[
            "gin_connection_count"
        ]
        assert properties["window_count"] == 8
        layout = dict(deep_gemm._C.get_sm120_routed_moe_layout())
        for name, size_name in (
            ("dispatch_header_out", "dispatch_header_window_bytes"),
            ("dispatch_header_inbox", "dispatch_header_window_bytes"),
            ("dispatch_payload_out", "dispatch_payload_window_bytes"),
            ("dispatch_payload_inbox", "dispatch_payload_window_bytes"),
            ("result_out", "result_window_bytes"),
            ("result_inbox", "result_window_bytes"),
            ("ack_out", "ack_window_bytes"),
            ("ack_inbox", "ack_window_bytes"),
        ):
            assert session._native.window_tensor(name).numel() == layout[size_name]
            assert session._native.window_handle(name) != 0
        assert session._native.device_communicator() != 0
        torch.distributed.barrier(group=control_group)
        workspace = deep_gemm.SM120RoutedMoEWorkspace(device)

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
            torch.testing.assert_close(
                output[list(sample_tokens)],
                expected,
                rtol=0,
                atol=0,
            )
            diagnostics = workspace.diagnostics()
            expected_routes, expected_tasks = expected_local_work(rank, active_rows)
            assert int(diagnostics["protocol_error"].item()) == 0
            assert int(diagnostics["total_valid_routes"].item()) == expected_routes
            assert int(diagnostics["total_m_tasks"].item()) == expected_tasks
            assert workspace._epoch == epoch + 1

        passed = torch.tensor(1, dtype=torch.int32)
        torch.distributed.all_reduce(
            passed,
            op=torch.distributed.ReduceOp.MIN,
            group=control_group,
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
