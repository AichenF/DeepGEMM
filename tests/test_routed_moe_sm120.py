import pytest
import torch

import deep_gemm
from deep_gemm.mega.routed_moe_sm120 import _workspace_specs
from deep_gemm.testing.routed_moe_sm120 import (
    HIDDEN,
    TOP_K,
    deterministic_route_expert,
    deterministic_weight_exponents,
    make_dsv4_w4a8_inputs,
)
from tests.bench_routed_moe_sm120 import _network_inventory, _sol_receipt


def test_sm120_testing_inputs_match_public_api_contract():
    inputs = make_dsv4_w4a8_inputs(rank=3, active_rows=2, device="cpu")

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


def test_sm120_sol_separates_theoretical_and_provenanced_empirical_roofs():
    rank_receipts = [
        {"total_m_tasks": 64, "application_egress_bytes": 1_000}
        for _ in range(8)
    ]
    layout = {"task_rows": 128}

    incomplete = _sol_receipt(
        rank_receipts,
        layout,
        active_rows=1024,
        observed_ms=2.0,
        empirical_compute_roof_tflops=None,
        empirical_compute_roof_source=None,
        sm_counts=[110] * 8,
        maximum_sm_clocks_mhz=[3090.0] * 8,
        network_port_count=8,
        network_port_gbps=400.0,
    )
    assert incomplete["status"] == "theoretical_only"
    assert incomplete["compute"]["physical_theoretical_floor_ms"] > 0
    assert incomplete["empirical_attainable_floor_ms"] is None

    complete = _sol_receipt(
        rank_receipts,
        layout,
        active_rows=1024,
        observed_ms=2.0,
        empirical_compute_roof_tflops=500.0,
        empirical_compute_roof_source="sha256:test-roof-receipt",
        sm_counts=[110] * 8,
        maximum_sm_clocks_mhz=[3090.0] * 8,
        network_port_count=8,
        network_port_gbps=400.0,
    )
    assert complete["status"] == "complete"
    assert complete["compute"]["empirical_floor_ms"] > 0
    assert complete["empirical_attainable_floor_ms"] >= complete["compute"]["empirical_floor_ms"]


def test_sm120_network_roof_requires_matching_active_hca_inventory(
    tmp_path, monkeypatch
):
    for hca in ("mlx5_0", "mlx5_1"):
        port = tmp_path / hca / "ports" / "1"
        port.mkdir(parents=True)
        (port / "rate").write_text("400 Gb/sec (4X NDR)\n")
        (port / "state").write_text("4: ACTIVE\n")
        (port / "phys_state").write_text("5: LinkUp\n")
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_0,mlx5_1")

    accepted = _network_inventory(2, 400.0, tmp_path)
    rejected = _network_inventory(2, 200.0, tmp_path)

    assert accepted["accepted"]
    assert not rejected["accepted"]


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
