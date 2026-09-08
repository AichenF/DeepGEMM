import os

import pytest
import torch

import deep_gemm


def _initialized_process_group():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    if not torch.distributed.is_initialized():
        if int(os.environ.get("WORLD_SIZE", "1")) != 8:
            pytest.skip("run this test with torchrun --nproc-per-node=8")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    group = torch.distributed.group.WORLD
    control_group = torch.distributed.new_group(backend="gloo")
    backend = group._get_backend(torch.device("cuda"))
    return group, control_group, backend


def test_sm120_routed_moe_session():
    if not deep_gemm._C.has_sm120_routed_moe():
        pytest.skip("DeepGEMM was built without NCCL GIN support")

    group, control_group, backend = _initialized_process_group()
    rank = torch.distributed.get_rank(group)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.barrier(group=group, device_ids=[local_rank])
    torch.distributed.barrier(group=control_group)
    session = deep_gemm._C.SM120RoutedMoESession(
        int(backend._comm_ptr()), backend, rank, 8, local_rank
    )
    torch.distributed.barrier(group=control_group)
    properties = dict(session.properties())
    layout = dict(deep_gemm._C.get_sm120_routed_moe_layout())

    assert properties["rank"] == rank
    assert properties["world_size"] == 8
    assert properties["gin_connection_count"] >= 1
    assert len(properties["gin_net_device_types"]) == properties["gin_connection_count"]
    assert properties["window_count"] == 8
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
        assert session.window_tensor(name).numel() == layout[size_name]
        assert session.window_handle(name) != 0
    assert session.device_communicator() != 0

    launch = deep_gemm._C.sm120_fp8_fp4_routed_moe
    launch_parameters = {
        "session": session,
        "active_rows": 1,
        "epoch": 0,
        "grid_ctas": 8,
        "activation_clamp": 10.0,
        "fast_math": True,
    }
    for name in (
        "rank",
        "world_size",
        "device",
        "gin_device_communicator",
        "dispatch_header_out",
        "dispatch_header_out_window",
        "dispatch_payload_out",
        "dispatch_payload_out_window",
        "dispatch_header_inbox",
        "dispatch_header_inbox_window",
        "dispatch_payload_inbox",
        "dispatch_payload_inbox_window",
        "result_out",
        "result_out_window",
        "result_inbox",
        "result_inbox_window",
        "ack_out_window",
        "ack_inbox_window",
    ):
        with pytest.raises(ValueError, match=f"owned by the session: {name}"):
            launch(arguments={name: 0}, **launch_parameters)

    if torch.cuda.device_count() > 1:
        mismatched_device = (local_rank + 1) % torch.cuda.device_count()
        try:
            torch.cuda.set_device(mismatched_device)
            with pytest.raises(RuntimeError, match="current CUDA device does not match"):
                launch(arguments={}, **launch_parameters)
        finally:
            torch.cuda.set_device(local_rank)

    torch.distributed.barrier(group=control_group)
    session.close()
    torch.distributed.barrier(group=control_group)
    assert session.closed
    with pytest.raises(RuntimeError, match="GIN session is closed"):
        launch(arguments={}, **launch_parameters)
    torch.distributed.destroy_process_group(control_group)
    torch.distributed.destroy_process_group(group)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
