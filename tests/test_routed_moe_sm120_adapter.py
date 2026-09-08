import copy
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

import deep_gemm
from deep_gemm.mega import routed_moe_sm120 as adapter


@pytest.fixture
def sm120_layout():
    return {
        'world_size': 8,
        'experts_per_rank': 32,
        'num_topk': 6,
        'hidden': 4096,
        'intermediate_hidden': 2048,
        'max_rows': 8192,
        'pool_rows': 397280,
        'max_tasks': 3104,
        'mailbox_entries': 8192,
        'codec_max_chunks_per_peer': 769,
        'dispatch_chunks': 8,
    }


def test_sm120_adapter_is_publicly_exported():
    assert deep_gemm.SM120RoutedMoESession is adapter.SM120RoutedMoESession
    assert deep_gemm.SM120RoutedMoEWorkspace is adapter.SM120RoutedMoEWorkspace
    assert deep_gemm.fp8_fp4_routed_moe_sm120 is adapter.fp8_fp4_routed_moe_sm120


def test_sm120_workspace_plan_matches_kernel_contract(sm120_layout):
    specs = adapter._workspace_specs(sm120_layout)

    assert specs['w1_warp_done'] == (torch.int32, 3104 * 32)
    assert specs['w2_warp_done'] == (torch.int32, 3104 * 32)
    assert specs['result_chunk_total'] == (torch.int32, 8 * 769)
    assert specs['dispatch_chunk_scatter_counter'] == (torch.uint32, 8 * 8)
    assert specs['pipeline_tile_mailbox'] == (torch.int32, 8192)
    assert specs['route_result_index'] == (torch.int32, 8192 * 6)
    assert 'result_owner_ready' not in specs
    assert 'pull_request_scratch' not in specs
    assert 'task_expert' not in specs
    assert 'total_padded_rows' not in specs


def test_sm120_tensor_validation_is_fail_closed():
    tensor = torch.empty((2, 4), dtype=torch.float32)
    adapter._require_tensor(tensor, 'tensor', tensor.device, torch.float32, (2, 4))

    with pytest.raises(ValueError, match='dtype'):
        adapter._require_tensor(tensor, 'tensor', tensor.device, torch.int32, (2, 4))
    with pytest.raises(ValueError, match='shape'):
        adapter._require_tensor(tensor, 'tensor', tensor.device, torch.float32, (4, 2))
    with pytest.raises(ValueError, match='contiguous'):
        adapter._require_tensor(
            tensor.transpose(0, 1), 'tensor', tensor.device, torch.float32, (4, 2)
        )


def test_sm120_tensor_map_recipes_and_cache(monkeypatch):
    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace.layout = {
        'experts_per_rank': 32,
        'hidden': 4096,
        'intermediate_hidden': 2048,
        'pool_rows': 397280,
        'task_rows': 128,
    }
    workspace._tensor_map_key = None
    workspace._tensor_maps = {}
    workspace._tensor_map_sources = ()
    workspace._pool_fp8 = torch.empty(0, dtype=torch.uint8, device='meta')
    workspace._pool_scales = torch.empty(0, dtype=torch.int32, device='meta')
    workspace._intermediate_fp8 = torch.empty(0, dtype=torch.uint8, device='meta')
    workspace._intermediate_scales = torch.empty(0, dtype=torch.int32, device='meta')
    workspace._w1_output = torch.empty(0, dtype=torch.bfloat16, device='meta')
    workspace._w2_output = torch.empty(0, dtype=torch.bfloat16, device='meta')

    w1 = torch.empty((32, 4096, 2048), dtype=torch.int8, device='meta')
    w1_scales = torch.empty((32, 32, 4096), dtype=torch.int32, device='meta')
    w2 = torch.empty((32, 4096, 1024), dtype=torch.int8, device='meta')
    w2_scales = torch.empty((32, 16, 4096), dtype=torch.int32, device='meta')
    calls = []

    def make_tensor_map(*arguments):
        calls.append(arguments)
        return len(calls)

    monkeypatch.setattr(adapter._C, 'make_sm120_tma_2d', make_tensor_map)
    workspace._prepare_tensor_maps(w1, w1_scales, w2, w2_scales)
    workspace._prepare_tensor_maps(w1, w1_scales, w2, w2_scales)

    assert len(calls) == 10
    assert calls[1][1:] == ('fp4', 4096, 4096 * 32, 2048, 128, 128, 128)
    assert calls[3][1:] == ('int32', 4096, 32 * 32, 4096 * 4, 128, 1, 0)
    assert calls[6][1:] == ('fp4', 2048, 4096 * 32, 1024, 128, 128, 128)
    assert calls[8][1:] == ('int32', 4096, 16 * 32, 4096 * 4, 128, 1, 0)
    assert workspace._tensor_map_sources == (w1, w1_scales, w2, w2_scales)


def test_sm120_typed_launch_builds_private_arguments_and_advances_epoch(monkeypatch):
    layout = {
        'world_size': 8,
        'experts_per_rank': 32,
        'num_topk': 6,
        'hidden': 4096,
        'intermediate_hidden': 2048,
        'max_rows': 8192,
        'max_grid_ctas': 110,
        'activation_clamp': 10.0,
        'fast_math': True,
    }
    session = object.__new__(adapter.SM120RoutedMoESession)
    session.device = torch.device('meta')
    session.rank = 0
    session.world_size = 8
    session._native = SimpleNamespace(closed=False)
    session._launch_lock = threading.RLock()
    session._bound_workspace = None
    session._next_epoch = 0
    session._pending_launch = None

    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace.device = session.device
    workspace.layout = layout
    workspace.enable_phase_trace = False
    workspace._closed = False
    workspace._bound_session = None
    workspace._epoch = 0
    workspace._arguments = {'workspace_marker': object()}
    workspace._tensor_maps = {'W1_A': object()}
    workspace.output = torch.empty((8192, 4096), dtype=torch.bfloat16, device='meta')
    workspace._prepare_tensor_maps = lambda *args: None

    x = torch.empty((1, 4096), dtype=torch.uint8, device='meta')
    x_scales = torch.empty((1, 32), dtype=torch.int32, device='meta')
    topk_indices = torch.empty((1, 6), dtype=torch.int64, device='meta')
    topk_weights = torch.empty((1, 6), dtype=torch.float32, device='meta')
    w1 = torch.empty((32, 4096, 2048), dtype=torch.int8, device='meta')
    w1_scales = torch.empty((32, 32, 4096), dtype=torch.int32, device='meta')
    w2 = torch.empty((32, 4096, 1024), dtype=torch.int8, device='meta')
    w2_scales = torch.empty((32, 16, 4096), dtype=torch.int32, device='meta')

    calls = []
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: None)
    monkeypatch.setattr(adapter._C, 'sm120_fp8_fp4_routed_moe', lambda **kwargs: calls.append(kwargs))

    output = adapter.fp8_fp4_routed_moe_sm120(
        session,
        workspace,
        x,
        x_scales,
        topk_indices,
        topk_weights,
        (w1, w1_scales),
        (w2, w2_scales),
    )
    adapter.fp8_fp4_routed_moe_sm120(
        session,
        workspace,
        x,
        x_scales,
        topk_indices,
        topk_weights,
        (w1, w1_scales),
        (w2, w2_scales),
    )

    assert output.shape == (1, 4096)
    assert [call['epoch'] for call in calls] == [0, 1]
    assert calls[0]['session'] is session._native
    assert calls[0]['arguments']['workspace_marker'] is workspace._arguments['workspace_marker']
    assert calls[0]['arguments']['W1_A'] is workspace._tensor_maps['W1_A']
    assert calls[0]['arguments']['topk_idx_i32'].dtype == torch.int32
    assert calls[0]['arguments']['x_fp8_i32'].dtype == torch.int32
    assert 'rank' not in calls[0]['arguments']
    assert 'world_size' not in calls[0]['arguments']

    assert session._bound_workspace is workspace
    assert workspace._bound_session() is session
    assert session._next_epoch == 2

    other_session = object.__new__(adapter.SM120RoutedMoESession)
    other_session._bound_workspace = None
    other_session._next_epoch = 0
    with pytest.raises(RuntimeError, match='workspace is already bound to a session'):
        other_session._require_workspace(workspace)

    other_workspace = copy.copy(workspace)
    other_workspace._closed = False
    other_workspace._bound_session = None
    other_workspace._epoch = 0
    with pytest.raises(RuntimeError, match='already bound to another workspace'):
        adapter.fp8_fp4_routed_moe_sm120(
            session,
            other_workspace,
            x,
            x_scales,
            topk_indices,
            topk_weights,
            (w1, w1_scales),
            (w2, w2_scales),
        )
    assert len(calls) == 2
    assert other_workspace._epoch == 0

    workspace._epoch = 7
    with pytest.raises(RuntimeError, match='epoch does not match'):
        adapter.fp8_fp4_routed_moe_sm120(
            session,
            workspace,
            x,
            x_scales,
            topk_indices,
            topk_weights,
            (w1, w1_scales),
            (w2, w2_scales),
        )
    workspace._epoch = 2
    pending_launch = session._pending_launch

    def fail_launch(**kwargs):
        raise RuntimeError('launch failed')

    monkeypatch.setattr(adapter._C, 'sm120_fp8_fp4_routed_moe', fail_launch)
    with pytest.raises(RuntimeError, match='launch failed'):
        adapter.fp8_fp4_routed_moe_sm120(
            session,
            workspace,
            x,
            x_scales,
            topk_indices,
            topk_weights,
            (w1, w1_scales),
            (w2, w2_scales),
        )
    assert workspace._epoch == 2
    assert session._pending_launch is pending_launch


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
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device)))
    monkeypatch.setattr(
        adapter.dist,
        "barrier",
        lambda **arguments: calls.append(("barrier", arguments)),
    )

    session.close()

    assert [name for name, _ in calls] == ["quiesce", "sync", "barrier", "close"]
    assert calls[0][1]["active_rows"] == 2048
    assert calls[0][1]["grid_ctas"] == 110
    assert calls[2][1] == {"group": session.group, "device_ids": [0]}
    assert session._pending_launch is None
    assert session._bound_workspace is None
    assert session.closed


def test_sm120_workspace_close_requires_bound_session_to_close(monkeypatch):
    workspace = object.__new__(adapter.SM120RoutedMoEWorkspace)
    workspace._closed = False
    workspace.device = torch.device('cuda', 0)

    class Session:
        closed = False

    session = Session()
    workspace._bound_session = weakref.ref(session)
    monkeypatch.setattr(
        torch.cuda,
        'synchronize',
        lambda device: pytest.fail('workspace synchronized before checking its session'),
    )

    with pytest.raises(RuntimeError, match='close the bound .* session'):
        workspace.close()
    assert not workspace.closed
