import threading
import weakref

import torch
import torch.distributed as dist

from .. import _C

TensorPair = tuple[torch.Tensor, torch.Tensor]
DeviceLike = int | str | torch.device


def _layout() -> dict[str, int | float | bool]:
    return dict(_C.get_sm120_routed_moe_layout())


def _cuda_device(device: DeviceLike | None) -> torch.device:
    if device is None:
        return torch.device('cuda', torch.cuda.current_device())
    if isinstance(device, int):
        result = torch.device('cuda', device)
    else:
        result = torch.device(device)
    if result.type != 'cuda':
        raise ValueError('SM120 routed MoE requires a CUDA device')
    return torch.device('cuda', torch.cuda.current_device() if result.index is None else result.index)


def _require_available() -> None:
    if not getattr(_C, 'has_sm120_routed_moe', lambda: False)():
        raise RuntimeError(
            'SM120 routed MoE support is not built; rebuild with '
            'DG_WITH_NCCL_GIN=1 and DG_NCCL_ROOT set'
        )


def _require_tensor(
    tensor: torch.Tensor,
    name: str,
    device: torch.device,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    expected_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if tensor.dtype not in expected_dtypes:
        names = ', '.join(str(value) for value in expected_dtypes)
        raise ValueError(f'{name} must have dtype {names}, got {tensor.dtype}')
    if tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous')
    if tuple(tensor.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(tensor.shape)}')


def _workspace_specs(layout: dict[str, int | float | bool]):
    world_size = int(layout['world_size'])
    experts_per_rank = int(layout['experts_per_rank'])
    topk = int(layout['num_topk'])
    hidden = int(layout['hidden'])
    intermediate = int(layout['intermediate_hidden'])
    max_rows = int(layout['max_rows'])
    pool_rows = int(layout['pool_rows'])
    max_tasks = int(layout['max_tasks'])
    max_chunks = world_size * int(layout['codec_max_chunks_per_peer'])
    chunk_entries = world_size * int(layout['dispatch_chunks'])
    w1_tiles = 2 * intermediate // 128
    w2_tiles = hidden // 128

    return {
        'requant_groups_done': (torch.int32, 1),
        'w2_warp_done': (torch.int32, max_tasks * w2_tiles),
        'w2_tiles_completed': (torch.int32, 1),
        'owner_record_counts': (torch.int32, world_size),
        'owner_route_counts': (torch.int32, world_size),
        'owner_minexp_record_base': (torch.int32, world_size * experts_per_rank),
        'owner_minexp_record_cursor': (torch.int32, world_size * experts_per_rank),
        'sorted_record_token': (torch.int32, world_size * max_rows),
        'sorted_record_route_base': (torch.int32, world_size * max_rows),
        'route_result_index': (torch.int32, max_rows * topk),
        'protocol_error': (torch.int32, 1),
        'w2_task_counter': (torch.uint32, max_tasks),
        'w1_task_counter': (torch.uint32, max_tasks),
        'dispatch_chunk_scatter_counter': (torch.uint32, chunk_entries),
        'pull_chunk_arrived': (torch.uint32, chunk_entries),
        'result_owner_progress': (torch.uint32, world_size),
        'dispatch_chunk_targets': (torch.int32, chunk_entries),
        'pipeline_claim_cursor': (torch.int32, 1),
        'combine_claim_cursor': (torch.int32, 1),
        'pipeline_tile_mailbox': (torch.int32, int(layout['mailbox_entries'])),
        'task_gate_packed': (torch.int32, max_tasks),
        'result_chunk_total': (torch.int32, max_chunks),
        'result_chunk_tally': (torch.int32, max_chunks),
        'result_ovf_cursor': (torch.int32, max_chunks),
        'signal_base_scratch': (torch.uint64, world_size),
        'dispatch_chunk_signal_base_scratch': (torch.uint64, chunk_entries),
        'result_signal_base_scratch': (torch.uint64, world_size),
        'ack_signal_base_scratch': (torch.uint64, world_size + 1),
        'routing_weight_pool': (torch.float32, pool_rows),
        'meta_source_rank': (torch.int32, pool_rows),
        'meta_result_index': (torch.int32, pool_rows),
        'expert_counts': (torch.int32, experts_per_rank),
        'owner_expert_route_counts': (torch.int32, world_size * experts_per_rank),
        'source_route_sum': (torch.int32, world_size),
        'source_expert_counts': (torch.int32, world_size * experts_per_rank),
        'expert_source_base': (torch.int32, world_size * experts_per_rank),
        'expert_source_offsets': (torch.int32, world_size * experts_per_rank),
        'source_expert_prefix': (torch.int32, world_size * experts_per_rank),
        'source_record_counts': (torch.int32, world_size),
        'source_route_counts': (torch.int32, world_size),
        'source_active_rows': (torch.int32, world_size),
        'expert_row_offsets': (torch.int32, experts_per_rank),
        'task_local_expert': (torch.int32, max_tasks),
        'task_pool_row': (torch.int32, max_tasks),
        'total_valid_routes': (torch.int32, 1),
        'total_m_tasks': (torch.int32, 1),
        'histogram_done': (torch.int32, 1),
        'prefix_done': (torch.int32, 1),
        'w1_warp_done': (torch.int32, max_tasks * w1_tiles),
        'w1_tiles_completed': (torch.int32, 1),
    }


class SM120RoutedMoESession:
    """Owns the NCCL GIN communicator resources for one eight-rank EP group."""

    def __init__(self, group: dist.ProcessGroup, device: DeviceLike | None = None):
        _require_available()
        self.group = group
        self.device = _cuda_device(device)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size != int(_layout()['world_size']):
            raise ValueError('SM120 routed MoE requires an eight-rank EP group')
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError('current CUDA device must match the SM120 routed MoE session')

        backend = group._get_backend(self.device)
        if not hasattr(backend, '_comm_ptr'):
            raise RuntimeError('the NCCL process-group backend does not expose its communicator')
        self._backend = backend
        self._launch_lock = threading.RLock()
        self._bound_workspace = None
        self._next_epoch = 0
        self._pending_launch = None
        self._prepared_trace_modes: set[bool] = set()
        self._native = _C.SM120RoutedMoESession(
            int(backend._comm_ptr()), backend, self.rank, self.world_size, self.device.index
        )

    @property
    def closed(self) -> bool:
        return self._native.closed

    @property
    def properties(self) -> dict[str, int]:
        return dict(self._native.properties())

    def close(self) -> None:
        with self._launch_lock:
            if self._native.closed:
                self._pending_launch = None
                self._bound_workspace = None
                return
            if self._pending_launch is not None:
                workspace, arguments, active_rows, grid_ctas, epoch = self._pending_launch
                if (
                    workspace is not self._bound_workspace
                    or ((epoch + 1) & 0xFFFFFFFF) != self._next_epoch
                    or workspace._epoch != self._next_epoch
                ):
                    raise RuntimeError('SM120 routed MoE session launch state is inconsistent')
                self._native.quiesce(
                    arguments=arguments,
                    active_rows=active_rows,
                    grid_ctas=grid_ctas,
                )
                torch.cuda.synchronize(self.device)
                self._pending_launch = None
            dist.barrier(group=self.group, device_ids=[self.device.index])
            self._native.close()
            self._bound_workspace = None

    def _require_workspace(self, workspace: 'SM120RoutedMoEWorkspace') -> None:
        if self._bound_workspace is not None and self._bound_workspace is not workspace:
            raise RuntimeError('SM120 routed MoE session is already bound to another workspace')
        if workspace._bound_session is not None and workspace._bound_session() is not self:
            raise RuntimeError('SM120 routed MoE workspace is already bound to a session')
        if self._bound_workspace is workspace and workspace._epoch != self._next_epoch:
            raise RuntimeError('SM120 routed MoE workspace epoch does not match its session')

    def _bind_workspace(self, workspace: 'SM120RoutedMoEWorkspace') -> None:
        self._require_workspace(workspace)
        if self._bound_workspace is None:
            if workspace._epoch != 0:
                raise RuntimeError('SM120 routed MoE workspace has an invalid initial epoch')
            self._bound_workspace = workspace
            workspace._bound_session = weakref.ref(self)
            self._next_epoch = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SM120RoutedMoEWorkspace:
    """Reusable fixed-shape workspace and tensor-map cache for the SM120 kernel."""

    def __init__(
        self,
        device: DeviceLike | None = None,
        enable_phase_trace: bool = False,
    ):
        _require_available()
        self.device = _cuda_device(device)
        self.layout = _layout()
        self.enable_phase_trace = bool(enable_phase_trace)
        self._closed = False
        self._bound_session = None
        self._epoch = 0
        self._tensor_map_key = None
        self._tensor_maps = {}
        self._tensor_map_sources = ()
        self._arguments = {
            name: torch.zeros(count, dtype=dtype, device=self.device)
            for name, (dtype, count) in _workspace_specs(self.layout).items()
        }
        for name in (
            'route_result_index',
            'meta_source_rank',
            'meta_result_index',
            'task_local_expert',
            'task_pool_row',
        ):
            self._arguments[name].fill_(-1)

        pool_rows = int(self.layout['pool_rows'])
        max_rows = int(self.layout['max_rows'])
        hidden = int(self.layout['hidden'])
        intermediate = int(self.layout['intermediate_hidden'])
        up_gate = 2 * intermediate

        self._pool_fp8 = torch.empty(pool_rows * hidden, dtype=torch.uint8, device=self.device)
        self._pool_scales = torch.empty(
            hidden // 128 * pool_rows, dtype=torch.int32, device=self.device
        )
        self._intermediate_fp8 = torch.empty(
            pool_rows * intermediate, dtype=torch.uint8, device=self.device
        )
        self._intermediate_scales = torch.empty(
            intermediate // 128 * pool_rows, dtype=torch.int32, device=self.device
        )
        self._w1_output = torch.empty(
            pool_rows * up_gate, dtype=torch.bfloat16, device=self.device
        )
        self._w2_output = torch.empty(
            pool_rows * hidden, dtype=torch.bfloat16, device=self.device
        )
        self.output = torch.empty(
            (max_rows, hidden), dtype=torch.bfloat16, device=self.device
        )

        self._arguments.update({
            'intermediate_fp8': self._intermediate_fp8,
            'intermediate_sfa_u8': self._intermediate_scales.view(torch.uint8),
            'final_output': self.output,
            'pool_fp8_u32': self._pool_fp8.view(torch.int32),
            'pool_sf_u32': self._pool_scales,
        })
        if self.enable_phase_trace:
            self._arguments['phase_timestamps'] = torch.empty(
                int(self.layout['phase_timestamp_count']),
                dtype=torch.uint64,
                device=self.device,
            )
            self._arguments['peer_phase_timestamps'] = torch.empty(
                int(self.layout['world_size']), dtype=torch.uint64, device=self.device
            )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def phase_timestamps(self) -> torch.Tensor | None:
        return self._arguments.get('phase_timestamps')

    @property
    def peer_phase_timestamps(self) -> torch.Tensor | None:
        return self._arguments.get('peer_phase_timestamps')

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Clone post-launch counters without exposing mutable workspace state."""

        if self._closed:
            raise RuntimeError('SM120 routed MoE workspace is closed')
        names = (
            'protocol_error',
            'owner_record_counts',
            'source_route_counts',
            'result_ovf_cursor',
            'total_valid_routes',
            'total_m_tasks',
        )
        return {
            name: self._arguments[name].detach().clone()
            for name in names
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._bound_session is not None:
            session = self._bound_session()
            if session is not None and not session.closed:
                raise RuntimeError(
                    'close the bound SM120 routed MoE session before its workspace'
                )
        torch.cuda.synchronize(self.device)
        self._arguments.clear()
        self._tensor_maps.clear()
        self._tensor_map_sources = ()
        for name in (
            '_pool_fp8', '_pool_scales', '_intermediate_fp8', '_intermediate_scales',
            '_w1_output', '_w2_output', 'output',
        ):
            setattr(self, name, None)
        self._bound_session = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _prepare_tensor_maps(
        self,
        w1_weight: torch.Tensor,
        w1_scales: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scales: torch.Tensor,
    ) -> None:
        key = tuple(
            (tensor.data_ptr(), tensor.dtype, tensor.device)
            for tensor in (w1_weight, w1_scales, w2_weight, w2_scales)
        )
        if key == self._tensor_map_key:
            return

        layout = self.layout
        local_experts = int(layout['experts_per_rank'])
        hidden = int(layout['hidden'])
        intermediate = int(layout['intermediate_hidden'])
        pool_rows = int(layout['pool_rows'])
        task_rows = int(layout['task_rows'])
        up_gate = 2 * intermediate
        make = _C.make_sm120_tma_2d

        maps = {
            'W1_A': make(self._pool_fp8, 'uint8', hidden, pool_rows, hidden,
                         128, task_rows, 128),
            'W1_B': make(w1_weight, 'fp4', hidden, up_gate * local_experts,
                         hidden // 2, 128, 128, 128),
            'W1_SFA': make(self._pool_scales, 'int32', pool_rows, hidden // 128,
                           pool_rows * 4, task_rows, 1, 0),
            'W1_SFB': make(w1_scales, 'int32', up_gate,
                           hidden // 128 * local_experts, up_gate * 4, 128, 1, 0),
            'W1_D': make(self._w1_output, 'bfloat16', up_gate, pool_rows,
                         up_gate * 2, 64, task_rows, 128),
            'W2_A': make(self._intermediate_fp8, 'uint8', intermediate, pool_rows,
                         intermediate, 128, task_rows, 128),
            'W2_B': make(w2_weight, 'fp4', intermediate, hidden * local_experts,
                         intermediate // 2, 128, 128, 128),
            'W2_SFA': make(self._intermediate_scales, 'int32', pool_rows,
                           intermediate // 128, pool_rows * 4, task_rows, 1, 0),
            'W2_SFB': make(w2_scales, 'int32', hidden,
                           intermediate // 128 * local_experts, hidden * 4, 128, 1, 0),
            'W2_D': make(self._w2_output, 'bfloat16', hidden, pool_rows,
                         hidden * 2, 64, task_rows, 128),
        }
        self._tensor_maps = maps
        self._tensor_map_sources = (w1_weight, w1_scales, w2_weight, w2_scales)
        self._tensor_map_key = key

    def _advance_epoch(self) -> None:
        self._epoch = (self._epoch + 1) & 0xFFFFFFFF


def fp8_fp4_routed_moe_sm120(
    session: SM120RoutedMoESession,
    workspace: SM120RoutedMoEWorkspace,
    x: torch.Tensor,
    x_scales: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_up_gate: TensorPair,
    w2_down: TensorPair,
    grid_ctas: int = 110,
) -> torch.Tensor:
    """Launch the fixed DeepSeek-V4-Flash SM120 routed MoE kernel.

    ``w1_up_gate`` uses contiguous ``[32, 4096, 2048]`` packed-MXFP4 weights
    and ``[32, 32, 4096]`` packed scale words. ``w2_down`` uses
    ``[32, 4096, 1024]`` weights and ``[32, 16, 4096]`` scale words. The
    returned BF16 view is owned by ``workspace`` and is reused by later calls.
    """
    if not isinstance(session, SM120RoutedMoESession):
        raise TypeError('session must be an SM120RoutedMoESession')
    if not isinstance(workspace, SM120RoutedMoEWorkspace):
        raise TypeError('workspace must be an SM120RoutedMoEWorkspace')
    with session._launch_lock:
        return _fp8_fp4_routed_moe_sm120_locked(
            session,
            workspace,
            x,
            x_scales,
            topk_indices,
            topk_weights,
            w1_up_gate,
            w2_down,
            grid_ctas,
        )


def _fp8_fp4_routed_moe_sm120_locked(
    session: SM120RoutedMoESession,
    workspace: SM120RoutedMoEWorkspace,
    x: torch.Tensor,
    x_scales: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_up_gate: TensorPair,
    w2_down: TensorPair,
    grid_ctas: int,
) -> torch.Tensor:
    if session.closed:
        raise RuntimeError('SM120 routed MoE session is closed')
    if workspace.closed:
        raise RuntimeError('SM120 routed MoE workspace is closed')
    if session.device != workspace.device:
        raise ValueError('session and workspace must use the same CUDA device')
    if torch.cuda.current_device() != session.device.index:
        raise RuntimeError('current CUDA device must match the SM120 routed MoE session')
    session._require_workspace(workspace)

    layout = workspace.layout
    hidden = int(layout['hidden'])
    intermediate = int(layout['intermediate_hidden'])
    local_experts = int(layout['experts_per_rank'])
    topk = int(layout['num_topk'])
    max_rows = int(layout['max_rows'])
    max_grid_ctas = int(layout['max_grid_ctas'])
    if not isinstance(grid_ctas, int) or not session.world_size <= grid_ctas <= max_grid_ctas:
        raise ValueError(
            f'grid_ctas must be between {session.world_size} and {max_grid_ctas}'
        )
    rows = x.size(0) if isinstance(x, torch.Tensor) and x.ndim == 2 else -1
    if rows < 1 or rows > max_rows:
        raise ValueError(f'x must contain between 1 and {max_rows} rows')

    float8_dtype = getattr(torch, 'float8_e4m3fn', torch.uint8)
    _require_tensor(x, 'x', workspace.device, (torch.uint8, float8_dtype), (rows, hidden))
    _require_tensor(x_scales, 'x_scales', workspace.device, torch.int32,
                    (rows, hidden // 128))
    _require_tensor(topk_indices, 'topk_indices', workspace.device, torch.int64,
                    (rows, topk))
    _require_tensor(topk_weights, 'topk_weights', workspace.device, torch.float32,
                    (rows, topk))

    if not isinstance(w1_up_gate, tuple) or len(w1_up_gate) != 2:
        raise TypeError('w1_up_gate must be a (packed_weight, packed_scales) tuple')
    if not isinstance(w2_down, tuple) or len(w2_down) != 2:
        raise TypeError('w2_down must be a (packed_weight, packed_scales) tuple')
    w1_weight, w1_scales = w1_up_gate
    w2_weight, w2_scales = w2_down
    _require_tensor(w1_weight, 'w1_up_gate.weight', workspace.device, torch.int8,
                    (local_experts, 2 * intermediate, hidden // 2))
    _require_tensor(w1_scales, 'w1_up_gate.scales', workspace.device, torch.int32,
                    (local_experts, hidden // 128, 2 * intermediate))
    _require_tensor(w2_weight, 'w2_down.weight', workspace.device, torch.int8,
                    (local_experts, hidden, intermediate // 2))
    _require_tensor(w2_scales, 'w2_down.scales', workspace.device, torch.int32,
                    (local_experts, intermediate // 128, hidden))

    trace_mode = workspace.enable_phase_trace
    if trace_mode not in session._prepared_trace_modes:
        _C.prepare_sm120_fp8_fp4_routed_moe(trace_mode)
        dist.barrier(group=session.group, device_ids=[session.device.index])
        session._prepared_trace_modes.add(trace_mode)

    workspace._prepare_tensor_maps(w1_weight, w1_scales, w2_weight, w2_scales)
    arguments = dict(workspace._arguments)
    arguments.update(workspace._tensor_maps)
    arguments.update({
        'topk_idx_i32': topk_indices.view(torch.int32),
        'topk_weights': topk_weights,
        'x_fp8_i32': x.view(torch.int32),
        'x_sf_i32': x_scales,
    })
    session._bind_workspace(workspace)
    epoch = workspace._epoch
    _C.sm120_fp8_fp4_routed_moe(
        session=session._native,
        arguments=arguments,
        active_rows=rows,
        epoch=epoch,
        grid_ctas=grid_ctas,
        activation_clamp=float(layout['activation_clamp']),
        fast_math=bool(layout['fast_math']),
        enable_phase_trace=workspace.enable_phase_trace,
    )
    workspace._advance_epoch()
    session._next_epoch = workspace._epoch
    session._pending_launch = (workspace, arguments, rows, grid_ctas, epoch)
    return workspace.output[:rows]
