"""Lifecycle tests for the SM90 NVFP4 grouped-nibble prepack.

These tests are intentionally separate from the protected MegaMoE correctness
and benchmark harnesses.  The CPU cases exercise the byte-layout contract and
the process-local/persistent layout metadata.  CUDA-only cases cover the
stream, device, and graph-capture behavior of the native one-launch transform.
"""

from __future__ import annotations

import io

import pytest
import torch

import deep_gemm
from deep_gemm import quantization_nvfp4 as nvfp4


_STORAGE_STATE_ATTR = "_dg_sm90_nvfp4_nibble_group_state"
_GROUPED_LAYOUT = "sm90_nvfp4_grouped_nibbles_v1"
_PLAIN_LAYOUT = "sm90_nvfp4_plain_nibbles_v1"
_FAILED_LAYOUT = "sm90_nvfp4_failed_v1"
_MARKER = torch.tensor(list(b"DGNGv1!!"), dtype=torch.uint8)


def _make_fused_weight(
    *, device: torch.device | str = "cpu", seed: int = 17
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small valid fused 80-byte-row weight and tile-major scales."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # E=3 makes it possible to test a middle-expert slice.  K has two BK128
    # blocks, and block_n=4 keeps the fixture small while preserving the exact
    # serving layout within every row block.
    packed = torch.randint(
        0,
        256,
        (3, 4, 2 * 64),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)
    scale = torch.randint(
        1,
        0x7F,
        (3, 1, 2, 4, 8),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)
    fused = nvfp4.nvfp4_fuse_packed_with_scale_tile_major(
        packed, scale, block_k=128
    )
    return fused, scale


def _row_blocks(fused: torch.Tensor) -> torch.Tensor:
    experts, rows, storage_k = fused.shape
    assert storage_k % 80 == 0
    return fused.view(experts, rows, storage_k // 80, 80)


def _expected_grouped_payload(plain_fused: torch.Tensor) -> torch.Tensor:
    payload = _row_blocks(plain_fused)[..., :64]
    quad = payload.reshape(*payload.shape[:-1], 16, 4)
    expected = torch.empty_like(quad)
    expected[..., 0] = (quad[..., 0] >> 4) | ((quad[..., 1] >> 4) << 4)
    expected[..., 1] = (quad[..., 2] >> 4) | ((quad[..., 3] >> 4) << 4)
    expected[..., 2] = (quad[..., 0] & 0x0F) | ((quad[..., 1] & 0x0F) << 4)
    expected[..., 3] = (quad[..., 2] & 0x0F) | ((quad[..., 3] & 0x0F) << 4)
    return expected.reshape_as(payload)


def _storage_layout(tensor: torch.Tensor) -> str | None:
    state = getattr(tensor.untyped_storage(), _STORAGE_STATE_ATTR, None)
    return state.get("layout") if isinstance(state, dict) else None


def _drop_process_local_state(tensor: torch.Tensor) -> None:
    storage = tensor.untyped_storage()
    if hasattr(storage, _STORAGE_STATE_ATTR):
        delattr(storage, _STORAGE_STATE_ATTR)


def _load_tensor(buffer: io.BytesIO) -> torch.Tensor:
    buffer.seek(0)
    try:
        return torch.load(buffer, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        buffer.seek(0)
        return torch.load(buffer)


def _assert_all_row_markers(fused: torch.Tensor) -> None:
    markers = _row_blocks(fused)[..., 72:80].cpu()
    assert torch.equal(markers, _MARKER.expand_as(markers))


def _require_sm90_devices(count: int = 1) -> list[int]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not hasattr(
        getattr(deep_gemm, "_C", object()),
        "nvfp4_group_nibbles_inplace_sm90",
    ):
        pytest.skip("DeepGEMM native nibble-group transform is not built")
    if torch.cuda.device_count() < count:
        pytest.skip(f"test requires {count} CUDA devices")
    devices = list(range(count))
    if any(torch.cuda.get_device_capability(index)[0] != 9 for index in devices):
        pytest.skip("native nibble-group lifecycle tests require SM90 devices")
    return devices


def test_cpu_transform_is_bit_exact_idempotent_and_dequant_inverse() -> None:
    fused, scale = _make_fused_weight()
    plain = fused.clone()
    plain_rows = _row_blocks(plain)
    expected_payload = _expected_grouped_payload(plain)
    reference = nvfp4.dequantize_nvfp4_to_fp32(fused, scale)

    returned = nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)
    assert returned is fused
    rows = _row_blocks(fused)
    assert torch.equal(rows[..., :64], expected_payload)
    assert torch.equal(rows[..., 64:72], plain_rows[..., 64:72])
    _assert_all_row_markers(fused)
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(fused)
    assert _storage_layout(fused) == _GROUPED_LAYOUT
    assert torch.equal(nvfp4.dequantize_nvfp4_to_fp32(fused, scale), reference)

    snapshot = fused.clone()
    alias = fused.view_as(fused)
    assert alias.untyped_storage() is fused.untyped_storage()
    nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(alias)
    assert torch.equal(fused, snapshot)


def test_cpu_marker_survives_clone_middle_slice_and_save_load() -> None:
    fused, _ = _make_fused_weight(seed=23)
    nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)

    full_clone = fused.clone()
    _drop_process_local_state(full_clone)
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(full_clone)

    # A marker only at the allocation's outer endpoints is insufficient here:
    # this clone contains solely the middle expert and must still self-identify.
    middle_clone = fused[1:2].clone()
    _drop_process_local_state(middle_clone)
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(middle_clone)
    _assert_all_row_markers(middle_clone)

    buffer = io.BytesIO()
    torch.save(fused[1:2], buffer)
    restored = _load_tensor(buffer)
    _drop_process_local_state(restored)
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(restored)
    before = restored.clone()
    nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(restored)
    assert torch.equal(restored, before)


def test_cpu_copy_invalidates_cached_layout_via_tensor_version() -> None:
    grouped, _ = _make_fused_weight(seed=31)
    plain, _ = _make_fused_weight(seed=37)
    nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(grouped)

    destination = grouped.clone()
    _drop_process_local_state(destination)
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(destination)
    grouped_source = grouped.clone()

    version = destination._version
    destination.copy_(plain)
    assert destination._version > version
    assert not nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(destination)
    assert _storage_layout(destination) == _PLAIN_LAYOUT

    version = destination._version
    destination.copy_(grouped_source)
    assert destination._version > version
    assert nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(destination)
    assert _storage_layout(destination) == _GROUPED_LAYOUT


def test_cpu_transform_failure_is_fail_closed() -> None:
    fused, _ = _make_fused_weight(seed=41)
    with pytest.raises((AssertionError, ValueError)):
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused, chunk_rows=0)

    assert _storage_layout(fused) == _FAILED_LAYOUT
    with pytest.raises(RuntimeError, match="fail-closed"):
        nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(fused)
    with pytest.raises(RuntimeError, match="fail-closed"):
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)


def test_cuda_alias_double_call_and_two_stream_visibility() -> None:
    device = _require_sm90_devices()[0]
    fused, _ = _make_fused_weight(device=f"cuda:{device}", seed=47)
    plain = fused.cpu()
    expected_payload = _expected_grouped_payload(plain)
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)

    with torch.cuda.stream(producer):
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)
    # The prepack API promises completion before it publishes grouped state, so
    # a different stream may consume it without a wrapper-local event.
    with torch.cuda.stream(consumer):
        observed = fused.clone()
    consumer.synchronize()

    assert torch.equal(_row_blocks(observed.cpu())[..., :64], expected_payload)
    _assert_all_row_markers(observed)
    snapshot = observed.clone()
    with torch.cuda.stream(consumer):
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused.view_as(fused))
    consumer.synchronize()
    assert torch.equal(fused, snapshot)


def test_cuda_transform_uses_weight_device_not_current_device() -> None:
    current_device, weight_device = _require_sm90_devices(count=2)
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(current_device)
        fused, _ = _make_fused_weight(
            device=f"cuda:{weight_device}", seed=53
        )
        plain = fused.cpu()
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)
        assert torch.cuda.current_device() == current_device
        assert torch.equal(
            _row_blocks(fused.cpu())[..., :64], _expected_grouped_payload(plain)
        )
    finally:
        torch.cuda.set_device(previous_device)


def test_cuda_graph_rejects_lazy_transform_before_prepack() -> None:
    device = _require_sm90_devices()[0]
    torch.cuda.set_device(device)
    fused, _ = _make_fused_weight(device=f"cuda:{device}", seed=59)
    before = fused.clone()
    graph = torch.cuda.CUDAGraph()

    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(graph):
            nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)
    assert not nvfp4.nvfp4_is_nibble_grouped_for_mega_moe_sm90(fused)
    assert torch.equal(fused, before)


def test_cuda_graph_keeps_prepacked_weight_unchanged() -> None:
    device = _require_sm90_devices()[0]
    torch.cuda.set_device(device)
    fused, _ = _make_fused_weight(device=f"cuda:{device}", seed=61)
    nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused)
    before = fused.clone()
    captured = torch.empty_like(fused)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # An already-prepacked alias is a metadata-only no-op and is therefore
        # safe during capture; the copy proves subsequent graph work sees it.
        nvfp4.nvfp4_group_nibbles_for_mega_moe_sm90(fused.view_as(fused))
        captured.copy_(fused)
    graph.replay()
    torch.cuda.synchronize(device)

    assert torch.equal(fused, before)
    assert torch.equal(captured, before)
