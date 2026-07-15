"""Offline NVFP4 quantization for SM90 fused MegaMoE."""
import threading

import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0

UE4M3_MIN_DENORM = 2.0 ** -9
UE4M3_MAX_DENORM = 7.0 * UE4M3_MIN_DENORM
UE4M3_MIN_NORMAL = 2.0 ** -6
UE4M3_MAX_FINITE = 448.0


_NVFP4_NIBBLE_GROUP_LOCK = threading.RLock()
_NVFP4_NIBBLE_GROUP_STORAGE_ATTR = '_dg_sm90_nvfp4_nibble_group_state'
_NVFP4_NIBBLE_GROUP_MARKER = b'DGNGv1!!'
_NVFP4_NIBBLE_GROUP_MARKER_U64 = int.from_bytes(
    _NVFP4_NIBBLE_GROUP_MARKER, byteorder='little', signed=False)


def _set_nvfp4_storage_layout(fused_weight: torch.Tensor, layout: str) -> None:
    storage = fused_weight.untyped_storage()
    setattr(storage, _NVFP4_NIBBLE_GROUP_STORAGE_ATTR, {
        'layout': layout,
        'nbytes': storage.nbytes(),
        'version': int(fused_weight._version),
    })


def nvfp4_is_nibble_grouped_for_mega_moe_sm90(fused_weight: torch.Tensor) -> bool:
    """Return whether the underlying storage uses the grouped-nibble layout.

    The marker lives on ``UntypedStorage`` rather than on a Tensor wrapper so
    views and aliases observe the same state and cannot regroup the bytes.
    """
    storage = fused_weight.untyped_storage()
    state = getattr(storage, _NVFP4_NIBBLE_GROUP_STORAGE_ATTR, None)
    state_matches_storage = (
        isinstance(state, dict)
        and state.get('nbytes') == storage.nbytes()
    )
    if state_matches_storage:
        layout = state.get('layout')
        if layout in ('sm90_nvfp4_transforming_v1', 'sm90_nvfp4_failed_v1'):
            raise RuntimeError(
                f'NVFP4 storage is in fail-closed layout state {layout!r}; '
                'reload and prepack the weights before use')
        if state.get('version') == int(fused_weight._version):
            return layout == 'sm90_nvfp4_grouped_nibbles_v1'

    # Tensor wrapper attributes disappear across clone/.to()/torch.save. Every
    # fused 80-byte row therefore carries an in-band marker. Validate all rows
    # whenever metadata is absent or its Tensor version is stale: checking only
    # the allocation endpoints would misclassify a partially copied expert.
    if (
        fused_weight.dtype != torch.uint8
        or fused_weight.dim() != 3
        or fused_weight.shape[2] % 80 != 0
        or fused_weight.numel() == 0
    ):
        return False

    def count_grouped_markers() -> tuple[int, int]:
        rows = fused_weight.view(-1, 80)
        # Padding words are naturally 8-byte aligned (72 mod 8 == 0). Viewing
        # them as int64 avoids allocating an eight-times-larger bytewise mask.
        marker_words = rows[:, 72:80].view(torch.int64).reshape(-1)
        count = int(torch.count_nonzero(
            marker_words == _NVFP4_NIBBLE_GROUP_MARKER_U64).item())
        return count, marker_words.numel()

    if fused_weight.is_cuda:
        with torch.cuda.device(fused_weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    'NVFP4 layout metadata must be restored before CUDA graph capture')
            num_grouped, num_rows = count_grouped_markers()
    else:
        num_grouped, num_rows = count_grouped_markers()
    if num_grouped not in (0, num_rows):
        _set_nvfp4_storage_layout(fused_weight, 'sm90_nvfp4_failed_v1')
        raise RuntimeError(
            'NVFP4 storage contains a mix of grouped and plain row markers; '
            'reload and prepack the complete weight before use (fail-closed)')
    grouped = num_grouped == num_rows
    # Storage metadata describes the whole allocation, so do not publish it
    # from a partial view whose out-of-view rows were not validated. Managed
    # aliases normally inherit the state already attached by the full tensor.
    if fused_weight.storage_offset() == 0 and storage.nbytes() == fused_weight.numel():
        _set_nvfp4_storage_layout(
            fused_weight,
            'sm90_nvfp4_grouped_nibbles_v1' if grouped else 'sm90_nvfp4_plain_nibbles_v1',
        )
    return grouped


def fp32_to_fp4_nibble(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs().clamp_max(FP4_MAX)
    # Midpoints for nearest E2M1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    # This avoids materializing an extra trailing dimension of size 8.
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    nibble_idx = torch.bucketize(mag.to(torch.float32), boundaries).to(torch.uint8)
    return sign | nibble_idx


def fp32_to_ue4m3_ceil(x: torch.Tensor) -> torch.Tensor:
    """Encode non-negative scales to the smallest finite UE4M3 value >= x."""
    x = x.to(torch.float32).clamp(min=UE4M3_MIN_DENORM, max=UE4M3_MAX_FINITE)

    denorm_code = torch.ceil(x / UE4M3_MIN_DENORM).to(torch.int32).clamp(1, 7)

    x_norm = torch.clamp(x, min=UE4M3_MIN_NORMAL)
    exp_unbiased = torch.floor(torch.log2(x_norm))
    exp_bits = (exp_unbiased + 7).to(torch.int32)
    base = torch.exp2(exp_unbiased)
    mant = torch.ceil((x_norm / base - 1.0) * 8.0 - 1e-6).to(torch.int32)
    overflow = mant > 7
    exp_bits = torch.where(overflow, exp_bits + 1, exp_bits)
    mant = torch.where(overflow, torch.zeros_like(mant), mant).clamp(0, 7)
    normal_code = (exp_bits * 8 + mant).clamp(8, 0x7E)

    code = torch.where(x <= UE4M3_MAX_DENORM, denorm_code, normal_code)
    return code.to(torch.uint8)


def ue4m3_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    code = scale.to(torch.int32) & 0x7F
    exp_bits = code >> 3
    mant = code & 0x7
    denorm = mant.to(torch.float32) * UE4M3_MIN_DENORM
    normal = (1.0 + mant.to(torch.float32) * 0.125) * torch.exp2((exp_bits - 7).to(torch.float32))
    value = torch.where(exp_bits == 0, denorm, normal)
    value = torch.where(code == 0x7F, torch.full_like(value, UE4M3_MAX_FINITE), value)
    return value


def quantize_to_nvfp4(weight: torch.Tensor, group_size: int = 16):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-16 UE4M3 scale."""
    assert weight.is_floating_point() or weight.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    scale_ue4m3 = fp32_to_ue4m3_ceil(desired_scale.squeeze(-1))
    scale = ue4m3_to_fp32(scale_ue4m3).unsqueeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # Marlin permutation: chunk of 8 K nibbles -> 4 bytes with
    #   byte b: low = K[b+4], high = K[b].
    # Marlin's bit shift produces frag_b[0]=[K0..K3], frag_b[1]=[K4..K7].
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    return packed, scale_ue4m3.contiguous()


def nvfp4_scale_to_tile_major(
    scale_ue4m3: torch.Tensor,
    block_n: int = 128,
    block_k: int = 128,
    group_size: int = 16,
) -> torch.Tensor:
    """Repack row-major ``(E, N, K/16)`` UE4M3 scales for SM90 tile-local loads.

    The kernel consumes scales as ``(E, N/block_n, K/block_k, block_n, block_k/16)``.
    This makes the 128 row-local scale vectors touched by a math warpgroup
    contiguous instead of striding by the full K scale dimension.
    """
    assert scale_ue4m3.dtype == torch.uint8
    assert scale_ue4m3.dim() == 3
    assert block_k % group_size == 0
    groups_per_k_block = block_k // group_size
    E, N, G = scale_ue4m3.shape
    assert N % block_n == 0
    assert G % groups_per_k_block == 0
    return (
        scale_ue4m3.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def nvfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
) -> torch.Tensor:
    """Pack each BK128 NVFP4 row as ``64B FP4 + 8B UE4M3 scale + 8B padding``.

    The SM90 NVFP4 bridge still keeps compact FP4 weights, but this layout lets
    the B TMA load bring packed values and their row-local scale bytes together.
    The returned tensor keeps a 3D public weight shape ``(E, N, K/128*80)`` so
    the normal K-major TMA descriptor path can be reused.
    """
    assert packed.dtype == torch.uint8
    assert scale_tile_major.dtype == torch.uint8
    assert packed.dim() == 3
    assert scale_tile_major.dim() == 5
    E, N, K_half = packed.shape
    E_s, n_blocks, k_blocks, block_n, groups_per_k_block = scale_tile_major.shape
    fused_row_bytes = block_k // 2 + 16
    scale_offset = block_k // 2
    assert E == E_s
    assert N == n_blocks * block_n
    assert K_half == k_blocks * (block_k // 2)
    assert groups_per_k_block == block_k // 16
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    fused = torch.empty(
        (E, n_blocks, k_blocks, block_n, fused_row_bytes),
        dtype=torch.uint8,
        device=packed.device,
    )
    fused[..., :scale_offset] = packed_tile
    fused[..., scale_offset : scale_offset + groups_per_k_block] = scale_tile_major
    result = (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(E, N, k_blocks * fused_row_bytes)
        .contiguous()
    )
    # Padding was previously left uninitialized. Zeroing it makes the standard
    # layout deterministic and reserves its first/last words for a persistent
    # grouped-layout marker.
    result.view(E, N, k_blocks, fused_row_bytes)[..., scale_offset + groups_per_k_block:].zero_()
    _set_nvfp4_storage_layout(result, 'sm90_nvfp4_plain_nibbles_v1')
    return result


def dequantize_nvfp4_to_fp32(packed: torch.Tensor, scale_ue4m3: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    grouped_nibble_layout = nvfp4_is_nibble_grouped_for_mega_moe_sm90(packed)
    if scale_ue4m3.dim() == 5:
        E, n_blocks, k_blocks, block_n, groups_per_k_block = scale_ue4m3.shape
        fused_row_bytes = 80
        fused_k = k_blocks * fused_row_bytes
        if packed.dim() == 3 and packed.shape == (E, n_blocks * block_n, fused_k):
            packed = (
                packed.view(E, n_blocks, block_n, k_blocks, fused_row_bytes)
                .permute(0, 1, 3, 2, 4)[..., :64]
                .permute(0, 1, 3, 2, 4)
                .reshape(E, n_blocks * block_n, k_blocks * 64)
                .contiguous()
            )
        scale_ue4m3 = (
            scale_ue4m3.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, n_blocks * block_n, k_blocks * groups_per_k_block)
        )
    if grouped_nibble_layout:
        # Convert the debug/reference copy back to Marlin byte order. The
        # serving tensor itself remains grouped and is never mutated here.
        *packed_outer_shape, packed_k_half = packed.shape
        q = packed.view(*packed_outer_shape, packed_k_half // 4, 4)
        high = torch.stack(
            [q[..., 0] & 0x0F, q[..., 0] >> 4,
             q[..., 1] & 0x0F, q[..., 1] >> 4],
            dim=-1,
        )
        low = torch.stack(
            [q[..., 2] & 0x0F, q[..., 2] >> 4,
             q[..., 3] & 0x0F, q[..., 3] >> 4],
            dim=-1,
        )
        packed = (low | (high << 4)).reshape(*packed_outer_shape, packed_k_half).contiguous()
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    G = K // group_size
    # Inverse Marlin permutation: each 4-byte chunk represents 8 K elements;
    # low nibbles -> K[4..7], high nibbles -> K[0..3].
    pck = packed.view(*outer_shape, K // 8, 4)
    low = pck & 0x0F
    high = (pck >> 4) & 0x0F
    nibbles = torch.cat([high, low], dim=-1).view(*outer_shape, K)
    sign_bit = (nibbles >> 3) & 0x1
    mag_idx = (nibbles & 0x7).to(torch.long)
    fp4_values = FP4_VALUES.to(packed.device)
    mag = fp4_values[mag_idx]
    values = torch.where(sign_bit.bool(), -mag, mag)
    scale = ue4m3_to_fp32(scale_ue4m3)
    scale_expanded = scale.unsqueeze(-1).expand(*outer_shape, G, group_size).reshape(*outer_shape, K)
    return values * scale_expanded


if __name__ == "__main__":
    torch.manual_seed(0)
    E, N, K = 4, 256, 4096
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16) * 0.3
    w_fp32_ref = w_bf16.to(torch.float32)

    packed, scale_ue4m3 = quantize_to_nvfp4(w_bf16, group_size=16)
    print(f"packed shape: {packed.shape}, scale shape: {scale_ue4m3.shape}")
    print(f"weight bytes: BF16={w_bf16.numel() * 2} -> packed={packed.numel()} + scale={scale_ue4m3.numel()}")

    w_recovered = dequantize_nvfp4_to_fp32(packed, scale_ue4m3, group_size=16)
    err = (w_recovered - w_fp32_ref).abs()
    print(f"Element error: max_abs={err.max():.4f}  mean_abs={err.mean():.4f}")
    mask = w_fp32_ref.abs() > 0.05
    rel_err = (err[mask] / w_fp32_ref.abs()[mask]).mean().item()
    print(f"Element rel error (|ref|>0.05): {rel_err*100:.2f}%")
    print("OK")


def nvfp4_group_nibbles_for_mega_moe_sm90(fused_weight: torch.Tensor,
                                          chunk_rows: int = 512) -> torch.Tensor:
    """Lossless in-place nibble regrouping of a fused 80B/row NVFP4 weight.

    Within each 80-byte row block the 64 payload bytes are permuted so that each
    4-byte group's eight E2M1 nibbles are stored as (4 high nibbles -> low u16,
    4 low nibbles -> high u16), matching the SM90 grouped-nibble decode kernel.
    Scale bytes are untouched; every otherwise-unused padding word receives a
    persistent layout marker. CUDA tensors use one native kernel launch for the
    whole tensor. The CPU fallback retains bounded chunks for offline/debug
    use. State is attached to the underlying storage, making the operation safe
    for aliases and concurrent first use.
    """
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    assert fused_weight.is_contiguous()
    assert fused_weight.shape[2] % 80 == 0
    storage = fused_weight.untyped_storage()

    with _NVFP4_NIBBLE_GROUP_LOCK:
        if nvfp4_is_nibble_grouped_for_mega_moe_sm90(fused_weight):
            fused_weight._dg_sm90_nvfp4_nibble_group = True
            return fused_weight
        if fused_weight.storage_offset() != 0 or storage.nbytes() != fused_weight.numel():
            raise ValueError(
                'NVFP4 nibble grouping requires a dedicated, full uint8 storage; '
                'sliced plain-weight views are not safe to transform in place')

        experts, rows, storage_k = fused_weight.shape
        k_blocks = storage_k // 80
        if fused_weight.is_cuda:
            with torch.cuda.device(fused_weight.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        'NVFP4 nibble grouping must finish before CUDA graph capture')
        _set_nvfp4_storage_layout(fused_weight, 'sm90_nvfp4_transforming_v1')
        try:
            if fused_weight.is_cuda:
                with torch.cuda.device(fused_weight.device):
                    from . import _C
                    _C._nvfp4_group_nibbles_inplace_sm90(fused_weight)
                    # Do not publish the process-local grouped state until the
                    # payload and all persistent markers finish successfully.
                    completion = torch.cuda.Event()
                    completion.record(torch.cuda.current_stream(fused_weight.device))
                    completion.synchronize()
            else:
                assert chunk_rows > 0
                rows_view = fused_weight.view(experts, rows, k_blocks, 80)
                for e in range(experts):
                    for r0 in range(0, rows, chunk_rows):
                        chunk = rows_view[e, r0:r0 + chunk_rows, :, :64]
                        q = chunk.reshape(-1, 16, 4).to(torch.int32)
                        high = (q >> 4) & 0xf
                        low = q & 0xf
                        h16 = high[..., 0] | (high[..., 1] << 4) | (high[..., 2] << 8) | (high[..., 3] << 12)
                        l16 = low[..., 0] | (low[..., 1] << 4) | (low[..., 2] << 8) | (low[..., 3] << 12)
                        grouped = (h16 | (l16 << 16)).to(torch.int32)
                        chunk.copy_(grouped.contiguous().view(torch.uint8).view(chunk.shape))

                marker = torch.tensor(
                    list(_NVFP4_NIBBLE_GROUP_MARKER), dtype=torch.uint8,
                    device=fused_weight.device)
                rows_view[..., 72:80].copy_(marker)
        except BaseException:
            _set_nvfp4_storage_layout(fused_weight, 'sm90_nvfp4_failed_v1')
            raise

        _set_nvfp4_storage_layout(fused_weight, 'sm90_nvfp4_grouped_nibbles_v1')
        # Retain the legacy Tensor marker for callers that inspect it, while
        # correctness decisions use the alias-safe storage marker above.
        fused_weight._dg_sm90_nvfp4_nibble_group = True
    return fused_weight
