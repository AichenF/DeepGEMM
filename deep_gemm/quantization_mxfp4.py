"""Offline MXFP4 quantization for SM90 fused MegaMoE.

MXFP4 = E2M1 FP4 elements + a per-32 E8M0 (power-of-two) shared scale, and no
second-level/global scale. The packed FP4 byte layout matches the predecessor
Marlin-style path; the only micro-scale differences vs the NVFP4 predecessor are
the 32-element group (instead of 16) and the E8M0 scale code (instead of UE4M3).
"""
import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0

# E8M0 shared scale: value = 2^(code - 127). Code 0xff is reserved (NaN) and is
# never emitted by the quantizer (codes are clamped to 0..0xfe).
E8M0_BIAS = 127
E8M0_MAX_CODE = 254


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


def fp32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Encode a non-negative scale to the smallest power-of-two E8M0 code >= x.

    Using ceil(log2(x)) guarantees the chosen scale is >= x = amax / FP4_MAX, so
    the normalized magnitudes never exceed FP4_MAX and no FP4 overflow occurs.
    """
    x = x.to(torch.float32)
    tiny = float(2.0 ** -E8M0_BIAS)
    x = torch.where(x > 0, x, torch.full_like(x, tiny))
    exp = torch.ceil(torch.log2(x))
    code = (exp + E8M0_BIAS).to(torch.int32).clamp(0, E8M0_MAX_CODE)
    return code.to(torch.uint8)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    code = scale.to(torch.int32) & 0xFF
    return torch.exp2((code - E8M0_BIAS).to(torch.float32))


def quantize_to_mxfp4(weight: torch.Tensor, group_size: int = 32):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-32 E8M0 scale."""
    assert weight.is_floating_point() or weight.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    scale_e8m0 = fp32_to_e8m0(desired_scale.squeeze(-1))
    scale = e8m0_to_fp32(scale_e8m0).unsqueeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # Marlin permutation: chunk of 8 K nibbles -> 4 bytes with
    #   byte b: low = K[b+4], high = K[b].
    # Marlin's bit shift produces frag_b[0]=[K0..K3], frag_b[1]=[K4..K7].
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    return packed, scale_e8m0.contiguous()


def mxfp4_scale_to_tile_major(
    scale_e8m0: torch.Tensor,
    block_n: int = 256,
    block_k: int = 128,
    group_size: int = 32,
) -> torch.Tensor:
    """Repack row-major ``(E, N, K/32)`` E8M0 scales for SM90 tile-local loads.

    The kernel consumes scales as ``(E, N/block_n, K/block_k, block_n, 8)`` where
    the trailing 8 bytes are the ``block_k/32`` per-32 E8M0 codes duplicated x2.
    The duplication keeps the fused 80-byte weight rows on the predecessor per-16
    scale-byte cadence, so the FP4->FP8 decoders apply one code across two
    adjacent 16-element sub-blocks (i.e. per-32) with no kernel index change.
    """
    assert scale_e8m0.dtype == torch.uint8
    assert scale_e8m0.dim() == 3
    assert block_k % group_size == 0
    groups_per_k_block = block_k // group_size
    E, N, G = scale_e8m0.shape
    assert N % block_n == 0
    assert G % groups_per_k_block == 0
    tile_major = (
        scale_e8m0.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    # Duplicate each per-32 code x2 -> 8 bytes per BK128 row: [e0,e0,e1,e1,...].
    return tile_major.repeat_interleave(2, dim=-1).contiguous()


def mxfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
) -> torch.Tensor:
    """Pack each BK128 MXFP4 row as ``64B FP4 + 8B E8M0 scale + 8B padding``.

    The 8 scale bytes are the 4 per-32 E8M0 codes duplicated x2 (see
    ``mxfp4_scale_to_tile_major``). The returned tensor keeps a 3D public weight
    shape ``(E, N, K/128*80)`` so the normal K-major TMA descriptor path can be
    reused.
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
    # 8 duplicated E8M0 bytes per BK128 row (block_k / 16 slots).
    assert groups_per_k_block == block_k // 16
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    # Keep the final 8-byte padding in every BK128 row deterministic.
    fused = torch.zeros(
        (E, n_blocks, k_blocks, block_n, fused_row_bytes),
        dtype=torch.uint8,
        device=packed.device,
    )
    fused[..., :scale_offset] = packed_tile
    fused[..., scale_offset : scale_offset + groups_per_k_block] = scale_tile_major
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(E, N, k_blocks * fused_row_bytes)
        .contiguous()
    )


def dequantize_mxfp4_to_fp32(packed: torch.Tensor, scale_e8m0: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    if scale_e8m0.dim() == 5:
        E, n_blocks, k_blocks, block_n, groups_per_k_block = scale_e8m0.shape
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
        scale_e8m0 = (
            scale_e8m0.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, n_blocks * block_n, k_blocks * groups_per_k_block)
        )
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    # Derive the effective group from the scale count so this handles both the
    # raw per-32 layout (K/32 codes) and the tile-major per-16 duplicated layout
    # (K/16 codes) transparently.
    num_scales = scale_e8m0.shape[-1]
    assert K % num_scales == 0
    group_size_eff = K // num_scales
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
    scale = e8m0_to_fp32(scale_e8m0)
    G = K // group_size_eff
    scale_expanded = scale.unsqueeze(-1).expand(*outer_shape, G, group_size_eff).reshape(*outer_shape, K)
    return values * scale_expanded
