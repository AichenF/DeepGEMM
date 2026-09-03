"""Offline MXFP4 quantization for SM90 fused MegaMoE.

Mirrors ``quantization_nvfp4.py``'s API and Marlin-style nibble packing, but
targets OCP MXFP4 (E2M1 elements, group_size=32, E8M0 power-of-two shared
scale) instead of NVIDIA NVFP4 (group_size=16, UE4M3 shared scale). This is
the format the DeepSeek-V4-Flash checkpoint's routed experts are stored in
(see sglang's ``layers/quantization/mxfp4.py``, ``_UE8M0_ONE = 127``).
"""
import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0

# E8M0: unsigned 8-bit power-of-two exponent, value = 2 ** (code - 127).
# code=255 is reserved (NaN) per the OCP MX spec; the quantizer never emits it.
UE8M0_BIAS = 127
UE8M0_NAN_CODE = 255


def fp32_to_fp4_nibble(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs().clamp_max(FP4_MAX)
    # Midpoints for nearest E2M1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    nibble_idx = torch.bucketize(mag.to(torch.float32), boundaries).to(torch.uint8)
    return sign | nibble_idx


def fp32_to_ue8m0_ceil(x: torch.Tensor) -> torch.Tensor:
    """Encode non-negative scales to the smallest power-of-two >= x (E8M0)."""
    x = x.to(torch.float32).clamp(min=2.0 ** (-UE8M0_BIAS), max=2.0 ** (254 - UE8M0_BIAS))
    exp_unbiased = torch.ceil(torch.log2(x))
    code = (exp_unbiased + UE8M0_BIAS).to(torch.int32).clamp(0, UE8M0_NAN_CODE - 1)
    return code.to(torch.uint8)


def ue8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    code = scale.to(torch.int32) & 0xFF
    value = torch.exp2((code - UE8M0_BIAS).to(torch.float32))
    # NaN code has no finite representation; callers must never emit it, but
    # guard dequant so a corrupt scale byte surfaces as NaN, not a silent 2**128.
    value = torch.where(code == UE8M0_NAN_CODE, torch.full_like(value, float("nan")), value)
    return value


def quantize_to_mxfp4(weight: torch.Tensor, group_size: int = 32):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-32 E8M0 scale."""
    assert weight.is_floating_point() or weight.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    scale_ue8m0 = fp32_to_ue8m0_ceil(desired_scale.squeeze(-1))
    scale = ue8m0_to_fp32(scale_ue8m0).unsqueeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # Marlin permutation: chunk of 8 K nibbles -> 4 bytes with
    #   byte b: low = K[b+4], high = K[b].
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    return packed, scale_ue8m0.contiguous()


def dequantize_mxfp4_to_fp32(
    packed: torch.Tensor, scale_ue8m0: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    *outer_shape, half_k = packed.shape
    K = half_k * 2
    hi = (packed >> 4) & 0xF
    lo = packed & 0xF
    # Undo the Marlin chunk-of-8 permutation: byte b -> nibbles [b, b+4].
    nibbles = torch.empty(*outer_shape, K, dtype=torch.uint8, device=packed.device)
    nibbles_view = nibbles.view(*outer_shape, K // 8, 8)
    nibbles_view[..., 0:4] = hi.view(*outer_shape, K // 8, 4)
    nibbles_view[..., 4:8] = lo.view(*outer_shape, K // 8, 4)

    sign = ((nibbles >> 3) & 0x1).to(torch.float32)
    mag = FP4_VALUES.to(packed.device)[(nibbles & 0x7).long()]
    elements = torch.where(sign.bool(), -mag, mag)

    G = K // group_size
    elements = elements.view(*outer_shape, G, group_size)
    scale = ue8m0_to_fp32(scale_ue8m0).view(*outer_shape, G, 1)
    return (elements * scale).view(*outer_shape, K)


def mxfp4_scale_to_tile_major(
    scale_ue8m0: torch.Tensor,
    block_n: int = 256,
    block_k: int = 128,
    group_size: int = 32,
) -> torch.Tensor:
    """Repack row-major ``(E, N, K/32)`` E8M0 scales for SM90 tile-local loads.

    Same reshape as ``nvfp4_scale_to_tile_major``; only ``group_size`` differs,
    so a BK128 row carries 4 scale bytes instead of NVFP4's 8.
    """
    assert scale_ue8m0.dtype == torch.uint8
    assert scale_ue8m0.dim() == 3
    assert block_k % group_size == 0
    groups_per_k_block = block_k // group_size
    E, N, G = scale_ue8m0.shape
    assert N % block_n == 0
    assert G % groups_per_k_block == 0
    return (
        scale_ue8m0.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def mxfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
) -> torch.Tensor:
    """Pack each BK128 MXFP4 row as ``64B FP4 + 4B E8M0 scale + 12B padding``.

    The 80-byte row stride is kept identical to the NVFP4 bridge so the same
    K-major TMA descriptor path is reused; MXFP4 simply leaves 12 padding bytes
    instead of 8 because group_size=32 halves the scale bytes per row.
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
    assert groups_per_k_block == block_k // 32
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
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
