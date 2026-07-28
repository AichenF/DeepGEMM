"""Offline MXFP4 quantization and an SM90 FP8-kernel compatibility bridge.

The standard representation produced here is:

* E2M1 payload values packed in the same Marlin byte order used by DeepGEMM's
  SM90 NVFP4 path.
* One round-toward-positive-infinity UE8M0 scale for every 32 values along K.

Hopper has no native MXFP4 tensor-core instruction.  The compatibility helper
therefore expands each per-32 UE8M0 scale into two identical per-16 UE4M3
scales, which can be consumed by the existing SM90 FP4-to-FP8 bridge without
changing its kernel ABI.  The scale range is restricted so every E2M1 value
remains exactly representable in the kernel's E4M3 intermediate.
"""

from typing import Tuple

import torch


MXFP4_BLOCK_SIZE = 32
FP4_MAX = 6.0
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

# UE8M0 code 0x00 represents 2^-127, codes 0x01..0xfe have the
# corresponding FP32 exponent field, and 0xff is NaN.
UE8M0_NAN = 0xFF
UE8M0_BIAS = 127

# The SM90 kernel expands E2M1 * scale into E4M3 before WGMMA.  Restricting the
# scale exponent to this interval keeps both 0.5 * scale and 6 * scale exactly
# representable without E4M3 underflow or saturation.
SM90_MXFP4_MIN_SCALE_EXP = -8
SM90_MXFP4_MAX_SCALE_EXP = 6
SM90_ZERO_BLOCK_SCALE_CODE = UE8M0_BIAS + SM90_MXFP4_MIN_SCALE_EXP


def _check_uint8_scale(scale: torch.Tensor, name: str) -> None:
    if scale.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype torch.uint8, got {scale.dtype}")


def fp32_to_fp4_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Encode FP32 values as E2M1 nibbles with saturation and ties-to-even."""
    x_fp32 = x.to(torch.float32)
    magnitude = x_fp32.abs().clamp_max(FP4_MAX)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=x.device,
    )

    # bucketize maps an exact midpoint to its lower neighbor.  At the three
    # midpoints whose lower E2M1 code is odd, ties-to-even selects the upper
    # neighbor instead.
    index = torch.bucketize(magnitude, boundaries)
    boundary_index = index.clamp_max(boundaries.numel() - 1)
    is_midpoint = (index < boundaries.numel()) & (
        magnitude == boundaries[boundary_index]
    )
    index = index + (is_midpoint & ((index & 1) != 0)).to(index.dtype)

    sign = ((x_fp32 < 0) & (index != 0)).to(torch.uint8) << 3
    return sign | index.to(torch.uint8)


def fp32_to_ue8m0_ceil(x: torch.Tensor) -> torch.Tensor:
    """Encode finite non-negative FP32 values as UE8M0, rounding upward.

    This follows CUTLASS/PTX ``cvt.rp.satfinite.ue8m0`` behavior, including
    mapping zero and values up to 2^-127 to code 0x00.
    """
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")

    x_fp32 = x.to(torch.float32)
    if not bool(torch.isfinite(x_fp32).all().item()):
        raise ValueError("UE8M0 scales must be finite")
    if bool((x_fp32 < 0).any().item()):
        raise ValueError("UE8M0 scales must be non-negative")

    bits = x_fp32.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    # UE8M0 code 0 denotes 2^-127.  FP32 subnormals no larger than that value
    # therefore already fit without incrementing the exponent field.
    round_up = (
        (mantissa != 0)
        & (exponent != 0xFE)
        & ~((exponent == 0) & (mantissa <= 0x00400000))
    )
    return (exponent + round_up.to(torch.int32)).clamp_max(0xFE).to(torch.uint8)


def ue8m0_to_fp32(scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode standard UE8M0 bytes to FP32 values."""
    _check_uint8_scale(scale_ue8m0, "scale_ue8m0")
    code = scale_ue8m0.to(torch.int32)
    fp32_bits = code << 23
    fp32_bits = torch.where(
        code == 0,
        torch.full_like(fp32_bits, 0x00400000),
        fp32_bits,
    )
    # UE8M0 reserves 0xff for NaN rather than infinity.
    fp32_bits = torch.where(
        code == UE8M0_NAN,
        torch.full_like(fp32_bits, 0x7FFFFFFF),
        fp32_bits,
    )
    return fp32_bits.contiguous().view(torch.float32)


def _pack_marlin(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibbles in the SM90 NVFP4 path's Marlin byte order."""
    *outer_shape, k = nibbles.shape
    if k % 8 != 0:
        raise ValueError(f"K must be divisible by 8 for Marlin packing, got {k}")
    chunks = nibbles.view(*outer_shape, k // 8, 8)
    return (
        (chunks[..., 4:8] | (chunks[..., 0:4] << 4))
        .to(torch.uint8)
        .view(*outer_shape, k // 2)
        .contiguous()
    )


def _unpack_marlin(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_marlin`."""
    *outer_shape, k_half = packed.shape
    k = k_half * 2
    if k % 8 != 0:
        raise ValueError(
            f"packed K storage must decode to a multiple of 8, got K={k}"
        )
    chunks = packed.view(*outer_shape, k // 8, 4)
    low = chunks & 0x0F
    high = (chunks >> 4) & 0x0F
    return torch.cat((high, low), dim=-1).view(*outer_shape, k)


def _as_uint8_packed(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype == torch.uint8:
        return packed.contiguous()
    if packed.dtype == torch.int8:
        return packed.contiguous().view(torch.uint8)
    raise TypeError(
        f"packed MXFP4 payload must have dtype torch.uint8 or torch.int8, "
        f"got {packed.dtype}"
    )


def linear_packed_mxfp4_to_marlin(packed: torch.Tensor) -> torch.Tensor:
    """Repack adjacent-pair MXFP4 bytes into the SM90 Marlin byte order.

    ``packed`` uses the common linear convention where each byte contains
    ``K[2*i]`` in its low nibble and ``K[2*i+1]`` in its high nibble.
    """
    packed_u8 = _as_uint8_packed(packed)
    if packed_u8.dim() == 0:
        raise ValueError("packed must have a K dimension")

    *outer_shape, k_half = packed_u8.shape
    if k_half % 4 != 0:
        raise ValueError(
            "linear MXFP4 storage must contain a whole 8-value packing group; "
            f"got {k_half} bytes along K"
        )

    chunks = packed_u8.view(*outer_shape, k_half // 4, 4)
    even = chunks & 0x0F
    odd = (chunks >> 4) & 0x0F
    marlin = torch.stack(
        (
            (even[..., 0] << 4) | even[..., 2],
            (odd[..., 0] << 4) | odd[..., 2],
            (even[..., 1] << 4) | even[..., 3],
            (odd[..., 1] << 4) | odd[..., 3],
        ),
        dim=-1,
    )
    return marlin.view(*outer_shape, k_half).contiguous()


def quantize_to_mxfp4(
    weight: torch.Tensor,
    group_size: int = MXFP4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to Marlin-packed E2M1 plus per-32 UE8M0 scales."""
    if group_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"standard MXFP4 requires group_size={MXFP4_BLOCK_SIZE}, "
            f"got {group_size}"
        )
    if not weight.is_floating_point():
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    if weight.dim() == 0:
        raise ValueError("weight must have a K dimension")

    *outer_shape, k = weight.shape
    if k % group_size != 0:
        raise ValueError(
            f"K must be divisible by MXFP4 group_size={group_size}, got {k}"
        )

    groups = k // group_size
    weight_fp32 = weight.to(torch.float32).view(
        *outer_shape, groups, group_size
    )
    if not bool(torch.isfinite(weight_fp32).all().item()):
        raise ValueError("weight must contain only finite values")

    desired_scale = weight_fp32.abs().amax(dim=-1) / FP4_MAX
    scale_ue8m0 = fp32_to_ue8m0_ceil(desired_scale)
    scale_fp32 = ue8m0_to_fp32(scale_ue8m0).unsqueeze(-1)
    normalized = (weight_fp32 / scale_fp32).clamp(-FP4_MAX, FP4_MAX)
    nibbles = fp32_to_fp4_e2m1_nibble(normalized).view(*outer_shape, k)
    return _pack_marlin(nibbles), scale_ue8m0.contiguous()


def dequantize_mxfp4_to_fp32(
    packed: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    group_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize Marlin-packed MXFP4 payload and row-major UE8M0 scales."""
    if group_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"standard MXFP4 requires group_size={MXFP4_BLOCK_SIZE}, "
            f"got {group_size}"
        )
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must have dtype torch.uint8, got {packed.dtype}")
    _check_uint8_scale(scale_ue8m0, "scale_ue8m0")
    if packed.dim() == 0:
        raise ValueError("packed must have a K dimension")

    *outer_shape, k_half = packed.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(
            f"decoded K must be divisible by MXFP4 group_size={group_size}, "
            f"got {k}"
        )
    expected_scale_shape = (*outer_shape, k // group_size)
    if tuple(scale_ue8m0.shape) != expected_scale_shape:
        raise ValueError(
            f"scale_ue8m0 shape must be {expected_scale_shape}, "
            f"got {tuple(scale_ue8m0.shape)}"
        )
    if scale_ue8m0.device != packed.device:
        raise ValueError("packed and scale_ue8m0 must be on the same device")

    nibbles = _unpack_marlin(packed)
    sign = ((nibbles >> 3) & 1).bool()
    magnitude_index = (nibbles & 0x7).to(torch.long)
    magnitude = FP4_VALUES.to(device=packed.device)[magnitude_index]
    value = torch.where(sign & (magnitude_index != 0), -magnitude, magnitude)

    groups = k // group_size
    scale_fp32 = ue8m0_to_fp32(scale_ue8m0)
    scale_expanded = (
        scale_fp32.unsqueeze(-1)
        .expand(*outer_shape, groups, group_size)
        .reshape(*outer_shape, k)
    )
    return value * scale_expanded


def mxfp4_ue8m0_scale_to_nvfp4_ue4m3(
    scale_ue8m0: torch.Tensor,
) -> torch.Tensor:
    """Convert kernel-safe per-32 MXFP4 scales to repeated per-16 scales.

    The SM90 bridge materializes ``E2M1 * scale`` in E4M3.  To prevent
    underflow or saturation for any E2M1 code, only UE8M0 scale powers in
    ``[2^-8, 2^6]`` are accepted.  UE8M0 code zero needs payload-aware
    validation and is handled by :func:`prepare_mxfp4_weight_for_sm90`.
    """
    _check_uint8_scale(scale_ue8m0, "scale_ue8m0")
    code = scale_ue8m0.to(torch.int32)

    if bool((code == UE8M0_NAN).any().item()):
        raise ValueError("UE8M0 NaN code 0xff cannot be bridged to UE4M3")
    if bool((code == 0).any().item()):
        raise ValueError(
            "UE8M0 code 0 requires packed-payload validation; use "
            "prepare_mxfp4_weight_for_sm90"
        )

    exponent = code - UE8M0_BIAS
    representable = (
        (exponent >= SM90_MXFP4_MIN_SCALE_EXP)
        & (exponent <= SM90_MXFP4_MAX_SCALE_EXP)
    )
    if not bool(representable.all().item()):
        invalid_exponents = torch.unique(exponent[~representable]).cpu().tolist()
        raise ValueError(
            "MXFP4 UE8M0 scale exponent is outside the SM90 E4M3-safe "
            f"range [{SM90_MXFP4_MIN_SCALE_EXP}, "
            f"{SM90_MXFP4_MAX_SCALE_EXP}]; "
            f"got {invalid_exponents}"
        )

    # UE4M3 powers 2^-8 and 2^-7 are subnormal codes 2 and 4.
    # Powers 2^-6..2^6 are normal codes with a zero mantissa.
    subnormal_code = torch.where(
        exponent == -8,
        torch.full_like(exponent, 2),
        torch.full_like(exponent, 4),
    )
    normal_code = (exponent + 7) << 3
    scale_ue4m3 = torch.where(
        exponent < -6,
        subnormal_code,
        normal_code,
    ).to(torch.uint8)

    # MXFP4 has one scale per 32 K values; the SM90 bridge reads one UE4M3
    # scale per 16, so adjacent output scale bytes must be identical.
    return scale_ue4m3.repeat_interleave(2, dim=-1).contiguous()


def prepare_mxfp4_weight_for_sm90(
    packed: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    packed_layout: str = "marlin",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a pre-quantized MXFP4 tensor for the Hopper compatibility path.

    The scale tensor must be row-major raw UE8M0 with one byte per 32 K
    elements.  ``packed_layout`` may be ``"marlin"`` (the output of
    :func:`quantize_to_mxfp4`) or ``"linear"`` (adjacent K pairs per byte).

    UE8M0 code-zero blocks are accepted only when every corresponding E2M1
    magnitude is zero.  Their scale is then replaced by a kernel-safe value
    without changing the represented tensor.
    """
    if packed_layout == "marlin":
        packed_marlin = _as_uint8_packed(packed)
    elif packed_layout == "linear":
        packed_marlin = linear_packed_mxfp4_to_marlin(packed)
    else:
        raise ValueError(
            f"packed_layout must be 'marlin' or 'linear', got {packed_layout!r}"
        )

    _check_uint8_scale(scale_ue8m0, "scale_ue8m0")
    if packed_marlin.dim() == 0:
        raise ValueError("packed must have a K dimension")
    if packed_marlin.device != scale_ue8m0.device:
        raise ValueError("packed and scale_ue8m0 must be on the same device")

    *outer_shape, k_half = packed_marlin.shape
    k = k_half * 2
    if k % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"decoded K must be divisible by {MXFP4_BLOCK_SIZE}, got {k}"
        )
    num_groups = k // MXFP4_BLOCK_SIZE
    expected_scale_shape = (*outer_shape, num_groups)
    if tuple(scale_ue8m0.shape) != expected_scale_shape:
        raise ValueError(
            f"scale_ue8m0 shape must be {expected_scale_shape}, "
            f"got {tuple(scale_ue8m0.shape)}"
        )

    scale_for_bridge = scale_ue8m0
    code_zero = scale_ue8m0 == 0
    if bool(code_zero.any().item()):
        packed_groups = packed_marlin.view(
            *outer_shape, num_groups, MXFP4_BLOCK_SIZE // 2
        )
        zero_magnitude = ((packed_groups & 0x77) == 0).all(dim=-1)
        invalid_zero_scale = code_zero & ~zero_magnitude
        if bool(invalid_zero_scale.any().item()):
            raise ValueError(
                "UE8M0 code 0 can only be bridged when its packed E2M1 block "
                "contains all-zero magnitudes"
            )
        scale_for_bridge = scale_ue8m0.clone()
        scale_for_bridge[code_zero] = SM90_ZERO_BLOCK_SCALE_CODE

    return (
        packed_marlin,
        mxfp4_ue8m0_scale_to_nvfp4_ue4m3(scale_for_bridge),
    )


# Shorter spelling for callers that only care about the kernel-compat scale.
mxfp4_scale_to_nvfp4_scale = mxfp4_ue8m0_scale_to_nvfp4_ue4m3
