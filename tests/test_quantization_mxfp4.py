import pytest
import torch

import deep_gemm
from deep_gemm.quantization_mxfp4 import (
    dequantize_mxfp4_to_fp32,
    fp32_to_ue8m0_ceil,
    linear_packed_mxfp4_to_marlin,
    mxfp4_ue8m0_scale_to_nvfp4_ue4m3,
    prepare_mxfp4_weight_for_sm90,
    quantize_to_mxfp4,
    ue8m0_to_fp32,
)
from deep_gemm.quantization_nvfp4 import dequantize_nvfp4_to_fp32


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_ue8m0_round_up_and_decode():
    values = torch.tensor(
        [0.0, 2.0**-127, 2.0**-9, 0.75, 1.0, 1.01, 256.0],
        dtype=torch.float32,
    )
    encoded = fp32_to_ue8m0_ceil(values)
    assert encoded.tolist() == [0, 0, 118, 127, 127, 128, 135]

    decoded = ue8m0_to_fp32(encoded)
    expected = torch.tensor(
        [2.0**-127, 2.0**-127, 2.0**-9, 1.0, 1.0, 2.0, 256.0],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_marlin_payload_order_is_compatible():
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    weight = e2m1_values.repeat(4).view(1, 1, 32)
    packed, scale = quantize_to_mxfp4(weight)
    assert scale.tolist() == [[[127]]]
    assert packed[0, 0].tolist() == [0x04, 0x15, 0x26, 0x37] * 4
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(packed, scale),
        weight,
        rtol=0,
        atol=0,
    )


def test_ue8m0_to_ue4m3_bridge_mapping_and_range_checks():
    # UE8M0 codes correspond to the E4M3-safe exponents -8, -7, -6, 0, and 6.
    scale = torch.tensor([119, 120, 121, 127, 133], dtype=torch.uint8)
    bridged = mxfp4_ue8m0_scale_to_nvfp4_ue4m3(scale)
    assert bridged.tolist() == [
        0x02, 0x02,
        0x04, 0x04,
        0x08, 0x08,
        0x38, 0x38,
        0x68, 0x68,
    ]

    with pytest.raises(ValueError, match="E4M3-safe range"):
        mxfp4_ue8m0_scale_to_nvfp4_ue4m3(
            torch.tensor([118, 134], dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="packed-payload validation"):
        mxfp4_ue8m0_scale_to_nvfp4_ue4m3(
            torch.tensor([0], dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="NaN code 0xff"):
        mxfp4_ue8m0_scale_to_nvfp4_ue4m3(
            torch.tensor([0xFF], dtype=torch.uint8)
        )


def test_linear_payload_repack_and_zero_block_bridge():
    linear = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76] * 4],
        dtype=torch.uint8,
    )
    expected_marlin = torch.tensor(
        [[0x04, 0x15, 0x26, 0x37] * 4],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(
        linear_packed_mxfp4_to_marlin(linear),
        expected_marlin,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        linear_packed_mxfp4_to_marlin(linear.view(torch.int8)),
        expected_marlin,
        rtol=0,
        atol=0,
    )

    zero_packed, zero_scale = quantize_to_mxfp4(torch.zeros((1, 1, 32)))
    assert zero_scale.item() == 0
    prepared_packed, prepared_scale = prepare_mxfp4_weight_for_sm90(
        zero_packed, zero_scale
    )
    torch.testing.assert_close(prepared_packed, zero_packed, rtol=0, atol=0)
    assert prepared_scale.tolist() == [[[0x02, 0x02]]]

    nonzero_packed = zero_packed.clone()
    nonzero_packed[..., 0] = 0x01
    with pytest.raises(ValueError, match="all-zero magnitudes"):
        prepare_mxfp4_weight_for_sm90(nonzero_packed, zero_scale)


@pytest.mark.parametrize("device", DEVICES)
def test_quant_dequant_and_nvfp4_bridge(device):
    generator = torch.Generator(device=device).manual_seed(20260728)
    weight = (
        torch.randn((2, 5, 64), generator=generator, device=device)
        * 0.05
    )

    packed, scale_ue8m0 = quantize_to_mxfp4(weight)
    dequantized = dequantize_mxfp4_to_fp32(packed, scale_ue8m0)
    assert packed.shape == (2, 5, 32)
    assert scale_ue8m0.shape == (2, 5, 2)
    assert packed.dtype == torch.uint8
    assert scale_ue8m0.dtype == torch.uint8

    # E2M1's largest adjacent-value gap is 2, so no-clipping quantization has
    # an absolute error no larger than one shared scale.
    scale = ue8m0_to_fp32(scale_ue8m0)
    error_bound = scale.repeat_interleave(32, dim=-1)
    assert bool(
        ((dequantized - weight.float()).abs() <= error_bound).all().item()
    )

    # Offline scale conversion must make the existing NVFP4 dequant path
    # bit-for-bit equivalent to standard MXFP4 dequantization.
    scale_ue4m3 = mxfp4_ue8m0_scale_to_nvfp4_ue4m3(scale_ue8m0)
    compat_dequantized = dequantize_nvfp4_to_fp32(
        packed, scale_ue4m3, group_size=16
    )
    torch.testing.assert_close(
        compat_dequantized,
        dequantized,
        rtol=0,
        atol=0,
    )


def test_prequantized_mxfp4_transform_entrypoint():
    def marlin_to_linear(packed):
        *outer_shape, k_half = packed.shape
        marlin_chunks = packed.view(*outer_shape, k_half // 4, 4)
        nibbles = torch.cat(
            ((marlin_chunks >> 4) & 0x0F, marlin_chunks & 0x0F),
            dim=-1,
        )
        return (
            nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
        ).view(*outer_shape, k_half).contiguous()

    generator = torch.Generator().manual_seed(20260728)
    l1_bf = torch.randn((1, 512, 128), generator=generator) * 0.05
    l2_bf = torch.randn((1, 256, 128), generator=generator) * 0.05
    l1_mx = quantize_to_mxfp4(l1_bf)
    l2_mx = quantize_to_mxfp4(l2_bf)

    got_l1, got_l2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
        l1_mx, l2_mx
    )
    expected_l1, expected_l2 = (
        deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
            prepare_mxfp4_weight_for_sm90(*l1_mx),
            prepare_mxfp4_weight_for_sm90(*l2_mx),
        )
    )
    for got, expected in zip((*got_l1, *got_l2), (*expected_l1, *expected_l2)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    linear_l1 = (marlin_to_linear(l1_mx[0]).view(torch.int8), l1_mx[1])
    linear_l2 = (marlin_to_linear(l2_mx[0]).view(torch.int8), l2_mx[1])
    linear_got_l1, linear_got_l2 = (
        deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
            linear_l1,
            linear_l2,
            packed_layout="linear",
        )
    )
    for got, expected in zip(
        (*linear_got_l1, *linear_got_l2),
        (*got_l1, *got_l2),
    ):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
