#!/usr/bin/env python3
"""Isolate native fused-row MXFP4 register decode and RS-WGMMA mapping."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


INCLUDE_CANDIDATES = (
    Path("/lustre/raplab/client/xutingz/fac/DeepGEMM/deep_gemm/include"),
    Path("/home/xutingz/fac/DeepGEMM/deep_gemm/include"),
)
DEEP_GEMM_INCLUDE = next(path for path in INCLUDE_CANDIDATES if path.exists())


def interleave_l1(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    experts, rows, *rest = tensor.shape
    half = rows // 2
    gate = tensor[:, :half].reshape(
        experts, half // granularity, granularity, *rest
    )
    up = tensor[:, half:].reshape(
        experts, half // granularity, granularity, *rest
    )
    return (
        torch.stack((gate, up), dim=2)
        .reshape(experts, rows, *rest)
        .contiguous()
    )


def scale_to_tile_major(scale: torch.Tensor) -> torch.Tensor:
    experts, rows, groups = scale.shape
    return (
        scale.view(experts, rows // 256, 256, groups // 4, 4)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .repeat_interleave(2, dim=-1)
        .contiguous()
    )


def fuse_packed_and_scale(
    packed: torch.Tensor, scale_tile_major: torch.Tensor
) -> torch.Tensor:
    experts, rows, packed_k = packed.shape
    k_blocks = packed_k // 64
    fused = torch.zeros(
        (experts, rows // 256, k_blocks, 256, 80),
        dtype=torch.uint8,
        device=packed.device,
    )
    fused[..., :64] = (
        packed.view(experts, rows // 256, 256, k_blocks, 64)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    fused[..., 64:72] = scale_tile_major
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(experts, rows, k_blocks * 80)
        .contiguous()
    )


def braid_mode2_signs(fused_weight: torch.Tensor) -> torch.Tensor:
    experts, rows, storage_k = fused_weight.shape
    fused_rows = fused_weight.view(experts, rows, storage_k // 80, 80).clone()
    packed = fused_rows[..., :64].view(
        experts, rows, storage_k // 80, 16, 4
    )
    codes = torch.cat(((packed >> 4) & 0x0F, packed & 0x0F), dim=-1)
    magnitudes = codes & 0x07
    signs = codes >> 3
    braided_signs = torch.stack(
        (
            signs[..., 4],
            signs[..., 0],
            signs[..., 5],
            signs[..., 1],
            signs[..., 6],
            signs[..., 2],
            signs[..., 7],
            signs[..., 3],
        ),
        dim=-1,
    )
    braided = magnitudes | (braided_signs << 3)
    fused_rows[..., :64] = (
        braided[..., 0::2] | (braided[..., 1::2] << 4)
    ).reshape(experts, rows, storage_k // 80, 64)
    return fused_rows.view(experts, rows, storage_k).contiguous()


def dequant_marlin(packed: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    rows, half_k = packed.shape
    chunks = packed.view(rows, half_k // 4, 4)
    nibble = torch.cat((chunks >> 4, chunks & 0x0F), dim=-1).reshape(
        rows, half_k * 2
    )
    fp4 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    value = fp4[(nibble & 7).long()]
    value = torch.where((nibble & 8).bool(), -value, value)
    return value * torch.exp2(
        (exponent.int() - 127).float()
    ).repeat_interleave(32, dim=1)


def marlin_to_legacy_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Repack canonical Marlin K8 codes for the proven RS operand mapping."""
    *leading, half_k = weight.shape
    chunks = weight.view(*leading, half_k // 4, 4)
    logical = torch.cat((chunks >> 4, chunks & 0x0F), dim=-1).reshape(
        *leading, half_k * 2
    )
    groups = logical.view(*leading, half_k // 16, 32)
    return (
        groups[..., :16] | (groups[..., 16:] << 4)
    ).reshape_as(weight).contiguous()


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

using namespace deep_gemm;

__device__ __forceinline__ cute::GmmaDescriptor probe_desc_128b(
        const void* pointer) {
    cute::GmmaDescriptor descriptor;
    descriptor.bitfield.start_address_ =
        static_cast<uint32_t>(__cvta_generic_to_shared(pointer)) >> 4;
    descriptor.bitfield.layout_type_ = 1;
    descriptor.bitfield.leading_byte_offset_ = 0;
    descriptor.bitfield.stride_byte_offset_ = 64;
    descriptor.bitfield.base_offset_ = 0;
    return descriptor;
}

__global__ __launch_bounds__(128) void direct_decode_kernel(
        const uint8_t* __restrict__ packed,
        uint8_t* __restrict__ decoded) {
    __shared__ uint2 lut[deep_gemm::mxfp4::kE8M0LutCount];
    const uint32_t row = threadIdx.x;
    lut[row] = deep_gemm::mxfp4::load_e2m1_e8m0_lut(
        row + deep_gemm::mxfp4::kE8M0LutBase);
    __syncthreads();
    const uint8_t* row_ptr = packed + row * 80u;
    #pragma unroll
    for (uint32_t word = 0; word < 16; ++word) {
        const uint32_t packed_word =
            *reinterpret_cast<const uint32_t*>(row_ptr + word * 4u);
        const uint32_t exponent = row_ptr[64u + (word >> 2) * 2u];
        const uint2 value = deep_gemm::mxfp4::dequant_mode2_nibble_word(
            packed_word,
            lut[deep_gemm::mxfp4::e8m0_lut_index(exponent)]);
        *reinterpret_cast<uint2*>(decoded + row * 128u + word * 8u) = value;
    }
}

__global__ __launch_bounds__(128) void rs_variants_kernel(
        const uint8_t* __restrict__ packed,
        const uint8_t* __restrict__ activation,
        float* __restrict__ output) {
    __shared__ __align__(1024) uint8_t packed_s[128 * 80];
    __shared__ __align__(1024) uint8_t activation_s[8 * 128];
    __shared__ uint2 lut[deep_gemm::mxfp4::kE8M0LutCount];
    const uint32_t tid = threadIdx.x;
    const uint32_t variant = blockIdx.x;
    for (uint32_t i = tid * 16; i < 128u * 80u; i += 128u * 16u)
        *reinterpret_cast<uint4*>(packed_s + i) =
            *reinterpret_cast<const uint4*>(packed + i);
    const uint32_t token = tid >> 4;
    const uint32_t k8 = (tid & 15u) * 8u;
    *reinterpret_cast<uint2*>(
        activation_s + token * 128u + (k8 ^ ((token & 7u) << 4))) =
        *reinterpret_cast<const uint2*>(activation + token * 128u + k8);
    lut[tid] = deep_gemm::mxfp4::load_e2m1_e8m0_lut(
        tid + deep_gemm::mxfp4::kE8M0LutBase);
    __syncthreads();

    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t row0 = warp * 16u + lane / 4u;
    const uint32_t row1 = row0 + 8u;
    const uint32_t packed_offset = (lane & 3u) * 4u;
    float tile[2][4] = {};
    #pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        #pragma unroll
        for (uint32_t group = 0; group < 2; ++group) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                ptx::warpgroup_fence_operand(tile[group][i]);
        }
        ptx::warpgroup_arrive();
        const auto desc = probe_desc_128b(activation_s + k * 32u);
        #pragma unroll
        for (uint32_t group = 0; group < 2; ++group) {
            const uint8_t* row_ptr0 =
                packed_s + (group * 64u + row0) * 80u;
            const uint8_t* row_ptr1 =
                packed_s + (group * 64u + row1) * 80u;
            const uint32_t packed0 =
                *reinterpret_cast<const uint32_t*>(
                    row_ptr0 + k * 16u + packed_offset);
            const uint32_t packed1 =
                *reinterpret_cast<const uint32_t*>(
                    row_ptr1 + k * 16u + packed_offset);
            const uint2 value0 = deep_gemm::mxfp4::dequant_mode2_nibble_word(
                packed0, lut[deep_gemm::mxfp4::e8m0_lut_index(
                    row_ptr0[64u + k * 2u])]);
            const uint2 value1 = deep_gemm::mxfp4::dequant_mode2_nibble_word(
                packed1, lut[deep_gemm::mxfp4::e8m0_lut_index(
                    row_ptr1[64u + k * 2u])]);
            uint32_t a0, a1, a2, a3;
            if (variant == 0) {
                a0 = value0.y; a1 = value1.y;
                a2 = value0.x; a3 = value1.x;
            } else if (variant == 1) {
                a0 = value0.x; a1 = value1.x;
                a2 = value0.y; a3 = value1.y;
            } else if (variant == 2) {
                a0 = value0.y; a1 = value0.x;
                a2 = value1.y; a3 = value1.x;
            } else if (variant == 3) {
                a0 = value0.x; a1 = value0.y;
                a2 = value1.x; a3 = value1.y;
            } else if (variant == 4) {
                a0 = value1.y; a1 = value0.y;
                a2 = value1.x; a3 = value0.x;
            } else if (variant == 5) {
                a0 = value1.x; a1 = value0.x;
                a2 = value1.y; a3 = value0.y;
            } else if (variant == 6) {
                a0 = value1.y; a1 = value1.x;
                a2 = value0.y; a3 = value0.x;
            } else {
                a0 = value1.x; a1 = value1.y;
                a2 = value0.x; a3 = value0.y;
            }
            cute::SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<>::fma(
                a0, a1, a2, a3, desc,
                tile[group][0], tile[group][1],
                tile[group][2], tile[group][3],
                cute::SM90::GMMA::ScaleOut::One);
        }
        ptx::warpgroup_commit_batch();
        #pragma unroll
        for (uint32_t group = 0; group < 2; ++group) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                ptx::warpgroup_fence_operand(tile[group][i]);
        }
        ptx::warpgroup_wait<0>();
    }

    const uint32_t token0 = (lane & 3u) * 2u;
    const uint32_t token1 = token0 + 1u;
    float* variant_output = output + variant * 8u * 128u;
    #pragma unroll
    for (uint32_t group = 0; group < 2; ++group) {
        const uint32_t n0 = group * 64u + row0;
        const uint32_t n1 = group * 64u + row1;
        variant_output[token0 * 128u + n0] = tile[group][0];
        variant_output[token1 * 128u + n0] = tile[group][1];
        variant_output[token0 * 128u + n1] = tile[group][2];
        variant_output[token1 * 128u + n1] = tile[group][3];
    }
}

std::vector<torch::Tensor> run_probe(
        torch::Tensor packed, torch::Tensor activation) {
    TORCH_CHECK(packed.is_cuda() && packed.is_contiguous()
                    && packed.scalar_type() == torch::kUInt8
                    && packed.sizes() == torch::IntArrayRef({128, 80}),
                "packed must be uint8 [128,80]");
    TORCH_CHECK(activation.is_cuda() && activation.is_contiguous()
                    && activation.scalar_type() == torch::kUInt8
                    && activation.sizes() == torch::IntArrayRef({8, 128}),
                "activation must be uint8 [8,128]");
    auto decoded = torch::empty({128, 128}, packed.options());
    auto output = torch::empty(
        {8, 8, 128},
        torch::TensorOptions().dtype(torch::kFloat32).device(packed.device()));
    const auto stream = at::cuda::getCurrentCUDAStream();
    direct_decode_kernel<<<1, 128, 0, stream>>>(
        packed.data_ptr<uint8_t>(), decoded.data_ptr<uint8_t>());
    rs_variants_kernel<<<8, 128, 0, stream>>>(
        packed.data_ptr<uint8_t>(), activation.data_ptr<uint8_t>(),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {decoded, output};
}
"""

CPP = r"""
std::vector<torch::Tensor> run_probe(
    torch::Tensor packed, torch::Tensor activation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_probe", &run_probe);
}
"""


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual64 = actual.double().flatten()
    reference64 = reference.double().flatten()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual64, reference64, dim=0
            ).item()
        ),
        "max_abs": float((actual64 - reference64).abs().max().item()),
        "rel_l2": float(
            (
                torch.linalg.vector_norm(actual64 - reference64)
                / torch.linalg.vector_norm(reference64).clamp_min(1e-40)
            ).item()
        ),
    }


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(20260904)
    raw_weight = torch.randint(
        0, 256, (1, 1024, 2048), dtype=torch.uint8, device=device
    )
    raw_scale = torch.randint(
        125, 129, (1, 1024, 128), dtype=torch.uint8, device=device
    )
    interleaved_weight = interleave_l1(raw_weight)
    interleaved_scale = interleave_l1(raw_scale)
    native_weight = braid_mode2_signs(
        fuse_packed_and_scale(
            marlin_to_legacy_mxfp4(interleaved_weight),
            scale_to_tile_major(interleaved_scale),
        )
    )
    packed_tile = native_weight[0, :128, :80].contiguous()
    activation = (torch.randn((8, 128), device=device) * 0.1).to(
        torch.float8_e4m3fn
    )

    extension = load_inline(
        name="v4_native_rs_tile_probe_v1",
        cpp_sources=CPP,
        cuda_sources=CUDA,
        functions=None,
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
        ],
        extra_include_paths=[str(DEEP_GEMM_INCLUDE)],
        with_cuda=True,
        verbose=False,
    )
    decoded_u8, variants = extension.run_probe(
        packed_tile, activation.view(torch.uint8)
    )
    torch.cuda.synchronize()

    reference_weight = dequant_marlin(
        interleaved_weight[0, :128, :64],
        interleaved_scale[0, :128, :4],
    ).to(torch.float8_e4m3fn)
    reference = activation.float() @ reference_weight.float().T
    print(
        "NATIVE_RS_TILE_RESULT "
        + json.dumps(
            {
                "decode_mismatch_bytes": int(
                    (
                        decoded_u8
                        != reference_weight.view(torch.uint8)
                    ).sum().item()
                ),
                "variants": [
                    metrics(variants[index], reference) for index in range(8)
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
