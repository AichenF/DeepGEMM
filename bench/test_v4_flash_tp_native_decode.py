#!/usr/bin/env python3
"""Bitwise probe for the native SM90 fused-row Mode2 weight decoder."""

from __future__ import annotations

import json

import torch
from torch.utils.cpp_extension import load_inline

import v4_flash_tp_native_megamoe as native


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

__global__ void decode_mode2_tile_kernel(
        const uint8_t* __restrict__ packed,
        uint8_t* __restrict__ output) {
    __shared__ uint2 lut[deep_gemm::mxfp4::kE8M0LutCount];
    __shared__ __align__(1024) uint8_t decoded[256 * 128];
    const uint32_t tid = threadIdx.x;
    if (tid < deep_gemm::mxfp4::kE8M0LutCount) {
        lut[tid] = deep_gemm::mxfp4::load_e2m1_e8m0_lut(
            tid + deep_gemm::mxfp4::kE8M0LutBase);
    }
    __syncthreads();
    deep_gemm::mxfp4::dequant_smem_b_from_packed_mode2_nibble<false>(
        decoded, packed, tid, lut);
    __syncthreads();
    for (uint32_t index = tid; index < 256u * 128u; index += 256u) {
        const uint32_t row = index / 128u;
        const uint32_t k = index % 128u;
        output[index] = decoded[row * 128u + (k ^ ((row & 7u) << 4))];
    }
}

torch::Tensor decode_mode2_tile(torch::Tensor packed) {
    TORCH_CHECK(packed.is_cuda() && packed.is_contiguous()
                    && packed.scalar_type() == torch::kUInt8
                    && packed.sizes() == torch::IntArrayRef({256, 80}),
                "packed tile must be contiguous CUDA uint8 [256,80]");
    auto output = torch::empty({256, 128}, packed.options());
    decode_mode2_tile_kernel<<<1, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<uint8_t>(), output.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

CPP = r"""
torch::Tensor decode_mode2_tile(torch::Tensor packed);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_mode2_tile", &decode_mode2_tile);
}
"""


def dequant_raw_tile(packed: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    chunks = packed.view(256, 16, 4)
    # Marlin K8 packing: byte b contains high=K[b], low=K[b+4].
    nibble = torch.cat((chunks >> 4, chunks & 0x0F), dim=-1).reshape(256, 128)
    fp4 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    magnitude = fp4[(nibble & 7).long()]
    value = torch.where((nibble & 8).bool(), -magnitude, magnitude)
    scale = torch.exp2((exponent.int() - 127).float()).repeat_interleave(
        32, dim=1
    )
    return value * scale


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(20260904)
    w13 = torch.randint(
        0, 256, (256, 1024, 2048), dtype=torch.uint8, device=device
    )
    s13 = torch.randint(
        125, 129, (256, 1024, 128), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 256, (256, 4096, 256), dtype=torch.uint8, device=device
    )
    s2 = torch.randint(
        125, 129, (256, 4096, 16), dtype=torch.uint8, device=device
    )
    native_w13, _ = native.transform_weights(w13, s13, w2, s2)
    packed_tile = native_w13[0, :256, :80].contiguous()

    ext = load_inline(
        name="v4_native_mode2_decode_probe",
        cpp_sources=CPP,
        cuda_sources=CUDA,
        functions=None,
        extra_include_paths=[str(native.DEEP_GEMM_INCLUDE)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        with_cuda=True,
        verbose=False,
    )
    decoded_u8 = ext.decode_mode2_tile(packed_tile)
    decoded = decoded_u8.view(torch.float8_e4m3fn).float()

    w13_il = native._interleave_l1(w13)
    s13_il = native._interleave_l1(s13)
    reference = dequant_raw_tile(w13_il[0, :256, :64], s13_il[0, :256, :4])
    reference_fp8 = reference.to(torch.float8_e4m3fn)
    reference_roundtrip = reference_fp8.float()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            decoded.double().flatten(), reference_roundtrip.double().flatten(), dim=0
        ).item()
    )
    rel_l2 = float(
        (
            torch.linalg.vector_norm(decoded.double() - reference_roundtrip.double())
            / torch.linalg.vector_norm(reference_roundtrip.double()).clamp_min(1e-40)
        ).item()
    )
    print(
        "NATIVE_MODE2_DECODE "
        + json.dumps(
            {
                "cosine": cosine,
                "max_abs": float((decoded - reference_roundtrip).abs().max()),
                "mismatch_bytes": int(
                    (decoded_u8 != reference_fp8.view(torch.uint8)).sum()
                ),
                "rel_l2": rel_l2,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
