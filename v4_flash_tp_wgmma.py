"""Route-aware DeepSeek-V4-Flash TP MXFP4 kernels for Hopper.

This is a direct evolution of ``step_e_lutg.py`` / ``step_e_fc2.py``.  It
retains their validated braided MXFP4 -> FP8 register dequantization and
swap-AB RS-WGMMA core, while replacing the synthetic shared ``X[8, K]`` and
raw ``G`` knob with SGLang-compatible indexed-MoE metadata.

The module implements only the per-rank local expert computation.  The
benchmark owns route alignment, activation quantization, and SGLang
``CustomAllReduceV2`` so those operations can all be captured in one graph.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


W13_SPLIT_K = int(os.environ.get("V4_W13_SPLIT_K", "2"))
if W13_SPLIT_K not in (1, 2, 4, 8):
    raise ValueError("V4_W13_SPLIT_K must be one of 1,2,4,8")
WOUT = int(os.environ.get("V4_WOUT", "128"))
if WOUT not in (64, 128, 256):
    raise ValueError("V4_WOUT must be one of 64,128,256")
W2_ROUTE_OUTPUT = os.environ.get("V4_W2_ROUTE_OUTPUT", "1") == "1"

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext_v4_tp")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

_include_candidates = (
    Path("/lustre/raplab/client/xutingz/fac/DeepGEMM/deep_gemm/include"),
    Path("/home/xutingz/fac/DeepGEMM/deep_gemm/include"),
)
DEEP_GEMM_INCLUDE = next((path for path in _include_candidates if path.exists()), None)
if DEEP_GEMM_INCLUDE is None:
    raise FileNotFoundError("Cannot locate the read-only DeepGEMM include tree")


_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cutlass/arch/barrier.h>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/mxfp4_dequant.cuh>

using namespace deep_gemm;

static constexpr int kWout = K_WOUT;
static_assert(kWout == 64 || kWout == 128 || kWout == 256);
static constexpr int kWgmmaGroups = kWout / 64;
static constexpr int kTok = 8;
static constexpr int kTopK = 6;
static constexpr int kBlockK = 128;
static constexpr int kStages = 2;
static constexpr float kRoutedScale = 1.5f;
static constexpr bool kW2RouteOutput = K_W2_ROUTE_OUTPUT;

__device__ __forceinline__ void mbar_init(uint32_t address) {
    asm volatile("mbarrier.init.shared.b64 [%0],1;" :: "r"(address));
}

__device__ __forceinline__ void mbar_wait(uint32_t address, uint32_t phase) {
    asm volatile(
        "{.reg .pred p; L_wait: mbarrier.try_wait.parity.shared.b64 "
        "p,[%0],%1; @!p bra L_wait;}"
        :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ cute::GmmaDescriptor desc_128b(uint32_t pointer) {
    cute::GmmaDescriptor descriptor;
    descriptor.bitfield.start_address_ = pointer >> 4;
    descriptor.bitfield.layout_type_ = 1;
    descriptor.bitfield.leading_byte_offset_ = 0;
    descriptor.bitfield.stride_byte_offset_ = 64;
    descriptor.bitfield.base_offset_ = 0;
    return descriptor;
}

template <int K, int N, int SplitK, bool IsW13>
__global__ __launch_bounds__(128) void route_gemm(
        const __grid_constant__ CUtensorMap tma_weight,
        const uint8_t* __restrict__ weight_scale,
        const uint8_t* __restrict__ activation,
        const float* __restrict__ activation_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ num_tokens_padded,
        const float* __restrict__ topk_weights,
        float* __restrict__ output,
        const uint2* __restrict__ global_lut,
        int max_routes) {
    static_assert(K % kBlockK == 0);
    static_assert(N % kWout == 0);
    static_assert((K / kBlockK) % SplitK == 0);
    constexpr int kNumKTiles = K / kBlockK;
    constexpr int kKTilesPerSplit = kNumKTiles / SplitK;
    constexpr int kWeightStageBytes = kWout * (kBlockK / 2);
    constexpr int kNumNTiles = N / kWout;

    const int split_idx = blockIdx.x % SplitK;
    const int task_idx = blockIdx.x / SplitK;
    const int m_block_idx = task_idx / kNumNTiles;
    const int n_block_idx = task_idx % kNumNTiles;
    if (m_block_idx * kTok >= __ldg(num_tokens_padded))
        return;

    const int expert_idx = __ldg(expert_ids + m_block_idx);
    if (expert_idx < 0)
        return;
    const int weight_row = expert_idx * N + n_block_idx * kWout;
    const int kt_begin = split_idx * kKTilesPerSplit;

    extern __shared__ __align__(1024) uint8_t dynamic_smem[];
    uint8_t* weight_smem = dynamic_smem;
    uint8_t* activation_smem = weight_smem + kStages * kWeightStageBytes;
    const uint32_t weight_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(weight_smem));
    const uint32_t activation_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(activation_smem));

    __shared__ __align__(8) uint64_t full_barriers[kStages];
    __shared__ uint2 lut_smem[256];
    __shared__ uint8_t scale_smem[kWout * (kBlockK / 32)];
    __shared__ float activation_scale_smem[kTok];
    __shared__ int32_t route_ids[kTok];
    __shared__ int32_t activation_rows[kTok];

    const int tid = threadIdx.x;
    if (tid < kTok) {
        const int route = __ldg(sorted_ids + m_block_idx * kTok + tid);
        route_ids[tid] = route;
        activation_rows[tid] = route < max_routes
            ? (IsW13 ? route / kTopK : route)
            : -1;
    }
    for (int i = tid; i < 256; i += blockDim.x)
        lut_smem[i] = global_lut[i];

    uint32_t barrier_addr[kStages];
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage)
        barrier_addr[stage] = static_cast<uint32_t>(
            __cvta_generic_to_shared(&full_barriers[stage]));
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage)
            mbar_init(barrier_addr[stage]);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const auto load_weight_stage = [&](int local_kt, int stage) {
        if (tid == 0) {
            const int global_kt = kt_begin + local_kt;
            const uint32_t dst =
                weight_smem_addr + stage * kWeightStageBytes;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                :: "r"(barrier_addr[stage]), "n"(kWeightStageBytes));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                :: "r"(dst), "l"(&tma_weight),
                   "r"(global_kt * (kBlockK / 2)), "r"(weight_row),
                   "r"(barrier_addr[stage]) : "memory");
        }
    };

    #pragma unroll
    for (int stage = 0; stage < kStages && stage < kKTilesPerSplit; ++stage)
        load_weight_stage(stage, stage);

    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row0 = warp * 16 + lane / 4;
    const int row1 = row0 + 8;
    const int packed_k_offset = (lane % 4) * 4;
    const int column_base = (lane % 4) * 2;
    float accum[kWgmmaGroups][4] = {};

    for (int local_kt = 0; local_kt < kKTilesPerSplit; ++local_kt) {
        const int stage = local_kt % kStages;
        const int global_kt = kt_begin + local_kt;

        // One uint2 per thread covers the complete 8x128 activation tile.
        const int token_slot = tid / 16;
        const int k8 = (tid % 16) * 8;
        uint2 value = make_uint2(0, 0);
        const int activation_row = activation_rows[token_slot];
        if (activation_row >= 0) {
            value = *reinterpret_cast<const uint2*>(
                activation + static_cast<int64_t>(activation_row) * K
                + global_kt * kBlockK + k8);
        }
        *reinterpret_cast<uint2*>(
            activation_smem + token_slot * kBlockK
            + (k8 ^ ((token_slot & 7) << 4))) = value;

        if (tid < kTok) {
            const int row = activation_rows[tid];
            activation_scale_smem[tid] = row >= 0
                ? __ldg(activation_scale + static_cast<int64_t>(row) * kNumKTiles
                        + global_kt)
                : 0.0f;
        }
        for (int i = tid; i < kWout * (kBlockK / 32); i += blockDim.x) {
            const int local_n = i >> 2;
            const int k_group = i & 3;
            scale_smem[i] = __ldg(
                weight_scale
                + static_cast<int64_t>(weight_row + local_n) * (K / 32)
                + global_kt * 4 + k_group);
        }

        mbar_wait(barrier_addr[stage], (local_kt / kStages) & 1u);
        asm volatile("bar.sync 0;" ::: "memory");

        float tile[kWgmmaGroups][4] = {};
        #pragma unroll
        for (int k_step = 0; k_step < kBlockK / 32; ++k_step) {
            const auto activation_desc = desc_128b(
                activation_smem_addr + k_step * 32);
            #pragma unroll
            for (int group = 0; group < kWgmmaGroups; ++group) {
                #pragma unroll
                for (int value = 0; value < 4; ++value)
                    ptx::warpgroup_fence_operand(tile[group][value]);
            }
            ptx::warpgroup_arrive();
            #pragma unroll
            for (int group = 0; group < kWgmmaGroups; ++group) {
                const int group_row0 = group * 64 + row0;
                const int group_row1 = group * 64 + row1;
                uint32_t packed0;
                uint32_t packed1;
                const uint32_t stage_base =
                    weight_smem_addr + stage * kWeightStageBytes;
                asm volatile("ld.shared.b32 %0,[%1];"
                    : "=r"(packed0)
                    : "r"(stage_base + group_row0 * (kBlockK / 2)
                          + k_step * 16 + packed_k_offset));
                asm volatile("ld.shared.b32 %0,[%1];"
                    : "=r"(packed1)
                    : "r"(stage_base + group_row1 * (kBlockK / 2)
                          + k_step * 16 + packed_k_offset));
                const uint32_t exponent0 =
                    scale_smem[group_row0 * 4 + k_step];
                const uint32_t exponent1 =
                    scale_smem[group_row1 * 4 + k_step];
                const uint2 fp8_0 =
                    mxfp4::dequant_mxfp4_to_fp8_pair_with_lut<true, true>(
                        packed0, lut_smem[exponent0]);
                const uint2 fp8_1 =
                    mxfp4::dequant_mxfp4_to_fp8_pair_with_lut<true, true>(
                        packed1, lut_smem[exponent1]);
                cute::SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<>::fma(
                    fp8_0.y, fp8_1.y, fp8_0.x, fp8_1.x,
                    activation_desc,
                    tile[group][0], tile[group][1],
                    tile[group][2], tile[group][3],
                    cute::SM90::GMMA::ScaleOut::One);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (int group = 0; group < kWgmmaGroups; ++group) {
                #pragma unroll
                for (int value = 0; value < 4; ++value)
                    ptx::warpgroup_fence_operand(tile[group][value]);
            }
            ptx::warpgroup_wait<0>();
        }
        #pragma unroll
        for (int group = 0; group < kWgmmaGroups; ++group) {
            accum[group][0] +=
                tile[group][0] * activation_scale_smem[column_base];
            accum[group][1] +=
                tile[group][1] * activation_scale_smem[column_base + 1];
            accum[group][2] +=
                tile[group][2] * activation_scale_smem[column_base];
            accum[group][3] +=
                tile[group][3] * activation_scale_smem[column_base + 1];
        }

        if (local_kt + kStages < kKTilesPerSplit)
            load_weight_stage(local_kt + kStages, stage);
    }

    const int route0 = route_ids[column_base];
    const int route1 = route_ids[column_base + 1];
    #pragma unroll
    for (int group = 0; group < kWgmmaGroups; ++group) {
        const int output_n0 = n_block_idx * kWout + group * 64 + row0;
        const int output_n1 = n_block_idx * kWout + group * 64 + row1;
        if constexpr (IsW13) {
            if (route0 < max_routes) {
                output[(static_cast<int64_t>(split_idx) * max_routes + route0) * N
                       + output_n0] = accum[group][0];
                output[(static_cast<int64_t>(split_idx) * max_routes + route0) * N
                       + output_n1] = accum[group][2];
            }
            if (route1 < max_routes) {
                output[(static_cast<int64_t>(split_idx) * max_routes + route1) * N
                       + output_n0] = accum[group][1];
                output[(static_cast<int64_t>(split_idx) * max_routes + route1) * N
                       + output_n1] = accum[group][3];
            }
        } else {
            if constexpr (kW2RouteOutput) {
                auto* route_output = reinterpret_cast<__nv_bfloat16*>(output);
                if (route0 < max_routes) {
                    route_output[static_cast<int64_t>(route0) * N + output_n0] =
                        __float2bfloat16(accum[group][0]);
                    route_output[static_cast<int64_t>(route0) * N + output_n1] =
                        __float2bfloat16(accum[group][2]);
                }
                if (route1 < max_routes) {
                    route_output[static_cast<int64_t>(route1) * N + output_n0] =
                        __float2bfloat16(accum[group][1]);
                    route_output[static_cast<int64_t>(route1) * N + output_n1] =
                        __float2bfloat16(accum[group][3]);
                }
            } else {
                if (route0 < max_routes) {
                    const int token = route0 / kTopK;
                    const float route_weight = topk_weights[route0] * kRoutedScale;
                    atomicAdd(output + static_cast<int64_t>(token) * N + output_n0,
                              route_weight * accum[group][0]);
                    atomicAdd(output + static_cast<int64_t>(token) * N + output_n1,
                              route_weight * accum[group][2]);
                }
                if (route1 < max_routes) {
                    const int token = route1 / kTopK;
                    const float route_weight = topk_weights[route1] * kRoutedScale;
                    atomicAdd(output + static_cast<int64_t>(token) * N + output_n0,
                              route_weight * accum[group][1]);
                    atomicAdd(output + static_cast<int64_t>(token) * N + output_n1,
                              route_weight * accum[group][3]);
                }
            }
        }
    }
}

template <int Intermediate, int SplitK>
__global__ void reduce_swiglu_kernel(
        const float* __restrict__ partials,
        __nv_bfloat16* __restrict__ output,
        int routes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int numel = routes * Intermediate;
    if (index >= numel)
        return;
    const int route = index / Intermediate;
    const int column = index - route * Intermediate;
    constexpr int N = 2 * Intermediate;
    float gate = 0.0f;
    float up = 0.0f;
    #pragma unroll
    for (int split = 0; split < SplitK; ++split) {
        const int64_t base =
            (static_cast<int64_t>(split) * routes + route) * N;
        gate += partials[base + column];
        up += partials[base + Intermediate + column];
    }
    // Humming emits BF16 after W13, then SGLang applies SwiGLU in BF16.
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    const float silu = gate / (1.0f + __expf(-gate));
    output[index] = __float2bfloat16(silu * up);
}

__global__ void cast_bf16_kernel(
        const float* __restrict__ input,
        __nv_bfloat16* __restrict__ output,
        int numel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
        output[index] = __float2bfloat16(input[index]);
}

CUtensorMap make_weight_desc(void* pointer, int K, int rows) {
    CUtensorMap descriptor;
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(K / 2), static_cast<cuuint64_t>(rows)};
    cuuint64_t global_strides[1] = {static_cast<cuuint64_t>(K / 2)};
    cuuint32_t box_dims[2] = {static_cast<cuuint32_t>(kBlockK / 2), kWout};
    cuuint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pointer,
        global_dims, global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
    return descriptor;
}

template <int K, int N, int SplitK, bool IsW13>
void launch_route_gemm(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int max_routes) {
    static CUtensorMap descriptor;
    static void* last_pointer = nullptr;
    if (last_pointer != weight.data_ptr()) {
        descriptor = make_weight_desc(
            weight.data_ptr(), K, weight.size(0) * weight.size(1));
        last_pointer = weight.data_ptr();
    }
    const int max_m_blocks = expert_ids.numel();
    const int grid = max_m_blocks * (N / kWout) * SplitK;
    constexpr int dynamic_smem_bytes =
        kStages * kWout * (kBlockK / 2) + kTok * kBlockK;
    const auto stream = at::cuda::getCurrentCUDAStream();
    route_gemm<K, N, SplitK, IsW13><<<
        grid, 128, dynamic_smem_bytes, stream>>>(
        descriptor,
        weight_scale.data_ptr<uint8_t>(),
        activation.data_ptr<uint8_t>(),
        activation_scale.data_ptr<float>(),
        sorted_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_padded.data_ptr<int32_t>(),
        topk_weights.numel() ? topk_weights.data_ptr<float>() : nullptr,
        static_cast<float*>(output.data_ptr()),
        reinterpret_cast<const uint2*>(lut.data_ptr<uint8_t>()),
        max_routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_w13_impl(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor partials,
        torch::Tensor lut, int intermediate) {
    const int routes = partials.size(1);
    if (intermediate == 512) {
        launch_route_gemm<4096, 1024, W13_SPLIT_K, true>(
            weight, weight_scale, activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, partials,
            partials, lut, routes);
    } else if (intermediate == 256) {
        launch_route_gemm<4096, 512, W13_SPLIT_K, true>(
            weight, weight_scale, activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, partials,
            partials, lut, routes);
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
}

void run_w2(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int intermediate) {
    const int routes = topk_weights.numel();
    if (intermediate == 512) {
        launch_route_gemm<512, 4096, 1, false>(
            weight, weight_scale, activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, topk_weights,
            output, lut, routes);
    } else if (intermediate == 256) {
        launch_route_gemm<256, 4096, 1, false>(
            weight, weight_scale, activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, topk_weights,
            output, lut, routes);
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
}

void reduce_swiglu(torch::Tensor partials, torch::Tensor output, int intermediate) {
    const int routes = partials.size(1);
    const int numel = routes * intermediate;
    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (intermediate == 512) {
        reduce_swiglu_kernel<512, W13_SPLIT_K><<<
            (numel + threads - 1) / threads, threads, 0, stream>>>(
            partials.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), routes);
    } else if (intermediate == 256) {
        reduce_swiglu_kernel<256, W13_SPLIT_K><<<
            (numel + threads - 1) / threads, threads, 0, stream>>>(
            partials.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), routes);
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cast_bf16(torch::Tensor input, torch::Tensor output) {
    const int numel = input.numel();
    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    cast_bf16_kernel<<<(numel + threads - 1) / threads, threads, 0, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_CPP = r"""
void run_w13_impl(torch::Tensor weight, torch::Tensor weight_scale,
                  torch::Tensor activation, torch::Tensor activation_scale,
                  torch::Tensor sorted_ids, torch::Tensor expert_ids,
                  torch::Tensor num_tokens_padded, torch::Tensor partials,
                  torch::Tensor lut, int intermediate);
void run_w2(torch::Tensor weight, torch::Tensor weight_scale,
            torch::Tensor activation, torch::Tensor activation_scale,
            torch::Tensor sorted_ids, torch::Tensor expert_ids,
            torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
            torch::Tensor output, torch::Tensor lut, int intermediate);
void reduce_swiglu(torch::Tensor partials, torch::Tensor output, int intermediate);
void cast_bf16(torch::Tensor input, torch::Tensor output);
"""


_ext = load_inline(
    name=(f"v4_flash_tp_wgmma_s{W13_SPLIT_K}_wo{WOUT}_"
          f"ro{int(W2_ROUTE_OUTPUT)}_v10"),
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run_w13_impl", "run_w2", "reduce_swiglu", "cast_bf16"],
    extra_cuda_cflags=[
        "-O3",
        f"-DW13_SPLIT_K={W13_SPLIT_K}",
        f"-DK_WOUT={WOUT}",
        f"-DK_W2_ROUTE_OUTPUT={int(W2_ROUTE_OUTPUT)}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-std=c++17",
        "-lineinfo",
        f"-I{DEEP_GEMM_INCLUDE}",
    ],
    extra_ldflags=["-lcuda"],
    verbose=os.environ.get("V4_VERBOSE_BUILD", "0") == "1",
)


def make_e2m1_e8m0_lut(device: torch.device | str) -> torch.Tensor:
    fp4 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    )
    exponent = torch.arange(256, dtype=torch.int32, device=device)
    scale = torch.exp2((exponent - 127).float())
    return (
        (scale[:, None] * fp4[None, :])
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )


def run_w13(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    partials: torch.Tensor,
    lut: torch.Tensor,
    intermediate: int,
) -> None:
    _ext.run_w13_impl(
        weight,
        weight_scale,
        activation,
        activation_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        partials,
        lut,
        intermediate,
    )


def run_w2(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    lut: torch.Tensor,
    intermediate: int,
) -> None:
    _ext.run_w2(
        weight,
        weight_scale,
        activation,
        activation_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        topk_weights,
        output,
        lut,
        intermediate,
    )


def reduce_swiglu(
    partials: torch.Tensor, output: torch.Tensor, intermediate: int
) -> None:
    _ext.reduce_swiglu(partials, output, intermediate)


def cast_bf16(input: torch.Tensor, output: torch.Tensor) -> None:
    _ext.cast_bf16(input, output)
