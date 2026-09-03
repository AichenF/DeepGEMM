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


W13_SPLIT_MODE = os.environ.get("V4_W13_SPLIT_K", "auto")
if W13_SPLIT_MODE not in ("auto", "2", "4"):
    raise ValueError("V4_W13_SPLIT_K must be auto, 2, or 4")
W13_MAX_SPLITS = 4


def select_w13_split_k(
    routed_rows: int, active_experts: int | None = None
) -> int:
    if W13_SPLIT_MODE != "auto":
        return int(W13_SPLIT_MODE)
    if routed_rows <= 192:
        return 4
    return 4 if active_experts is not None and active_experts <= 96 else 2
WOUT = int(os.environ.get("V4_WOUT", "128"))
if WOUT not in (64, 128, 256):
    raise ValueError("V4_WOUT must be one of 64,128,256")
LUT_ROWS = int(os.environ.get("V4_LUT_ROWS", "256"))
if LUT_ROWS not in (128, 256):
    raise ValueError("V4_LUT_ROWS must be 128 or 256")
SCALE_QUAD_REUSE = int(os.environ.get("V4_SCALE_QUAD_REUSE", "4"))
if SCALE_QUAD_REUSE not in (1, 4):
    raise ValueError("V4_SCALE_QUAD_REUSE must be 1 or 4")
SCALE_BUFFERS = int(os.environ.get("V4_SCALE_BUFFERS", "2"))
if SCALE_BUFFERS not in (1, 2):
    raise ValueError("V4_SCALE_BUFFERS must be 1 or 2")
if SCALE_QUAD_REUSE == 1 and SCALE_BUFFERS != 2:
    raise ValueError("V4_SCALE_QUAD_REUSE=1 requires V4_SCALE_BUFFERS=2")
WEIGHT_STAGES = int(os.environ.get("V4_WEIGHT_STAGES", "2"))
if WEIGHT_STAGES not in (2, 3, 4):
    raise ValueError("V4_WEIGHT_STAGES must be 2, 3, or 4")
WEIGHT_SWIZZLE = int(os.environ.get("V4_WEIGHT_SWIZZLE", "64"))
if WEIGHT_SWIZZLE not in (0, 64):
    raise ValueError("V4_WEIGHT_SWIZZLE must be 0 or 64")
WEIGHT_COMMON_ADDRESS = os.environ.get("V4_WEIGHT_COMMON_ADDRESS", "1") == "1"
if WEIGHT_COMMON_ADDRESS and WEIGHT_SWIZZLE != 64:
    raise ValueError("V4_WEIGHT_COMMON_ADDRESS=1 requires V4_WEIGHT_SWIZZLE=64")
DEQUANT_DP4A_HI = os.environ.get("V4_DEQUANT_DP4A_HI", "1") == "1"
DEQUANT_DP4A_LO = os.environ.get("V4_DEQUANT_DP4A_LO", "1") == "1"
DEQUANT_SYNTH_LUT = os.environ.get("V4_DEQUANT_SYNTH_LUT", "0") == "1"
NORMALIZED_WEIGHT_SCALE = (
    os.environ.get("V4_NORMALIZED_WEIGHT_SCALE", "1") == "1"
)
TILED_WEIGHT_LAYOUT = os.environ.get("V4_TILED_WEIGHT_LAYOUT", "1") == "1"
BULK_WEIGHT_COPY = os.environ.get("V4_BULK_WEIGHT_COPY", "1") == "1"
if BULK_WEIGHT_COPY and not TILED_WEIGHT_LAYOUT:
    raise ValueError("V4_BULK_WEIGHT_COPY requires tiled weight layout")
MODE2_BRAID = os.environ.get("V4_MODE2_BRAID", "1") == "1"
FUSED_ACT_QUANT = os.environ.get("V4_FUSED_ACT_QUANT", "1") == "1"
ACT_QUANT_PAIR = os.environ.get("V4_ACT_QUANT_PAIR", "0") == "1"
FUSED_ROUTE_QUANT = os.environ.get("V4_FUSED_ROUTE_QUANT", "1") == "1"
W2_ROUTE_OUTPUT = os.environ.get("V4_W2_ROUTE_OUTPUT", "1") == "1"
W2_GLOBAL_LUT = os.environ.get("V4_W2_GLOBAL_LUT", "0") == "1"
W2_S2R_PREFETCH = os.environ.get("V4_W2_S2R_PREFETCH", "1") == "1"
W13_S2R_PREFETCH = os.environ.get("V4_W13_S2R_PREFETCH", "1") == "1"
if W2_S2R_PREFETCH and (DEQUANT_SYNTH_LUT or W2_GLOBAL_LUT):
    raise ValueError(
        "V4_W2_S2R_PREFETCH currently probes only the shared-LUT path"
    )
if W13_S2R_PREFETCH and DEQUANT_SYNTH_LUT:
    raise ValueError(
        "V4_W13_S2R_PREFETCH currently probes only the shared-LUT path"
    )
MIN_BLOCKS_PER_SM = int(os.environ.get("V4_MIN_BLOCKS_PER_SM", "0"))
if MIN_BLOCKS_PER_SM not in (0, 8, 10, 12, 14, 16):
    raise ValueError("V4_MIN_BLOCKS_PER_SM must be one of 0,8,10,12,14,16")

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
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>

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
static constexpr int kLutRows = K_LUT_ROWS;
static_assert(kLutRows == 128 || kLutRows == 256);
static constexpr int kScaleQuadReuse = K_SCALE_QUAD_REUSE;
static_assert(kScaleQuadReuse == 1 || kScaleQuadReuse == 4);
static constexpr int kScaleBuffers = K_SCALE_BUFFERS;
static_assert(kScaleBuffers == 1 || kScaleBuffers == 2);
static_assert(kScaleQuadReuse == 4 || kScaleBuffers == 2);
static constexpr int kWeightSwizzle = K_WEIGHT_SWIZZLE;
static_assert(kWeightSwizzle == 0 || kWeightSwizzle == 64);
static constexpr bool kWeightCommonAddress = K_WEIGHT_COMMON_ADDRESS;
static_assert(!kWeightCommonAddress || kWeightSwizzle == 64);
static constexpr bool kDequantDp4aHi = K_DEQUANT_DP4A_HI;
static constexpr bool kDequantDp4aLo = K_DEQUANT_DP4A_LO;
static constexpr bool kDequantSynthLut = K_DEQUANT_SYNTH_LUT;
static constexpr bool kNormalizedWeightScale = K_NORMALIZED_WEIGHT_SCALE;
static constexpr bool kTiledWeightLayout = K_TILED_WEIGHT_LAYOUT;
static constexpr bool kBulkWeightCopy = K_BULK_WEIGHT_COPY;
static constexpr bool kMode2Braid = K_MODE2_BRAID;
static constexpr bool kW2GlobalLut = K_W2_GLOBAL_LUT;
static constexpr bool kW2S2RPrefetch = K_W2_S2R_PREFETCH;
static constexpr bool kW13S2RPrefetch = K_W13_S2R_PREFETCH;
static constexpr int kTok = 8;
static constexpr int kTopK = 6;
static constexpr int kBlockK = 128;
static constexpr int kStages = K_WEIGHT_STAGES;
static_assert(kStages == 2 || kStages == 3 || kStages == 4);
static constexpr float kRoutedScale = 1.5f;
static constexpr bool kW2RouteOutput = K_W2_ROUTE_OUTPUT;

#if K_MIN_BLOCKS_PER_SM > 0
#define ROUTE_LAUNCH_BOUNDS __launch_bounds__(128, K_MIN_BLOCKS_PER_SM)
#else
#define ROUTE_LAUNCH_BOUNDS __launch_bounds__(128)
#endif

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

__device__ __forceinline__ uint32_t scale_lut_index(uint32_t exponent) {
    if constexpr (kLutRows == 128)
        return mxfp4::e8m0_lut_index(exponent);
    return exponent;
}

// For E8M0 codes 125..128 used by the isolated performance probe, all eight
// scaled E2M1 magnitudes are normal finite E4M3 values.  Their packed bytes are
// affine in the exponent and can be synthesized without a shared-memory LUT.
// A production default requires offline scale normalization before this path
// can safely cover arbitrary model scales.
__device__ __forceinline__ uint2 synth_e2m1_e8m0_lut(uint32_t exponent) {
    const uint32_t offset = exponent - 121u;
    return make_uint2(
        offset * 0x08080800u + 0x0c080000u,
        offset * 0x08080808u + 0x1c181410u);
}

__device__ __forceinline__ uint2 synth_normalized_e2m1_lut(
        uint32_t exponent_offset) {
    return make_uint2(
        exponent_offset * 0x08080800u + 0x0c080000u,
        exponent_offset * 0x08080808u + 0x1c181410u);
}

__device__ __forceinline__ uint2 dequant_mode2_braided_word(
        uint32_t packed, const uint2& lut) {
    const uint32_t selector0 = packed & 0x00007777u;
    const uint32_t selector1 = (packed >> 16) & 0x00007777u;
    uint32_t out0 = mxfp4::byte_perm_unchecked(lut.x, lut.y, selector0);
    uint32_t out1 = mxfp4::byte_perm_unchecked(lut.x, lut.y, selector1);
    out0 |= packed & 0x80808080u;
    out1 |= (packed << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

template <bool kUseMode2>
__device__ __forceinline__ uint2 dequant_weight_word(
        uint32_t packed, const uint2& lut) {
    if constexpr (kUseMode2)
        return dequant_mode2_braided_word(packed, lut);
    return mxfp4::dequant_mxfp4_to_fp8_pair_with_lut<
        kDequantDp4aHi, kDequantDp4aLo>(packed, lut);
}

template <int K, int N, int SplitK, bool IsW13>
__global__ ROUTE_LAUNCH_BOUNDS void route_gemm(
        const __grid_constant__ CUtensorMap tma_weight,
        const __grid_constant__ CUtensorMap tma_weight_scale,
        const uint8_t* __restrict__ weight,
        const uint8_t* __restrict__ weight_scale,
        const float* __restrict__ weight_global_scale,
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
    // TMA requires the contiguous box dimension to span at least 16 bytes.
    // Fetch four adjacent K128 scale quads per row and consume the first quad.
    // The extra bytes are no worse than the sector overfetch of scalar loads.
    constexpr int kScaleRowBytes = 16;
    constexpr int kScaleStageBytes = kWout * kScaleRowBytes;
    constexpr bool kUseTmaScale = K >= 512;
    constexpr int kEffectiveScaleBuffers =
        kUseTmaScale ? kScaleBuffers : kStages;
    constexpr int kNumNTiles = N / kWout;
    constexpr bool kS2RPrefetch =
        IsW13 ? kW13S2RPrefetch : kW2S2RPrefetch;

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
    uint8_t* weight_scale_smem =
        weight_smem + kStages * kWeightStageBytes;
    uint8_t* activation_smem =
        weight_scale_smem + kEffectiveScaleBuffers * kScaleStageBytes;
    const uint32_t weight_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(weight_smem));
    const uint32_t weight_scale_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(weight_scale_smem));
    const uint32_t activation_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(activation_smem));
    const int weight_swizzle_row_offset =
        kWeightSwizzle == 64 ? ((weight_smem_addr >> 7) & 3) : 0;

    __shared__ __align__(8) uint64_t full_barriers[kStages];
    __shared__ __align__(8) uint64_t scale_barrier;
    __shared__ uint2 lut_smem[
        (kNormalizedWeightScale || kDequantSynthLut
         || (!IsW13 && kW2GlobalLut)) ? 1 : kLutRows];
    __shared__ float activation_scale_smem[kTok];
    __shared__ float expert_weight_scale;
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
    if (tid == 0) {
        expert_weight_scale = kNormalizedWeightScale
            ? __ldg(weight_global_scale + expert_idx)
            : 1.0f;
    }
    if constexpr (!kNormalizedWeightScale && !kDequantSynthLut
                  && (IsW13 || !kW2GlobalLut)) {
        for (int i = tid; i < kLutRows; i += blockDim.x) {
            constexpr int kGlobalLutOffset =
                kLutRows == 128 ? mxfp4::kE8M0LutBase : 0;
            lut_smem[i] = global_lut[kGlobalLutOffset + i];
        }
    }

    uint32_t barrier_addr[kStages];
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage)
        barrier_addr[stage] = static_cast<uint32_t>(
            __cvta_generic_to_shared(&full_barriers[stage]));
    const uint32_t scale_barrier_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&scale_barrier));
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage)
            mbar_init(barrier_addr[stage]);
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1)
            mbar_init(scale_barrier_addr);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const auto load_weight_stage = [&](int local_kt, int stage) {
        if (tid == 0) {
            const int global_kt = kt_begin + local_kt;
            const uint32_t weight_dst =
                weight_smem_addr + stage * kWeightStageBytes;
            bool load_scale = kUseTmaScale && kScaleBuffers == 2;
            int scale_kt = global_kt;
            int scale_stage = stage;
            if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                          && kScaleBuffers == 2) {
                // One 16-byte scale row covers four K128 tiles.  Load the
                // first quartet with tile 0; prefetch each following quartet
                // alongside the preceding quartet's final weight tile.
                const bool first_quartet = local_kt == 0;
                const bool prefetch_next =
                    (local_kt & 3) == 3 && local_kt + 1 < kKTilesPerSplit;
                load_scale = first_quartet || prefetch_next;
                scale_kt = first_quartet ? global_kt : global_kt + 1;
                scale_stage = (scale_kt >> 2) & 1;
            }
            const uint32_t scale_dst =
                weight_scale_smem_addr + scale_stage * kScaleStageBytes;
            if (load_scale) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(barrier_addr[stage]),
                       "n"(kWeightStageBytes + kScaleStageBytes));
            } else {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(barrier_addr[stage]), "n"(kWeightStageBytes));
            }
            if constexpr (kBulkWeightCopy) {
                const int64_t tile =
                    (static_cast<int64_t>(expert_idx) * kNumNTiles
                     + n_block_idx) * kNumKTiles + global_kt;
                const uint8_t* weight_src =
                    weight + tile * kWeightStageBytes;
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1],%2,[%3];"
                    :: "r"(weight_dst), "l"(weight_src),
                       "r"(kWeightStageBytes), "r"(barrier_addr[stage])
                    : "memory");
            } else if constexpr (kTiledWeightLayout) {
                const int tiled_row =
                    ((expert_idx * kNumNTiles + n_block_idx) * kNumKTiles
                     + global_kt) * kWout;
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                    :: "r"(weight_dst), "l"(&tma_weight),
                       "r"(0), "r"(tiled_row),
                       "r"(barrier_addr[stage]) : "memory");
            } else {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                    :: "r"(weight_dst), "l"(&tma_weight),
                       "r"(global_kt * (kBlockK / 2)), "r"(weight_row),
                       "r"(barrier_addr[stage]) : "memory");
            }
            if (load_scale) {
                if constexpr (kBulkWeightCopy) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int64_t scale_tile =
                        (static_cast<int64_t>(expert_idx) * kNumNTiles
                         + n_block_idx) * kScaleTiles + (scale_kt >> 2);
                    const uint8_t* scale_src =
                        weight_scale + scale_tile * kScaleStageBytes;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1],%2,[%3];"
                        :: "r"(scale_dst), "l"(scale_src),
                           "r"(kScaleStageBytes), "r"(barrier_addr[stage])
                        : "memory");
                } else if constexpr (kTiledWeightLayout) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int tiled_scale_row =
                        ((expert_idx * kNumNTiles + n_block_idx) * kScaleTiles
                         + (scale_kt >> 2)) * kWout;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(scale_dst), "l"(&tma_weight_scale),
                           "r"(0), "r"(tiled_scale_row),
                           "r"(barrier_addr[stage]) : "memory");
                } else {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(scale_dst), "l"(&tma_weight_scale),
                           "r"((scale_kt & ~3) * (kBlockK / 32)),
                           "r"(weight_row),
                           "r"(barrier_addr[stage]) : "memory");
                }
            }
        }
    };

    const auto load_single_scale = [&](int global_kt) {
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1) {
            if (tid == 0) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(scale_barrier_addr), "n"(kScaleStageBytes));
                if constexpr (kBulkWeightCopy) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int64_t scale_tile =
                        (static_cast<int64_t>(expert_idx) * kNumNTiles
                         + n_block_idx) * kScaleTiles + (global_kt >> 2);
                    const uint8_t* scale_src =
                        weight_scale + scale_tile * kScaleStageBytes;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1],%2,[%3];"
                        :: "r"(weight_scale_smem_addr), "l"(scale_src),
                           "r"(kScaleStageBytes), "r"(scale_barrier_addr)
                        : "memory");
                } else if constexpr (kTiledWeightLayout) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int tiled_scale_row =
                        ((expert_idx * kNumNTiles + n_block_idx) * kScaleTiles
                         + (global_kt >> 2)) * kWout;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(weight_scale_smem_addr), "l"(&tma_weight_scale),
                           "r"(0), "r"(tiled_scale_row),
                           "r"(scale_barrier_addr) : "memory");
                } else {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(weight_scale_smem_addr), "l"(&tma_weight_scale),
                           "r"((global_kt & ~3) * (kBlockK / 32)),
                           "r"(weight_row), "r"(scale_barrier_addr) : "memory");
                }
            }
        }
    };

    load_single_scale(kt_begin);

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
        const int scale_stage =
            !kUseTmaScale
            ? stage
            : (kScaleBuffers == 1
            ? 0
            : (kScaleQuadReuse == 4
               ? ((global_kt >> 2) & 1)
               : stage));

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
                        + global_kt) * expert_weight_scale
                : 0.0f;
        }
        if constexpr (!kUseTmaScale) {
            for (int i = tid; i < kWout * 4; i += blockDim.x) {
                const int local_n = i >> 2;
                const int k_group = i & 3;
                weight_scale_smem[stage * kScaleStageBytes
                                  + local_n * kScaleRowBytes
                                  + (global_kt & 3) * 4 + k_group] = __ldg(
                    weight_scale
                    + static_cast<int64_t>(weight_row + local_n) * (K / 32)
                    + global_kt * 4 + k_group);
            }
        }
        mbar_wait(barrier_addr[stage], (local_kt / kStages) & 1u);
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1) {
            if ((local_kt & 3) == 0)
                mbar_wait(scale_barrier_addr, (local_kt >> 2) & 1u);
        }
        asm volatile("bar.sync 0;" ::: "memory");

        float tile[kWgmmaGroups][4] = {};
        uint32_t next_packed0[kWgmmaGroups];
        uint32_t next_packed1[kWgmmaGroups];
        uint2 next_weight_lut0[kWgmmaGroups];
        uint2 next_weight_lut1[kWgmmaGroups];
        #pragma unroll
        for (int k_step = 0; k_step < kBlockK / 32; ++k_step) {
            const uint32_t stage_base =
                weight_smem_addr + stage * kWeightStageBytes;
            const int common_weight_chunk = k_step ^
                (((row0 >> 1) + weight_swizzle_row_offset) & 3);
            const uint32_t common_weight_address =
                stage_base + row0 * (kBlockK / 2)
                + common_weight_chunk * 16 + packed_k_offset;
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
                const int weight_chunk0 = kWeightSwizzle == 64
                    ? (k_step ^ (((group_row0 >> 1)
                                  + weight_swizzle_row_offset) & 3))
                    : k_step;
                const int weight_chunk1 = kWeightSwizzle == 64
                    ? (k_step ^ (((group_row1 >> 1)
                                  + weight_swizzle_row_offset) & 3))
                    : k_step;
                uint32_t packed0;
                uint32_t packed1;
                uint2 weight_lut0;
                uint2 weight_lut1;
                if constexpr (kS2RPrefetch) {
                    if (k_step == 0) {
                        if constexpr (kWeightCommonAddress) {
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(packed0)
                                : "r"(common_weight_address
                                      + group * 64 * (kBlockK / 2)));
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(packed1)
                                : "r"(common_weight_address
                                      + (group * 64 + 8) * (kBlockK / 2)));
                        } else {
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(packed0)
                                : "r"(stage_base + group_row0 * (kBlockK / 2)
                                      + weight_chunk0 * 16 + packed_k_offset));
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(packed1)
                                : "r"(stage_base + group_row1 * (kBlockK / 2)
                                      + weight_chunk1 * 16 + packed_k_offset));
                        }
                        const uint32_t exponent0 =
                            weight_scale_smem[scale_stage * kScaleStageBytes
                                              + group_row0 * kScaleRowBytes
                                              + (global_kt & 3) * 4];
                        const uint32_t exponent1 =
                            weight_scale_smem[scale_stage * kScaleStageBytes
                                              + group_row1 * kScaleRowBytes
                                              + (global_kt & 3) * 4];
                        if constexpr (kNormalizedWeightScale) {
                            weight_lut0 = synth_normalized_e2m1_lut(exponent0);
                            weight_lut1 = synth_normalized_e2m1_lut(exponent1);
                        } else {
                            weight_lut0 = lut_smem[scale_lut_index(exponent0)];
                            weight_lut1 = lut_smem[scale_lut_index(exponent1)];
                        }
                    } else {
                        packed0 = next_packed0[group];
                        packed1 = next_packed1[group];
                        weight_lut0 = next_weight_lut0[group];
                        weight_lut1 = next_weight_lut1[group];
                    }
                } else {
                    if constexpr (kWeightCommonAddress) {
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed0)
                            : "r"(common_weight_address
                                  + group * 64 * (kBlockK / 2)));
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed1)
                            : "r"(common_weight_address
                                  + (group * 64 + 8) * (kBlockK / 2)));
                    } else {
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed0)
                            : "r"(stage_base + group_row0 * (kBlockK / 2)
                                  + weight_chunk0 * 16 + packed_k_offset));
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed1)
                            : "r"(stage_base + group_row1 * (kBlockK / 2)
                                  + weight_chunk1 * 16 + packed_k_offset));
                    }
                    const uint32_t exponent0 =
                        weight_scale_smem[scale_stage * kScaleStageBytes
                                          + group_row0 * kScaleRowBytes
                                          + (global_kt & 3) * 4 + k_step];
                    const uint32_t exponent1 =
                        weight_scale_smem[scale_stage * kScaleStageBytes
                                          + group_row1 * kScaleRowBytes
                                          + (global_kt & 3) * 4 + k_step];
                    if constexpr (kNormalizedWeightScale) {
                        weight_lut0 = synth_normalized_e2m1_lut(exponent0);
                        weight_lut1 = synth_normalized_e2m1_lut(exponent1);
                    } else if constexpr (kDequantSynthLut) {
                        weight_lut0 = synth_e2m1_e8m0_lut(exponent0);
                        weight_lut1 = synth_e2m1_e8m0_lut(exponent1);
                    } else if constexpr (!IsW13 && kW2GlobalLut) {
                        constexpr int kGlobalLutOffset =
                            kLutRows == 128 ? mxfp4::kE8M0LutBase : 0;
                        weight_lut0 = __ldg(
                            global_lut + kGlobalLutOffset
                            + scale_lut_index(exponent0));
                        weight_lut1 = __ldg(
                            global_lut + kGlobalLutOffset
                            + scale_lut_index(exponent1));
                    } else {
                        weight_lut0 = lut_smem[scale_lut_index(exponent0)];
                        weight_lut1 = lut_smem[scale_lut_index(exponent1)];
                    }
                }
                const uint2 fp8_0 =
                    dequant_weight_word<kMode2Braid>(packed0, weight_lut0);
                const uint2 fp8_1 =
                    dequant_weight_word<kMode2Braid>(packed1, weight_lut1);
                if constexpr (kS2RPrefetch) {
                    if (k_step + 1 < kBlockK / 32) {
                        const int next_k_step = k_step + 1;
                        const int next_common_weight_chunk = next_k_step ^
                            (((row0 >> 1) + weight_swizzle_row_offset) & 3);
                        const uint32_t next_common_weight_address =
                            stage_base + row0 * (kBlockK / 2)
                            + next_common_weight_chunk * 16 + packed_k_offset;
                        if constexpr (kWeightCommonAddress) {
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(next_packed0[group])
                                : "r"(next_common_weight_address
                                      + group * 64 * (kBlockK / 2)));
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(next_packed1[group])
                                : "r"(next_common_weight_address
                                      + (group * 64 + 8) * (kBlockK / 2)));
                        } else {
                            const int next_weight_chunk0 = kWeightSwizzle == 64
                                ? (next_k_step ^ (((group_row0 >> 1)
                                    + weight_swizzle_row_offset) & 3))
                                : next_k_step;
                            const int next_weight_chunk1 = kWeightSwizzle == 64
                                ? (next_k_step ^ (((group_row1 >> 1)
                                    + weight_swizzle_row_offset) & 3))
                                : next_k_step;
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(next_packed0[group])
                                : "r"(stage_base + group_row0 * (kBlockK / 2)
                                      + next_weight_chunk0 * 16
                                      + packed_k_offset));
                            asm volatile("ld.shared.b32 %0,[%1];"
                                : "=r"(next_packed1[group])
                                : "r"(stage_base + group_row1 * (kBlockK / 2)
                                      + next_weight_chunk1 * 16
                                      + packed_k_offset));
                        }
                        const uint32_t next_exponent0 =
                            weight_scale_smem[scale_stage * kScaleStageBytes
                                              + group_row0 * kScaleRowBytes
                                              + (global_kt & 3) * 4
                                              + next_k_step];
                        const uint32_t next_exponent1 =
                            weight_scale_smem[scale_stage * kScaleStageBytes
                                              + group_row1 * kScaleRowBytes
                                              + (global_kt & 3) * 4
                                              + next_k_step];
                        if constexpr (kNormalizedWeightScale) {
                            next_weight_lut0[group] =
                                synth_normalized_e2m1_lut(next_exponent0);
                            next_weight_lut1[group] =
                                synth_normalized_e2m1_lut(next_exponent1);
                        } else {
                            next_weight_lut0[group] =
                                lut_smem[scale_lut_index(next_exponent0)];
                            next_weight_lut1[group] =
                                lut_smem[scale_lut_index(next_exponent1)];
                        }
                    }
                }
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

        if ((local_kt & 3) == 3 && local_kt + 1 < kKTilesPerSplit)
            load_single_scale(global_kt + 1);

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

template <int Intermediate, int SplitK>
__global__ __launch_bounds__(128) void reduce_swiglu_quant_kernel(
        const float* __restrict__ partials,
        __nv_bfloat16* __restrict__ activation,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        int routes) {
    static_assert(Intermediate % 128 == 0);
    constexpr int kGroupsPerRoute = Intermediate / 128;
    const int group = blockIdx.x;
    const int route = group / kGroupsPerRoute;
    const int group_in_route = group - route * kGroupsPerRoute;
    const int column = group_in_route * 128 + threadIdx.x;
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
    // Preserve the exact public pipeline semantics: W13 and SwiGLU each emit
    // BF16 before the group-128 FP8 quantizer observes the activation.
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    const float silu = gate / (1.0f + __expf(-gate));
    const __nv_bfloat16 activation_bf16 = __float2bfloat16(silu * up);
    const float value = __bfloat162float(activation_bf16);
    const int index = route * Intermediate + column;
    if (activation != nullptr)
        activation[index] = activation_bf16;

    float absmax = fabsf(value);
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, delta));

    __shared__ float warp_max[4];
    __shared__ float group_scale;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_max[warp] = absmax;
    __syncthreads();

    if (warp == 0) {
        absmax = lane < 4 ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            absmax = fmaxf(absmax,
                           __shfl_down_sync(0xffffffffu, absmax, delta));
        if (lane == 0) {
            group_scale = fmaxf(absmax, 1.0e-30f) * (1.0f / 448.0f);
            scale[group] = group_scale;
        }
    }
    __syncthreads();
    quantized[index] = __nv_fp8_e4m3(value / group_scale).__x;
}

template <int Intermediate, int SplitK>
__global__ __launch_bounds__(256) void reduce_swiglu_quant_pair_kernel(
        const float* __restrict__ partials,
        __nv_bfloat16* __restrict__ activation,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        int routes) {
    static_assert(Intermediate % 256 == 0);
    constexpr int kGroupsPerRoute = Intermediate / 128;
    const int subgroup = threadIdx.x >> 7;
    const int group_lane = threadIdx.x & 127;
    const int group = blockIdx.x * 2 + subgroup;
    const int route = group / kGroupsPerRoute;
    const int group_in_route = group - route * kGroupsPerRoute;
    const int column = group_in_route * 128 + group_lane;
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
    gate = __bfloat162float(__float2bfloat16(gate));
    up = __bfloat162float(__float2bfloat16(up));
    const float silu = gate / (1.0f + __expf(-gate));
    const __nv_bfloat16 activation_bf16 = __float2bfloat16(silu * up);
    const float value = __bfloat162float(activation_bf16);
    const int index = route * Intermediate + column;
    if (activation != nullptr)
        activation[index] = activation_bf16;

    float absmax = fabsf(value);
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, delta));

    __shared__ float warp_max[8];
    __shared__ float group_scale[2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_max[warp] = absmax;
    __syncthreads();
    if ((warp & 3) == 0) {
        absmax = lane < 4 ? warp_max[(warp & ~3) + lane] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            absmax = fmaxf(absmax,
                           __shfl_down_sync(0xffffffffu, absmax, delta));
        if (lane == 0) {
            group_scale[subgroup] =
                fmaxf(absmax, 1.0e-30f) * (1.0f / 448.0f);
            scale[group] = group_scale[subgroup];
        }
    }
    __syncthreads();
    quantized[index] = __nv_fp8_e4m3(value / group_scale[subgroup]).__x;
}

// Fixed DeepSeek-V4-Flash TP preparation.  Route alignment and input
// quantization are independent, so place them in one launch: CTA 0 performs
// the E=256/top-k=6/block-M=8 alignment, while every 256-thread CTA quantizes
// two H=4096 group-128 slices.
__global__ __launch_bounds__(256) void fused_route_quant_kernel(
        const int32_t* __restrict__ topk_ids,
        const __nv_bfloat16* __restrict__ input,
        int32_t* __restrict__ sorted_ids,
        int32_t* __restrict__ expert_ids,
        int32_t* __restrict__ num_tokens_padded,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        int routes) {
    constexpr int kExperts = 256;
    constexpr int kGroup = 128;
    using ExpertScan = cub::BlockScan<int, 256>;
    __shared__ int counts[kExperts];
    __shared__ int cursors[kExperts];
    __shared__ int total_padded;
    __shared__ typename ExpertScan::TempStorage scan_storage;
    __shared__ float warp_max[8];
    __shared__ float group_scale[2];

    const int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        counts[tid] = 0;
        __syncthreads();

        for (int route = tid; route < routes; route += blockDim.x) {
            const int expert = __ldg(topk_ids + route);
            if (static_cast<unsigned>(expert) < kExperts)
                atomicAdd(counts + expert, 1);
        }
        __syncthreads();

        const int padded_count = (counts[tid] + 7) & ~7;
        int offset;
        ExpertScan(scan_storage).ExclusiveSum(padded_count, offset);
        cursors[tid] = offset;
        if (tid == kExperts - 1) {
            total_padded = offset + padded_count;
            *num_tokens_padded = offset + padded_count;
        }
        __syncthreads();

        const int end = offset + padded_count;
        for (int position = offset; position < end; position += 8)
            expert_ids[position >> 3] = tid;
        for (int position = tid; position < total_padded;
             position += blockDim.x)
            sorted_ids[position] = routes;
        __syncthreads();

        for (int route = tid; route < routes; route += blockDim.x) {
            const int expert = __ldg(topk_ids + route);
            if (static_cast<unsigned>(expert) < kExperts) {
                const int position = atomicAdd(cursors + expert, 1);
                sorted_ids[position] = route;
            }
        }
        __syncthreads();
    }

    const int quant_subgroup = tid >> 7;
    const int quant_group = blockIdx.x * 2 + quant_subgroup;
    const int group_lane = tid & 127;
    const float value = __bfloat162float(
        input[quant_group * kGroup + group_lane]);
    float absmax = fabsf(value);
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, delta));

    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0)
        warp_max[warp] = absmax;
    __syncthreads();
    if ((warp & 3) == 0) {
        absmax = lane < 4 ? warp_max[(warp & ~3) + lane] : 0.0f;
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            absmax = fmaxf(absmax,
                           __shfl_down_sync(0xffffffffu, absmax, delta));
        if (lane == 0) {
            group_scale[quant_subgroup] =
                fmaxf(absmax, 1.0e-30f) * (1.0f / 448.0f);
            scale[quant_group] = group_scale[quant_subgroup];
        }
    }
    __syncthreads();
    quantized[quant_group * kGroup + group_lane] =
        __nv_fp8_e4m3(value / group_scale[quant_subgroup]).__x;
}

__global__ void cast_bf16_kernel(
        const float* __restrict__ input,
        __nv_bfloat16* __restrict__ output,
        int numel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
        output[index] = __float2bfloat16(input[index]);
}

__global__ void braid_mode2_kernel(uint32_t* weight, int64_t words) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= words)
        return;
    const uint32_t packed = weight[index];
    uint32_t nibble[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        nibble[i] = (packed >> (i * 4)) & 0xfu;
    constexpr int magnitude_source[8] = {1, 3, 5, 7, 0, 2, 4, 6};
    uint32_t braided = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t code = (nibble[magnitude_source[i]] & 7u)
            | (nibble[i] & 8u);
        braided |= code << (i * 4);
    }
    weight[index] = braided;
}

CUtensorMap make_weight_desc(void* pointer, int K, int64_t elements) {
    CUtensorMap descriptor;
    const int row_bytes = kTiledWeightLayout ? kBlockK / 2 : K / 2;
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(row_bytes),
        static_cast<cuuint64_t>(elements / row_bytes)};
    cuuint64_t global_strides[1] = {static_cast<cuuint64_t>(row_bytes)};
    cuuint32_t box_dims[2] = {static_cast<cuuint32_t>(kBlockK / 2), kWout};
    cuuint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pointer,
        global_dims, global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        kWeightSwizzle == 64
            ? CU_TENSOR_MAP_SWIZZLE_64B
            : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
    return descriptor;
}

CUtensorMap make_weight_scale_desc(void* pointer, int K, int64_t elements) {
    CUtensorMap descriptor;
    const int row_bytes = kTiledWeightLayout ? 16 : K / 32;
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(row_bytes),
        static_cast<cuuint64_t>(elements / row_bytes)};
    cuuint64_t global_strides[1] = {static_cast<cuuint64_t>(row_bytes)};
    cuuint32_t box_dims[2] = {16, kWout};
    cuuint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pointer,
        global_dims, global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled(scale) failed: ", result);
    return descriptor;
}

template <int K, int N, int SplitK, bool IsW13>
void launch_route_gemm(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int max_routes) {
    static CUtensorMap weight_descriptor;
    static CUtensorMap scale_descriptor;
    static void* last_weight_pointer = nullptr;
    static void* last_scale_pointer = nullptr;
    if (last_weight_pointer != weight.data_ptr()
            || last_scale_pointer != weight_scale.data_ptr()) {
        weight_descriptor = make_weight_desc(
            weight.data_ptr(), K, weight.numel());
        if constexpr (K >= 512) {
            scale_descriptor = make_weight_scale_desc(
                weight_scale.data_ptr(), K, weight_scale.numel());
        }
        last_weight_pointer = weight.data_ptr();
        last_scale_pointer = weight_scale.data_ptr();
    }
    const int max_m_blocks = expert_ids.numel();
    const int grid = max_m_blocks * (N / kWout) * SplitK;
    constexpr int effective_scale_buffers = K >= 512 ? kScaleBuffers : kStages;
    constexpr int dynamic_smem_bytes =
        kStages * kWout * (kBlockK / 2)
        + effective_scale_buffers * kWout * 16
        + kTok * kBlockK;
    const auto stream = at::cuda::getCurrentCUDAStream();
    route_gemm<K, N, SplitK, IsW13><<<
        grid, 128, dynamic_smem_bytes, stream>>>(
        weight_descriptor,
        scale_descriptor,
        weight.data_ptr<uint8_t>(),
        weight_scale.data_ptr<uint8_t>(),
        weight_global_scale.numel()
            ? weight_global_scale.data_ptr<float>()
            : nullptr,
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
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor partials,
        torch::Tensor lut, int intermediate, int split_k) {
    const int routes = partials.size(1);
    if (intermediate == 512) {
        if (split_k == 4) {
            launch_route_gemm<4096, 1024, 4, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_route_gemm<4096, 1024, 2, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4) {
            launch_route_gemm<4096, 512, 4, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_route_gemm<4096, 512, 2, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        }
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
}

void run_w2(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int intermediate) {
    const int routes = topk_weights.numel();
    if (intermediate == 512) {
        launch_route_gemm<512, 4096, 1, false>(
            weight, weight_scale, weight_global_scale,
            activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, topk_weights,
            output, lut, routes);
    } else if (intermediate == 256) {
        launch_route_gemm<256, 4096, 1, false>(
            weight, weight_scale, weight_global_scale,
            activation, activation_scale,
            sorted_ids, expert_ids, num_tokens_padded, topk_weights,
            output, lut, routes);
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
}

template <int Intermediate, int SplitK>
void launch_reduce_swiglu(
        torch::Tensor partials, torch::Tensor output, int routes) {
    const int numel = routes * Intermediate;
    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    reduce_swiglu_kernel<Intermediate, SplitK><<<
        (numel + threads - 1) / threads, threads, 0, stream>>>(
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), routes);
}

void reduce_swiglu(
        torch::Tensor partials, torch::Tensor output,
        int intermediate, int split_k) {
    const int routes = partials.size(1);
    if (intermediate == 512) {
        if (split_k == 4)
            launch_reduce_swiglu<512, 4>(partials, output, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu<512, 2>(partials, output, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4)
            launch_reduce_swiglu<256, 4>(partials, output, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu<256, 2>(partials, output, routes);
        }
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Intermediate, int SplitK>
void launch_reduce_swiglu_quant(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale, int routes) {
    constexpr int threads = 128;
    constexpr int groups_per_route = Intermediate / 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* activation_ptr = activation.numel()
        ? reinterpret_cast<__nv_bfloat16*>(activation.data_ptr())
        : nullptr;
    reduce_swiglu_quant_kernel<Intermediate, SplitK><<<
        routes * groups_per_route, threads, 0, stream>>>(
        partials.data_ptr<float>(), activation_ptr,
        quantized.data_ptr<uint8_t>(), scale.data_ptr<float>(), routes);
}

void reduce_swiglu_quant(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale,
        int intermediate, int split_k) {
    const int routes = partials.size(1);
    if (intermediate == 512) {
        if (split_k == 4)
            launch_reduce_swiglu_quant<512, 4>(
                partials, activation, quantized, scale, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu_quant<512, 2>(
                partials, activation, quantized, scale, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4)
            launch_reduce_swiglu_quant<256, 4>(
                partials, activation, quantized, scale, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu_quant<256, 2>(
                partials, activation, quantized, scale, routes);
        }
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Intermediate, int SplitK>
void launch_reduce_swiglu_quant_pair(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale, int routes) {
    constexpr int threads = 256;
    constexpr int groups_per_route = Intermediate / 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* activation_ptr = activation.numel()
        ? reinterpret_cast<__nv_bfloat16*>(activation.data_ptr())
        : nullptr;
    reduce_swiglu_quant_pair_kernel<Intermediate, SplitK><<<
        routes * groups_per_route / 2, threads, 0, stream>>>(
        partials.data_ptr<float>(), activation_ptr,
        quantized.data_ptr<uint8_t>(), scale.data_ptr<float>(), routes);
}

void reduce_swiglu_quant_pair(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale,
        int intermediate, int split_k) {
    const int routes = partials.size(1);
    if (intermediate == 512) {
        if (split_k == 4)
            launch_reduce_swiglu_quant_pair<512, 4>(
                partials, activation, quantized, scale, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu_quant_pair<512, 2>(
                partials, activation, quantized, scale, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4)
            launch_reduce_swiglu_quant_pair<256, 4>(
                partials, activation, quantized, scale, routes);
        else {
            TORCH_CHECK(split_k == 2, "split_k must be 2 or 4");
            launch_reduce_swiglu_quant_pair<256, 2>(
                partials, activation, quantized, scale, routes);
        }
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

void fused_route_quant(
        torch::Tensor topk_ids, torch::Tensor input,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor quantized,
        torch::Tensor scale) {
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32,
                "topk_ids must be int32");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "input must be bfloat16");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096,
                "input must have shape [M,4096]");
    TORCH_CHECK(topk_ids.numel() == input.size(0) * 6,
                "topk_ids must have shape [M,6]");
    TORCH_CHECK(quantized.numel() == input.numel(),
                "quantized output shape mismatch");
    TORCH_CHECK(scale.numel() == input.size(0) * 32,
                "scale output must have shape [M,32]");
    const int routes = topk_ids.numel();
    const int blocks = input.size(0) * 16;
    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_route_quant_kernel<<<blocks, 256, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_padded.data_ptr<int32_t>(),
        quantized.data_ptr<uint8_t>(), scale.data_ptr<float>(), routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void braid_mode2(torch::Tensor weight) {
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
                "mode2 weight must be uint8");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(),
                "mode2 weight must be contiguous CUDA");
    TORCH_CHECK(weight.numel() % sizeof(uint32_t) == 0,
                "mode2 weight byte count must be divisible by four");
    const int64_t words = weight.numel() / sizeof(uint32_t);
    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    braid_mode2_kernel<<<(words + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<uint32_t*>(weight.data_ptr<uint8_t>()), words);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_CPP = r"""
void run_w13_impl(torch::Tensor weight, torch::Tensor weight_scale,
                  torch::Tensor weight_global_scale,
                  torch::Tensor activation, torch::Tensor activation_scale,
                  torch::Tensor sorted_ids, torch::Tensor expert_ids,
                  torch::Tensor num_tokens_padded, torch::Tensor partials,
                  torch::Tensor lut, int intermediate, int split_k);
void run_w2(torch::Tensor weight, torch::Tensor weight_scale,
            torch::Tensor weight_global_scale,
            torch::Tensor activation, torch::Tensor activation_scale,
            torch::Tensor sorted_ids, torch::Tensor expert_ids,
            torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
            torch::Tensor output, torch::Tensor lut, int intermediate);
void reduce_swiglu(torch::Tensor partials, torch::Tensor output,
                   int intermediate, int split_k);
void reduce_swiglu_quant(torch::Tensor partials, torch::Tensor activation,
                         torch::Tensor quantized, torch::Tensor scale,
                         int intermediate, int split_k);
void reduce_swiglu_quant_pair(torch::Tensor partials, torch::Tensor activation,
                              torch::Tensor quantized, torch::Tensor scale,
                              int intermediate, int split_k);
void cast_bf16(torch::Tensor input, torch::Tensor output);
void fused_route_quant(torch::Tensor topk_ids, torch::Tensor input,
                       torch::Tensor sorted_ids, torch::Tensor expert_ids,
                       torch::Tensor num_tokens_padded,
                       torch::Tensor quantized, torch::Tensor scale);
void braid_mode2(torch::Tensor weight);
"""


_ext = load_inline(
    name=(f"v4_flash_tp_wgmma_sdyn_wo{WOUT}_lr{LUT_ROWS}_"
          f"sr{SCALE_QUAD_REUSE}_sb{SCALE_BUFFERS}_"
          f"st{WEIGHT_STAGES}_"
          f"ws{WEIGHT_SWIZZLE}_wca{int(WEIGHT_COMMON_ADDRESS)}_"
          f"dh{int(DEQUANT_DP4A_HI)}_dl{int(DEQUANT_DP4A_LO)}_"
          f"dsl{int(DEQUANT_SYNTH_LUT)}_"
          f"nws{int(NORMALIZED_WEIGHT_SCALE)}_"
          f"twl{int(TILED_WEIGHT_LAYOUT)}_"
          f"bwc{int(BULK_WEIGHT_COPY)}_"
          f"m2{int(MODE2_BRAID)}_"
          f"ro{int(W2_ROUTE_OUTPUT)}_w2gl{int(W2_GLOBAL_LUT)}_"
          f"w2pf{int(W2_S2R_PREFETCH)}_w13pf{int(W13_S2R_PREFETCH)}_"
          f"mb{MIN_BLOCKS_PER_SM}_v43"),
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=[
        "run_w13_impl", "run_w2", "reduce_swiglu", "reduce_swiglu_quant",
        "reduce_swiglu_quant_pair",
        "cast_bf16",
        "fused_route_quant",
        "braid_mode2",
    ],
    extra_cuda_cflags=[
        "-O3",
        f"-DK_WOUT={WOUT}",
        f"-DK_LUT_ROWS={LUT_ROWS}",
        f"-DK_SCALE_QUAD_REUSE={SCALE_QUAD_REUSE}",
        f"-DK_SCALE_BUFFERS={SCALE_BUFFERS}",
        f"-DK_WEIGHT_STAGES={WEIGHT_STAGES}",
        f"-DK_WEIGHT_SWIZZLE={WEIGHT_SWIZZLE}",
        f"-DK_WEIGHT_COMMON_ADDRESS={int(WEIGHT_COMMON_ADDRESS)}",
        f"-DK_DEQUANT_DP4A_HI={int(DEQUANT_DP4A_HI)}",
        f"-DK_DEQUANT_DP4A_LO={int(DEQUANT_DP4A_LO)}",
        f"-DK_DEQUANT_SYNTH_LUT={int(DEQUANT_SYNTH_LUT)}",
        f"-DK_NORMALIZED_WEIGHT_SCALE={int(NORMALIZED_WEIGHT_SCALE)}",
        f"-DK_TILED_WEIGHT_LAYOUT={int(TILED_WEIGHT_LAYOUT)}",
        f"-DK_BULK_WEIGHT_COPY={int(BULK_WEIGHT_COPY)}",
        f"-DK_MODE2_BRAID={int(MODE2_BRAID)}",
        f"-DK_W2_GLOBAL_LUT={int(W2_GLOBAL_LUT)}",
        f"-DK_W2_S2R_PREFETCH={int(W2_S2R_PREFETCH)}",
        f"-DK_W13_S2R_PREFETCH={int(W13_S2R_PREFETCH)}",
        f"-DK_W2_ROUTE_OUTPUT={int(W2_ROUTE_OUTPUT)}",
        f"-DK_MIN_BLOCKS_PER_SM={MIN_BLOCKS_PER_SM}",
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


def normalize_mxfp4_weight_scales_(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize one layer's E8M0 groups to offsets 1..12 at model load.

    The packed-weight adjustment is the same loss-minimizing MXFP4 operation
    used by Humming when an expert spans more than eleven exponents.  It is a
    one-time checkpoint transform and is never part of graph replay timing.
    The returned FP32 expert scale includes the E4M3/FP4 exponent offset.
    """
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise TypeError("MXFP4 weight and E8M0 scale must be uint8")
    if not weight.is_cuda or not weight_scale.is_cuda:
        raise ValueError("MXFP4 normalization requires CUDA tensors")
    if not weight.is_contiguous() or not weight_scale.is_contiguous():
        raise ValueError("MXFP4 weight and scale must be contiguous")
    if weight.shape[0] != weight_scale.shape[0]:
        raise ValueError("weight and scale expert dimensions must match")

    scale_i16 = weight_scale.to(torch.int16).view(weight_scale.shape[0], -1)
    scale_max = scale_i16.amax(dim=1, keepdim=True)
    scale_min = scale_i16.amin(dim=1, keepdim=True)
    scale_base = torch.maximum(scale_min, scale_max - 11)
    clamped = torch.maximum(scale_i16, scale_base)
    delta = (clamped - scale_i16).to(torch.uint8).contiguous()

    # Import lazily: the route GEMM itself has no runtime Humming dependency.
    from humming import ops as humming_ops

    humming_ops.process_mxfp4_w4a8_weight(
        weight.view(torch.int32), delta, inplace=True
    )
    normalized = (clamped - scale_base + 1).to(torch.uint8)
    normalized = normalized.view_as(weight_scale).contiguous()
    expert_scale = torch.exp2(
        scale_base.squeeze(1).to(torch.float32) - 122.0
    ).contiguous()
    return normalized, expert_scale


def tile_mxfp4_weight_layout(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each N128 x K128 MXFP4 tile contiguously at model load."""
    if weight.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError("tiled layout expects [experts, N, packed-K] tensors")
    experts, channels, packed_k = weight.shape
    if channels % WOUT or packed_k % (128 // 2):
        raise ValueError("weight shape is not divisible by N/WOUT and K128")
    if weight_scale.shape[:2] != (experts, channels):
        raise ValueError("weight and scale leading dimensions must match")
    ntiles = channels // WOUT
    ktiles = packed_k // (128 // 2)
    tiled_weight = (
        weight.view(experts, ntiles, WOUT, ktiles, 128 // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if BULK_WEIGHT_COPY:
        # TMA's 64-byte swizzle normally happens during the tensor copy.  A
        # linear bulk copy cannot transform bytes, so materialize the same
        # physical shared-memory order in the checkpoint layout once.
        chunks = tiled_weight.view(
            experts, ntiles, ktiles, WOUT, 4, 16
        )
        physical_chunk = torch.arange(4, device=weight.device)
        row_xor = (torch.arange(WOUT, device=weight.device) >> 1) & 3
        logical_chunk = physical_chunk[None, :] ^ row_xor[:, None]
        gather_index = logical_chunk.view(1, 1, 1, WOUT, 4, 1).expand(
            experts, ntiles, ktiles, WOUT, 4, 16
        )
        tiled_weight = torch.gather(chunks, 4, gather_index).contiguous()
    if weight_scale.shape[-1] >= 16:
        if weight_scale.shape[-1] != ktiles * 4:
            raise ValueError("E8M0 scale shape does not match group size 32")
        scale_tiles = weight_scale.shape[-1] // 16
        tiled_scale = (
            weight_scale.view(experts, ntiles, WOUT, scale_tiles, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    else:
        # TP8 W2 has eight scale bytes per logical row, below TMA's 16-byte
        # minimum contiguous box; keep its existing scalar-load fallback.
        tiled_scale = weight_scale
    return tiled_weight, tiled_scale


def run_w13(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    partials: torch.Tensor,
    lut: torch.Tensor,
    intermediate: int,
    split_k: int | None = None,
) -> None:
    if split_k is None:
        split_k = select_w13_split_k(partials.size(1))
    if split_k not in (2, 4):
        raise ValueError("W13 split_k must be 2 or 4")
    _ext.run_w13_impl(
        weight,
        weight_scale,
        weight_global_scale,
        activation,
        activation_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        partials,
        lut,
        intermediate,
        split_k,
    )


def run_w2(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
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
        weight_global_scale,
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
    partials: torch.Tensor,
    output: torch.Tensor,
    intermediate: int,
    split_k: int | None = None,
) -> None:
    if split_k is None:
        split_k = select_w13_split_k(partials.size(1))
    if split_k not in (2, 4):
        raise ValueError("W13 split_k must be 2 or 4")
    _ext.reduce_swiglu(
        partials,
        output,
        intermediate,
        split_k,
    )


def reduce_swiglu_quant(
    partials: torch.Tensor,
    activation: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    intermediate: int,
    split_k: int | None = None,
) -> None:
    if split_k is None:
        split_k = select_w13_split_k(partials.size(1))
    if split_k not in (2, 4):
        raise ValueError("W13 split_k must be 2 or 4")
    reduce_impl = (
        _ext.reduce_swiglu_quant_pair
        if ACT_QUANT_PAIR
        else _ext.reduce_swiglu_quant
    )
    reduce_impl(
        partials,
        activation,
        quantized,
        scale,
        intermediate,
        split_k,
    )


def cast_bf16(input: torch.Tensor, output: torch.Tensor) -> None:
    _ext.cast_bf16(input, output)


def fused_route_quant(
    topk_ids: torch.Tensor,
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    _ext.fused_route_quant(
        topk_ids,
        input,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        quantized,
        scale,
    )


def braid_mode2_(weight: torch.Tensor) -> torch.Tensor:
    """Offline in-place Mode2 sign/magnitude braid; excluded from inference."""
    _ext.braid_mode2(weight)
    return weight
