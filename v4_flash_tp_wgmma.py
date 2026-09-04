"""Route-aware DeepSeek-V4-Flash TP MXFP4 kernels for Hopper.

This is a direct evolution of ``step_e_lutg.py`` / ``step_e_fc2.py``.  It
retains their validated braided MXFP4 -> FP8 register dequantization and
swap-AB RS-WGMMA core, while replacing the synthetic shared ``X[8, K]`` and
raw ``G`` knob with SGLang-compatible indexed-MoE metadata.

The serving entry accepts an already-quantized FP8 activation and owns route
alignment, both expert GEMMs, the internal SwiGLU/FP8 requantization, fixed-k6
reduction, and the TP collective.  BF16-to-FP8 input quantization is upstream
of this module and is deliberately excluded from both compared graphs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


W13_SPLIT_MODE = os.environ.get("V4_W13_SPLIT_K", "auto")
if W13_SPLIT_MODE not in ("auto", "1", "2", "4"):
    raise ValueError("V4_W13_SPLIT_K must be auto, 1, 2, or 4")
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
NORMALIZED_SHARED_LUT = (
    os.environ.get("V4_NORMALIZED_SHARED_LUT", "0") == "1"
)
if NORMALIZED_SHARED_LUT and not NORMALIZED_WEIGHT_SCALE:
    raise ValueError(
        "V4_NORMALIZED_SHARED_LUT=1 requires normalized weight scales"
    )
ACTIVATION_EVICT_LAST = (
    os.environ.get("V4_ACTIVATION_EVICT_LAST", "0") == "1"
)
PREDICATED_PADDED_ACTIVATION = (
    os.environ.get("V4_PREDICATED_PADDED_ACTIVATION", "0") == "1"
)
TILED_WEIGHT_LAYOUT = os.environ.get("V4_TILED_WEIGHT_LAYOUT", "1") == "1"
BULK_WEIGHT_COPY = os.environ.get("V4_BULK_WEIGHT_COPY", "1") == "1"
if BULK_WEIGHT_COPY and not TILED_WEIGHT_LAYOUT:
    raise ValueError("V4_BULK_WEIGHT_COPY requires tiled weight layout")
TMA_CTA_SCOPE = os.environ.get("V4_TMA_CTA_SCOPE", "0") == "1"
if TMA_CTA_SCOPE and not BULK_WEIGHT_COPY:
    raise ValueError("V4_TMA_CTA_SCOPE requires bulk weight copy")
WEIGHT_EVICT_FIRST = os.environ.get("V4_WEIGHT_EVICT_FIRST", "1") == "1"
if WEIGHT_EVICT_FIRST and not BULK_WEIGHT_COPY:
    raise ValueError("V4_WEIGHT_EVICT_FIRST requires bulk weight copy")
WEIGHT_POLICY_HOIST = os.environ.get("V4_WEIGHT_POLICY_HOIST", "0") == "1"
if WEIGHT_POLICY_HOIST and not WEIGHT_EVICT_FIRST:
    raise ValueError("V4_WEIGHT_POLICY_HOIST requires weight evict-first")
WEIGHT_POLICY_CONSTANT = (
    os.environ.get("V4_WEIGHT_POLICY_CONSTANT", "0") == "1"
)
if WEIGHT_POLICY_CONSTANT and not WEIGHT_EVICT_FIRST:
    raise ValueError("V4_WEIGHT_POLICY_CONSTANT requires weight evict-first")
if WEIGHT_POLICY_CONSTANT and WEIGHT_POLICY_HOIST:
    raise ValueError(
        "V4_WEIGHT_POLICY_CONSTANT and V4_WEIGHT_POLICY_HOIST are exclusive"
    )
W2_NO_WEIGHT_EVICT_FIRST = (
    os.environ.get("V4_W2_NO_WEIGHT_EVICT_FIRST", "0") == "1"
)
if W2_NO_WEIGHT_EVICT_FIRST and not WEIGHT_EVICT_FIRST:
    raise ValueError(
        "V4_W2_NO_WEIGHT_EVICT_FIRST requires global weight evict-first"
    )
INTERLEAVED_BULK_COPY = (
    os.environ.get("V4_INTERLEAVED_BULK_COPY", "1") == "1"
)
if INTERLEAVED_BULK_COPY and not BULK_WEIGHT_COPY:
    raise ValueError("V4_INTERLEAVED_BULK_COPY requires bulk weight copy")
if INTERLEAVED_BULK_COPY and (
    WEIGHT_STAGES != 2 or SCALE_QUAD_REUSE != 4 or SCALE_BUFFERS != 2
):
    raise ValueError(
        "V4_INTERLEAVED_BULK_COPY requires two weight/scale stages "
        "and scale-quad reuse"
    )
COMPACT_INTERLEAVED_SCALE = (
    os.environ.get("V4_COMPACT_INTERLEAVED_SCALE", "1") == "1"
)
if COMPACT_INTERLEAVED_SCALE and not INTERLEAVED_BULK_COPY:
    raise ValueError(
        "V4_COMPACT_INTERLEAVED_SCALE requires interleaved bulk copy"
    )
MODE2_BRAID = os.environ.get("V4_MODE2_BRAID", "1") == "1"
FUSED_ACT_QUANT = os.environ.get("V4_FUSED_ACT_QUANT", "1") == "1"
# The serving MegaMoE contract starts from prequantized FP8 X.  Keep the old
# fused BF16 route+quant entry available for diagnostic scripts, but the
# production graph uses a route-only device preparation kernel.
FUSED_ROUTE_QUANT = os.environ.get("V4_FUSED_ROUTE_QUANT", "1") == "1"
FUSED_ROUTE_ALIGN = os.environ.get("V4_FUSED_ROUTE_ALIGN", "1") == "1"
W13_PAIRED_WG = os.environ.get("V4_W13_PAIRED_WG", "0") == "1"
if COMPACT_INTERLEAVED_SCALE and W13_PAIRED_WG:
    raise ValueError(
        "compact interleaved scales are not implemented for paired W13"
    )
W2_ROUTE_OUTPUT = os.environ.get("V4_W2_ROUTE_OUTPUT", "1") == "1"
W2_SORTED_ACT = os.environ.get("V4_W2_SORTED_ACT", "0") == "1"
W2_MBLOCK_SCALE = os.environ.get("V4_W2_MBLOCK_SCALE", "0") == "1"
W2_NEEDS_ROUTE_MAP = W2_SORTED_ACT or W2_MBLOCK_SCALE
W2_FOLD_GLOBAL_SCALE = (
    os.environ.get("V4_W2_FOLD_GLOBAL_SCALE", "0") == "1"
)
W2_COALESCED_STORE = (
    os.environ.get("V4_W2_COALESCED_STORE", "0") == "1"
)
if W2_COALESCED_STORE and (not W2_ROUTE_OUTPUT or WOUT != 128):
    raise ValueError(
        "V4_W2_COALESCED_STORE=1 requires route output and V4_WOUT=128"
    )
if W2_NEEDS_ROUTE_MAP and (
    not FUSED_ROUTE_ALIGN or not FUSED_ACT_QUANT or W13_PAIRED_WG
):
    raise ValueError(
        "sorted W2 activation/scale layouts require fused route/activation "
        "quantization and the split-K W13 path"
    )
if W2_FOLD_GLOBAL_SCALE and (
    not NORMALIZED_WEIGHT_SCALE or not FUSED_ACT_QUANT or W13_PAIRED_WG
):
    raise ValueError(
        "V4_W2_FOLD_GLOBAL_SCALE=1 requires normalized weights, fused "
        "activation quantization, and the split-K W13 path"
    )
TILED_K6_REDUCE_POLICY = os.environ.get("V4_TILED_K6_REDUCE_MODE", "auto")
if TILED_K6_REDUCE_POLICY not in ("auto", "0", "1", "2", "3", "4"):
    raise ValueError(
        "V4_TILED_K6_REDUCE_MODE must be auto or one of 0,1,2,3,4"
    )


def select_tiled_k6_reduce_mode(tokens: int) -> int:
    if TILED_K6_REDUCE_POLICY == "auto":
        return 4 if tokens <= 16 else 0
    return int(TILED_K6_REDUCE_POLICY)


FUSED_K6_PUSH_AR = os.environ.get("V4_FUSED_K6_PUSH_AR", "0") == "1"
FUSED_K6_MC_PUSH_AR = os.environ.get("V4_FUSED_K6_MC_PUSH_AR", "1") == "1"
FUSED_K6_MC_PUSH_MAX_M = int(
    os.environ.get("V4_FUSED_K6_MC_PUSH_MAX_M", "32")
)
if FUSED_K6_MC_PUSH_MAX_M not in (32, 64):
    raise ValueError("V4_FUSED_K6_MC_PUSH_MAX_M must be 32 or 64")
FUSED_K6_MC_PULL_AR = os.environ.get("V4_FUSED_K6_MC_PULL_AR", "0") == "1"
PIPELINED_W2_MC_PUSH_AR = (
    os.environ.get("V4_PIPELINED_W2_MC_PUSH_AR", "0") == "1"
)
W2_PROGRESS_MC_PUSH_AR = (
    os.environ.get("V4_W2_PROGRESS_MC_PUSH_AR", "0") == "1"
)
W2_PROGRESS_WORKERS = int(
    os.environ.get("V4_W2_PROGRESS_WORKERS", "8")
)
if W2_PROGRESS_WORKERS not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError("V4_W2_PROGRESS_WORKERS must be 1,2,4,8,16,32,64")
W2_PROGRESS_CHUNKS = int(os.environ.get("V4_W2_PROGRESS_CHUNKS", "4"))
if W2_PROGRESS_CHUNKS not in (2, 4, 8):
    raise ValueError("V4_W2_PROGRESS_CHUNKS must be 2,4,8")
W2_PROGRESS_INLINE_FINISH = (
    os.environ.get("V4_W2_PROGRESS_INLINE_FINISH", "0") == "1"
)
PIPELINE_CHUNKS = int(os.environ.get("V4_PIPELINE_CHUNKS", "4"))
PIPELINE_AR_BLOCKS = int(os.environ.get("V4_PIPELINE_AR_BLOCKS", "8"))
if PIPELINE_CHUNKS not in (2, 4, 8):
    raise ValueError("V4_PIPELINE_CHUNKS must be 2,4,8")
if PIPELINE_AR_BLOCKS not in (1, 2, 4, 8, 16, 32, 78):
    raise ValueError("V4_PIPELINE_AR_BLOCKS must be 1,2,4,8,16,32,78")
FUSED_RANK_ROUTE_MC_PULL_AR = (
    os.environ.get("V4_FUSED_RANK_ROUTE_MC_PULL_AR", "0") == "1"
)
RANK_ROUTE_PULL_BLOCKS = int(
    os.environ.get("V4_RANK_ROUTE_PULL_BLOCKS", "16")
)
if RANK_ROUTE_PULL_BLOCKS not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError("V4_RANK_ROUTE_PULL_BLOCKS must be 1,2,4,8,16,32,64")
FUSED_K6_NVLS_PULL_AR = (
    os.environ.get("V4_FUSED_K6_NVLS_PULL_AR", "0") == "1"
)
SINGLE_LAUNCH_TP4 = os.environ.get("V4_SINGLE_LAUNCH_TP4", "0") == "1"
SINGLE_LAUNCH_INTERLEAVED = (
    os.environ.get("V4_SINGLE_LAUNCH_INTERLEAVED", "1") == "1"
)
SINGLE_LAUNCH_CTAS_PER_SM = int(
    os.environ.get("V4_SINGLE_LAUNCH_CTAS_PER_SM", "5")
)
if SINGLE_LAUNCH_CTAS_PER_SM not in (1, 2, 3, 4, 5, 6, 7, 8):
    raise ValueError("V4_SINGLE_LAUNCH_CTAS_PER_SM must be in [1,8]")
K6_NVLS_PULL_BLOCKS = int(os.environ.get("V4_K6_NVLS_PULL_BLOCKS", "16"))
if K6_NVLS_PULL_BLOCKS not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError("V4_K6_NVLS_PULL_BLOCKS must be 1,2,4,8,16,32,64")
MC_PULL_BLOCKS = int(os.environ.get("V4_MC_PULL_BLOCKS", "0"))
MC_PULL_UNROLL = int(os.environ.get("V4_MC_PULL_UNROLL", "0"))
if MC_PULL_BLOCKS < 0:
    raise ValueError("V4_MC_PULL_BLOCKS must be nonnegative")
if MC_PULL_UNROLL not in (0, 2, 4, 8, 16):
    raise ValueError("V4_MC_PULL_UNROLL must be 0,2,4,8,16")
W2_GLOBAL_LUT = os.environ.get("V4_W2_GLOBAL_LUT", "0") == "1"
W2_S2R_PREFETCH = os.environ.get("V4_W2_S2R_PREFETCH", "1") == "1"
W13_S2R_PREFETCH = os.environ.get("V4_W13_S2R_PREFETCH", "1") == "1"
LEADER_MBAR_WAIT = os.environ.get("V4_LEADER_MBAR_WAIT", "1") == "1"
DIRECT_BARRIER_ADDR = os.environ.get("V4_DIRECT_BARRIER_ADDR", "0") == "1"
ROUTE_K_UNROLL2 = os.environ.get("V4_ROUTE_K_UNROLL2", "1") == "1"
ROUTE_K_UNROLL4 = os.environ.get("V4_ROUTE_K_UNROLL4", "1") == "1"
ROUTE_K_UNROLL8 = os.environ.get("V4_ROUTE_K_UNROLL8", "0") == "1"
ROUTE_K_UNROLL8_SPLIT2 = (
    os.environ.get("V4_ROUTE_K_UNROLL8_SPLIT2", "0") == "1"
)
W13_K_UNROLL8_SPLIT2 = (
    os.environ.get("V4_W13_K_UNROLL8_SPLIT2", "0") == "1"
)
W13_K_UNROLL16_SPLIT2 = (
    os.environ.get("V4_W13_K_UNROLL16_SPLIT2", "0") == "1"
)
W13_DISTRIBUTED_PREP = (
    os.environ.get("V4_W13_DISTRIBUTED_PREP", "1") == "1"
)
W13_DUAL_WG_SPLIT = os.environ.get("V4_W13_DUAL_WG_SPLIT", "0") == "1"
if W13_DUAL_WG_SPLIT and (
    WOUT != 128
    or not NORMALIZED_WEIGHT_SCALE
    or not BULK_WEIGHT_COPY
    or not INTERLEAVED_BULK_COPY
    or W13_PAIRED_WG
):
    raise ValueError(
        "V4_W13_DUAL_WG_SPLIT requires the selected WOUT128 normalized "
        "interleaved split-K W13 path"
    )
W2_DISTRIBUTED_PREP = (
    os.environ.get("V4_W2_DISTRIBUTED_PREP", "1") == "1"
)
W13_MERGED_WGMMA_GROUP = (
    os.environ.get("V4_W13_MERGED_WGMMA_GROUP", "0") == "1"
)
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
W13_LB10_MAX_SMEM = (
    os.environ.get("V4_W13_LB10_MAX_SMEM", "0") == "1"
)
W13_LAUNCH_BOUND_10 = W13_LB10_MAX_SMEM or (
    os.environ.get("V4_W13_LAUNCH_BOUND_10", "0") == "1"
)
W13_MAX_SMEM_CARVEOUT = W13_LB10_MAX_SMEM
if W13_LAUNCH_BOUND_10 and MIN_BLOCKS_PER_SM:
    raise ValueError(
        "V4_W13_LAUNCH_BOUND_10 and V4_MIN_BLOCKS_PER_SM are exclusive"
    )
if W13_LAUNCH_BOUND_10 and (W13_DUAL_WG_SPLIT or W13_PAIRED_WG):
    raise ValueError(
        "V4_W13_LAUNCH_BOUND_10 probes only the 128-thread split-K W13 path"
    )

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
#include <algorithm>
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
static constexpr bool kNormalizedSharedLut = K_NORMALIZED_SHARED_LUT;
static constexpr bool kActivationEvictLast = K_ACTIVATION_EVICT_LAST;
static constexpr bool kPredicatedPaddedActivation =
    K_PREDICATED_PADDED_ACTIVATION;
static constexpr bool kTiledWeightLayout = K_TILED_WEIGHT_LAYOUT;
static constexpr bool kBulkWeightCopy = K_BULK_WEIGHT_COPY;
static constexpr bool kTmaCtaScope = K_TMA_CTA_SCOPE;
static constexpr bool kWeightEvictFirst = K_WEIGHT_EVICT_FIRST;
static constexpr bool kWeightPolicyHoist = K_WEIGHT_POLICY_HOIST;
static constexpr bool kWeightPolicyConstant = K_WEIGHT_POLICY_CONSTANT;
static constexpr bool kW2NoWeightEvictFirst = K_W2_NO_WEIGHT_EVICT_FIRST;
static constexpr bool kInterleavedBulkCopy = K_INTERLEAVED_BULK_COPY;
static constexpr bool kCompactInterleavedScale =
    K_COMPACT_INTERLEAVED_SCALE;
static constexpr bool kMode2Braid = K_MODE2_BRAID;
static constexpr bool kW2GlobalLut = K_W2_GLOBAL_LUT;
static constexpr bool kW2S2RPrefetch = K_W2_S2R_PREFETCH;
static constexpr bool kW13S2RPrefetch = K_W13_S2R_PREFETCH;
static constexpr bool kLeaderMbarWait = K_LEADER_MBAR_WAIT;
static constexpr bool kDirectBarrierAddr = K_DIRECT_BARRIER_ADDR;
static constexpr bool kRouteKUnroll2 = K_ROUTE_K_UNROLL2;
static constexpr bool kRouteKUnroll4 = K_ROUTE_K_UNROLL4;
static constexpr bool kRouteKUnroll8 = K_ROUTE_K_UNROLL8;
static constexpr bool kRouteKUnroll8Split2 = K_ROUTE_K_UNROLL8_SPLIT2;
static constexpr bool kW13KUnroll8Split2 = K_W13_K_UNROLL8_SPLIT2;
static constexpr bool kW13KUnroll16Split2 = K_W13_K_UNROLL16_SPLIT2;
static constexpr bool kW13DistributedPrep = K_W13_DISTRIBUTED_PREP;
static constexpr bool kW13DualWgSplit = K_W13_DUAL_WG_SPLIT;
static constexpr bool kW2DistributedPrep = K_W2_DISTRIBUTED_PREP;
static constexpr bool kW13MergedWgmmaGroup = K_W13_MERGED_WGMMA_GROUP;
static constexpr bool kW13MaxSmemCarveout = K_W13_MAX_SMEM_CARVEOUT;
static constexpr int kTok = 8;
static constexpr int kTopK = 6;
static constexpr int kBlockK = 128;
static constexpr int kStages = K_WEIGHT_STAGES;
static_assert(kStages == 2 || kStages == 3 || kStages == 4);
static_assert(!kInterleavedBulkCopy
              || (kBulkWeightCopy && kTiledWeightLayout
                  && kStages == 2 && kScaleQuadReuse == 4
                  && kScaleBuffers == 2));
static_assert(!kCompactInterleavedScale || kInterleavedBulkCopy);
static constexpr float kRoutedScale = 1.5f;
static constexpr bool kW2RouteOutput = K_W2_ROUTE_OUTPUT;
static constexpr bool kW2SortedAct = K_W2_SORTED_ACT;
static constexpr bool kW2MblockScale = K_W2_MBLOCK_SCALE || kW2SortedAct;
static constexpr bool kW2CoalescedStore = K_W2_COALESCED_STORE;
static constexpr bool kW2FoldGlobalScale = K_W2_FOLD_GLOBAL_SCALE;
static constexpr bool kSingleLaunchInterleaved =
    K_SINGLE_LAUNCH_INTERLEAVED;

#if K_MIN_BLOCKS_PER_SM > 0
#define ROUTE_LAUNCH_BOUNDS(IS_W13, DUAL) \
    __launch_bounds__((DUAL) ? 256 : 128, K_MIN_BLOCKS_PER_SM)
#elif K_W13_LAUNCH_BOUND_10
// A min-block value of one lets ptxas inflate the non-W13 instantiations
// from their natural 55 registers to 62, reducing W2 occupancy.  Nine keeps
// the existing W2 occupancy ceiling while W13 alone is forced to ten CTAs.
#define ROUTE_LAUNCH_BOUNDS(IS_W13, DUAL) \
    __launch_bounds__((DUAL) ? 256 : 128, (IS_W13) ? 10 : 9)
#else
#define ROUTE_LAUNCH_BOUNDS(IS_W13, DUAL) \
    __launch_bounds__((DUAL) ? 256 : 128)
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

__device__ __forceinline__ void mbar_arrive(uint32_t address) {
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.shared.b64 %0,[%1];"
        : "=l"(state) : "r"(address) : "memory");
}

template <bool UseEvictFirst>
__device__ __forceinline__ void bulk_gmem_to_smem(
        uint32_t dst, const void* src, int bytes, uint32_t mbar,
        uint64_t cache_policy) {
    if constexpr (kTmaCtaScope) {
        if constexpr (UseEvictFirst) {
            if constexpr (kWeightPolicyHoist) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],%4;"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar),
                       "l"(cache_policy) : "memory");
            } else if constexpr (kWeightPolicyConstant) {
                // Read back from createpolicy.fractional.L2::evict_first on
                // the benchmark H20/sm90a toolchain.  Keep the value local to
                // the issue site so its lifetime matches the selected path.
                asm volatile(
                    "{.reg .b64 policy;"
                    " mov.b64 policy, 0x12f0000000000000;"
                    " cp.async.bulk.shared::cta.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],policy;}"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar)
                    : "memory");
            } else {
                asm volatile(
                    "{.reg .b64 policy;"
                    " createpolicy.fractional.L2::evict_first.b64 policy,1.0;"
                    " cp.async.bulk.shared::cta.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],policy;}"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar)
                    : "memory");
            }
        } else {
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::"
                "complete_tx::bytes [%0],[%1],%2,[%3];"
                :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
        }
    } else {
        if constexpr (UseEvictFirst) {
            if constexpr (kWeightPolicyHoist) {
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],%4;"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar),
                       "l"(cache_policy) : "memory");
            } else if constexpr (kWeightPolicyConstant) {
                asm volatile(
                    "{.reg .b64 policy;"
                    " mov.b64 policy, 0x12f0000000000000;"
                    " cp.async.bulk.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],policy;}"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar)
                    : "memory");
            } else {
                asm volatile(
                    "{.reg .b64 policy;"
                    " createpolicy.fractional.L2::evict_first.b64 policy,1.0;"
                    " cp.async.bulk.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes.L2::cache_hint "
                    "[%0],[%1],%2,[%3],policy;}"
                    :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar)
                    : "memory");
            }
        } else {
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::"
                "complete_tx::bytes [%0],[%1],%2,[%3];"
                :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
        }
    }
}

__device__ __forceinline__ uint2 load_reused_u64(
        const uint2* pointer, uint64_t cache_policy) {
    if constexpr (kActivationEvictLast) {
        uint2 value;
        asm volatile(
            "ld.global.L2::cache_hint.v2.u32 {%0,%1},[%2],%3;"
            : "=r"(value.x), "=r"(value.y)
            : "l"(pointer), "l"(cache_policy) : "memory");
        return value;
    }
    return __ldg(pointer);
}

__device__ __forceinline__ float load_reused_f32(
        const float* pointer, uint64_t cache_policy) {
    if constexpr (kActivationEvictLast) {
        float value;
        asm volatile(
            "ld.global.L2::cache_hint.f32 %0,[%1],%2;"
            : "=f"(value)
            : "l"(pointer), "l"(cache_policy) : "memory");
        return value;
    }
    return __ldg(pointer);
}

// Preserve the existing zero-fill semantics for padded route lanes without
// creating a divergent C++ load branch.  The pointer is always formed from a
// clamped row; the PTX predicate prevents any transaction for an invalid row.
__device__ __forceinline__ uint2 load_reused_u64_predicated(
        const uint2* pointer, int valid_row, uint64_t cache_policy) {
    uint2 value = make_uint2(0, 0);
    if constexpr (kActivationEvictLast) {
        asm volatile(
            "{.reg .pred p; setp.ge.s32 p,%3,0;"
            " @p ld.global.L2::cache_hint.v2.u32 {%0,%1},[%2],%4;}"
            : "+r"(value.x), "+r"(value.y)
            : "l"(pointer), "r"(valid_row), "l"(cache_policy) : "memory");
    } else {
        asm volatile(
            "{.reg .pred p; setp.ge.s32 p,%3,0;"
            " @p ld.global.v2.u32 {%0,%1},[%2];}"
            : "+r"(value.x), "+r"(value.y)
            : "l"(pointer), "r"(valid_row) : "memory");
    }
    return value;
}

__device__ __forceinline__ float load_reused_f32_predicated(
        const float* pointer, int valid_row, uint64_t cache_policy) {
    float value = 0.0f;
    if constexpr (kActivationEvictLast) {
        asm volatile(
            "{.reg .pred p; setp.ge.s32 p,%2,0;"
            " @p ld.global.L2::cache_hint.f32 %0,[%1],%3;}"
            : "+f"(value)
            : "l"(pointer), "r"(valid_row), "l"(cache_policy) : "memory");
    } else {
        asm volatile(
            "{.reg .pred p; setp.ge.s32 p,%2,0;"
            " @p ld.global.f32 %0,[%1];}"
            : "+f"(value)
            : "l"(pointer), "r"(valid_row) : "memory");
    }
    return value;
}

__device__ __forceinline__ void store_shared_f32_predicated(
        float* pointer, float value, int valid_slot) {
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(pointer));
    asm volatile(
        "{.reg .pred p; setp.ge.s32 p,%2,0; @p st.shared.f32 [%0],%1;}"
        :: "r"(address), "f"(value), "r"(valid_slot) : "memory");
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

__device__ __forceinline__ int32_t progress_load_acquire(
        const int32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.acquire.gpu.global.u32 %0,[%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return static_cast<int32_t>(value);
}

__device__ __forceinline__ void progress_store_release(
        int32_t* pointer, int32_t value) {
    asm volatile(
        "st.release.gpu.global.u32 [%0],%1;"
        : : "l"(pointer), "r"(static_cast<uint32_t>(value)) : "memory");
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

template <int K, int N, int SplitK, bool IsW13, int LaunchNTiles = 0,
          bool PublishW2Progress = false, bool DualWgW13 = false>
__device__ __forceinline__ void route_gemm_task(
        const CUtensorMap* tma_weight,
        const CUtensorMap* tma_weight_scale,
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
        int32_t* __restrict__ progress_state,
        int max_routes,
        int n_tile_begin,
        int linear_block_idx) {
    static_assert(K % kBlockK == 0);
    static_assert(N % kWout == 0);
    static_assert((K / kBlockK) % SplitK == 0);
    constexpr int kNumKTiles = K / kBlockK;
    constexpr int kKTilesPerSplit = kNumKTiles / SplitK;
    constexpr int kWeightStageBytes = kWout * (kBlockK / 2);
    // The legacy layout fetches four adjacent K128 scale quads per row to
    // satisfy a 16-byte tensor-map box.  The compact linear layout appends
    // exactly the four scale bytes consumed by each K128 weight tile.
    constexpr int kScaleRowBytes =
        kCompactInterleavedScale ? 4 : 16;
    constexpr int kScaleStageBytes = kWout * kScaleRowBytes;
    constexpr bool kUseTmaScale = K >= 512 || kCompactInterleavedScale;
    constexpr bool kInterleavedScale =
        kInterleavedBulkCopy && kUseTmaScale;
    constexpr int kCombinedStageBytes =
        kWeightStageBytes + kScaleStageBytes;
    constexpr int kWeightStageStride =
        kInterleavedScale ? kCombinedStageBytes : kWeightStageBytes;
    constexpr int kScaleStageStride =
        kInterleavedScale ? kCombinedStageBytes : kScaleStageBytes;
    constexpr int kEffectiveScaleBuffers =
        kUseTmaScale ? kScaleBuffers : kStages;
    constexpr int kNumNTiles = N / kWout;
    constexpr int kLaunchNTiles =
        LaunchNTiles == 0 ? kNumNTiles : LaunchNTiles;
    static_assert(kNumNTiles % kLaunchNTiles == 0);
    constexpr int kMathWGs = DualWgW13 ? 2 : 1;
    static_assert(kLaunchNTiles % kMathWGs == 0);
    static_assert(!DualWgW13 || (IsW13 && kWout == 128
                  && LaunchNTiles == 0 && kInterleavedScale),
                  "dual-WG split-K is specialized for selected W13");
    constexpr int kTaskNTiles = kLaunchNTiles / kMathWGs;
    constexpr bool kS2RPrefetch =
        IsW13 ? kW13S2RPrefetch : kW2S2RPrefetch;
    constexpr bool kMergedWgmmaGroup =
        IsW13 && kW13MergedWgmmaGroup;
    constexpr bool kDistributedPrep =
        IsW13 ? kW13DistributedPrep : kW2DistributedPrep;
    constexpr bool kUseWeightEvictFirst =
        kWeightEvictFirst && (IsW13 || !kW2NoWeightEvictFirst);
    constexpr int kTmaIssuerTid =
        kDistributedPrep ? 32 : 0;

    const int tid = threadIdx.x;
    const int math_wg = DualWgW13 ? tid >> 7 : 0;
    const int mtid = DualWgW13 ? tid & 127 : tid;
    const int split_idx = linear_block_idx % SplitK;
    const int task_idx = linear_block_idx / SplitK;
    const int m_block_idx = task_idx / kTaskNTiles;
    const int local_n_task_idx = task_idx % kTaskNTiles;
    const int local_n_block_idx = local_n_task_idx * kMathWGs + math_wg;
    const int n_block_idx = LaunchNTiles == 0
        ? local_n_block_idx
        : n_tile_begin + local_n_block_idx;
    if (m_block_idx * kTok >= __ldg(num_tokens_padded))
        return;

    const int expert_idx = __ldg(expert_ids + m_block_idx);
    if (expert_idx < 0)
        return;
    const int weight_row = expert_idx * N + n_block_idx * kWout;
    const int kt_begin = split_idx * kKTilesPerSplit;

    constexpr int kWeightWGBytes = kInterleavedScale
        ? kStages * kCombinedStageBytes
        : kStages * kWeightStageBytes
            + kEffectiveScaleBuffers * kScaleStageBytes;
    extern __shared__ __align__(1024) uint8_t dynamic_smem[];
    uint8_t* weight_smem = dynamic_smem + math_wg * kWeightWGBytes;
    uint8_t* weight_scale_smem =
        weight_smem + (kInterleavedScale
            ? kWeightStageBytes
            : kStages * kWeightStageBytes);
    uint8_t* activation_smem =
        kInterleavedScale
        ? dynamic_smem + kMathWGs * kWeightWGBytes
        : dynamic_smem + kMathWGs * kWeightWGBytes;
    __nv_bfloat16* w2_output_smem = reinterpret_cast<__nv_bfloat16*>(
        activation_smem + kTok * kBlockK);
    const uint32_t weight_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(weight_smem));
    const uint32_t weight_scale_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(weight_scale_smem));
    const uint32_t activation_smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(activation_smem));
    const int weight_swizzle_row_offset =
        kWeightSwizzle == 64 ? ((weight_smem_addr >> 7) & 3) : 0;

    __shared__ __align__(8) uint64_t full_barriers[kMathWGs][kStages];
    __shared__ __align__(8) uint64_t scale_barriers[kMathWGs];
    __shared__ __align__(8) uint64_t activation_empty_barrier;
    __shared__ uint2 lut_smem[
        (kNormalizedWeightScale && kNormalizedSharedLut) ? 13 :
        (kNormalizedWeightScale || kDequantSynthLut
         || (!IsW13 && kW2GlobalLut)) ? 1 : kLutRows];
    __shared__ float activation_scale_smem[kTok];
    __shared__ float expert_weight_scale;
    __shared__ int32_t route_ids[kTok];
    __shared__ int32_t activation_rows[kTok];

    uint64_t weight_cache_policy = 0;
    if constexpr (kUseWeightEvictFirst && kWeightPolicyHoist) {
        asm volatile(
            "createpolicy.fractional.L2::evict_first.b64 %0,1.0;"
            : "=l"(weight_cache_policy));
    }
    uint64_t reused_cache_policy = 0;
    if constexpr (kActivationEvictLast) {
        asm volatile(
            "createpolicy.fractional.L2::evict_last.b64 %0,1.0;"
            : "=l"(reused_cache_policy));
    }
    if (tid < kTok) {
        const int position = m_block_idx * kTok + tid;
        const int route = __ldg(sorted_ids + position);
        route_ids[tid] = route;
        activation_rows[tid] = route < max_routes
            ? (IsW13 ? route / kTopK
                     : (kW2SortedAct ? position : route))
            : -1;
    }
    if (tid == 0) {
        if constexpr (kNormalizedWeightScale
                      && (IsW13 || !kW2FoldGlobalScale)) {
            expert_weight_scale = load_reused_f32(
                weight_global_scale + expert_idx, reused_cache_policy);
        } else {
            expert_weight_scale = 1.0f;
        }
    }
    if constexpr (kNormalizedWeightScale && kNormalizedSharedLut) {
        for (int i = tid; i < 13; i += blockDim.x)
            lut_smem[i] = synth_normalized_e2m1_lut(i);
    } else if constexpr (!kNormalizedWeightScale && !kDequantSynthLut
                         && (IsW13 || !kW2GlobalLut)) {
        for (int i = tid; i < kLutRows; i += blockDim.x) {
            constexpr int kGlobalLutOffset =
                kLutRows == 128 ? mxfp4::kE8M0LutBase : 0;
            lut_smem[i] = global_lut[kGlobalLutOffset + i];
        }
    }

    const uint32_t barrier_base_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&full_barriers[math_wg][0]));
    uint32_t barrier_addr[kDirectBarrierAddr ? 1 : kStages];
    if constexpr (!kDirectBarrierAddr) {
        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage)
            barrier_addr[stage] = static_cast<uint32_t>(
                __cvta_generic_to_shared(&full_barriers[math_wg][stage]));
    }
    const auto weight_barrier_addr = [&](int stage) {
        if constexpr (kDirectBarrierAddr)
            return barrier_base_addr + static_cast<uint32_t>(stage) * 8u;
        else
            return barrier_addr[stage];
    };
    const uint32_t scale_barrier_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&scale_barriers[math_wg]));
    const uint32_t activation_empty_barrier_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&activation_empty_barrier));
    if constexpr (DualWgW13) {
        if (tid == 0)
            mbar_init(activation_empty_barrier_addr);
    }
    if (mtid == 0) {
        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage)
            mbar_init(weight_barrier_addr(stage));
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1)
            mbar_init(scale_barrier_addr);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const auto load_weight_stage = [&](int local_kt, int stage) {
        if (mtid == kTmaIssuerTid) {
            const uint32_t stage_barrier_addr = weight_barrier_addr(stage);
            const int global_kt = kt_begin + local_kt;
            const uint32_t weight_dst =
                weight_smem_addr + stage * kWeightStageStride;
            bool load_scale = kUseTmaScale && kScaleBuffers == 2;
            int scale_kt = global_kt;
            int scale_stage = stage;
            if constexpr (kInterleavedScale) {
                if constexpr (kCompactInterleavedScale) {
                    load_scale = true;
                    scale_stage = stage;
                } else {
                    const int record = global_kt & 7;
                    const bool even_quartet = record == 0;
                    const bool odd_quartet =
                        record == 3 && local_kt + 1 < kKTilesPerSplit;
                    load_scale = even_quartet || odd_quartet;
                    scale_kt = even_quartet ? global_kt : global_kt + 1;
                    scale_stage = (scale_kt >> 2) & 1;
                }
            } else if constexpr (kUseTmaScale && kScaleQuadReuse == 4
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
                weight_scale_smem_addr + scale_stage * kScaleStageStride;
            if (load_scale) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(stage_barrier_addr),
                       "n"(kWeightStageBytes + kScaleStageBytes));
            } else {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(stage_barrier_addr), "n"(kWeightStageBytes));
            }
            if constexpr (kInterleavedScale) {
                constexpr int kBytesPerNTile = kCompactInterleavedScale
                    ? kNumKTiles * kCombinedStageBytes
                    : kNumKTiles * kWeightStageBytes
                        + (kNumKTiles / 4) * kScaleStageBytes;
                int offset;
                if constexpr (kCompactInterleavedScale) {
                    offset = global_kt * kCombinedStageBytes;
                } else {
                    const int record = global_kt & 7;
                    const int scales_before =
                        (global_kt >> 3) * 2
                        + (record >= 1 ? 1 : 0)
                        + (record >= 4 ? 1 : 0);
                    offset = global_kt * kWeightStageBytes
                        + scales_before * kScaleStageBytes;
                }
                const int64_t ntile =
                    static_cast<int64_t>(expert_idx) * kNumNTiles
                    + n_block_idx;
                const uint8_t* weight_src =
                    weight + ntile * kBytesPerNTile + offset;
                const int copy_bytes = load_scale
                    ? kCombinedStageBytes
                    : kWeightStageBytes;
                bulk_gmem_to_smem<kUseWeightEvictFirst>(
                    weight_dst, weight_src, copy_bytes,
                    stage_barrier_addr, weight_cache_policy);
            } else if constexpr (kBulkWeightCopy) {
                const int64_t tile =
                    (static_cast<int64_t>(expert_idx) * kNumNTiles
                     + n_block_idx) * kNumKTiles + global_kt;
                const uint8_t* weight_src =
                    weight + tile * kWeightStageBytes;
                bulk_gmem_to_smem<kUseWeightEvictFirst>(
                    weight_dst, weight_src, kWeightStageBytes,
                    stage_barrier_addr, weight_cache_policy);
            } else if constexpr (kTiledWeightLayout) {
                const int tiled_row =
                    ((expert_idx * kNumNTiles + n_block_idx) * kNumKTiles
                     + global_kt) * kWout;
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                    :: "r"(weight_dst), "l"(tma_weight),
                       "r"(0), "r"(tiled_row),
                       "r"(stage_barrier_addr) : "memory");
            } else {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                    :: "r"(weight_dst), "l"(tma_weight),
                       "r"(global_kt * (kBlockK / 2)), "r"(weight_row),
                       "r"(stage_barrier_addr) : "memory");
            }
            if (load_scale) {
                if constexpr (kInterleavedScale) {
                    // The combined transaction above lands the scale bytes
                    // immediately after the selected weight stage.
                } else if constexpr (kBulkWeightCopy) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int64_t scale_tile =
                        (static_cast<int64_t>(expert_idx) * kNumNTiles
                         + n_block_idx) * kScaleTiles + (scale_kt >> 2);
                    const uint8_t* scale_src =
                        weight_scale + scale_tile * kScaleStageBytes;
                    bulk_gmem_to_smem<kUseWeightEvictFirst>(
                        scale_dst, scale_src, kScaleStageBytes,
                        stage_barrier_addr, weight_cache_policy);
                } else if constexpr (kTiledWeightLayout) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int tiled_scale_row =
                        ((expert_idx * kNumNTiles + n_block_idx) * kScaleTiles
                         + (scale_kt >> 2)) * kWout;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(scale_dst), "l"(tma_weight_scale),
                           "r"(0), "r"(tiled_scale_row),
                           "r"(stage_barrier_addr) : "memory");
                } else {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(scale_dst), "l"(tma_weight_scale),
                           "r"((scale_kt & ~3) * (kBlockK / 32)),
                           "r"(weight_row),
                           "r"(stage_barrier_addr) : "memory");
                }
            }
        }
    };

    const auto load_single_scale = [&](int global_kt) {
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1) {
            if (mtid == kTmaIssuerTid) {
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
                    bulk_gmem_to_smem<kUseWeightEvictFirst>(
                        weight_scale_smem_addr, scale_src,
                        kScaleStageBytes, scale_barrier_addr,
                        weight_cache_policy);
                } else if constexpr (kTiledWeightLayout) {
                    constexpr int kScaleTiles = kNumKTiles / 4;
                    const int tiled_scale_row =
                        ((expert_idx * kNumNTiles + n_block_idx) * kScaleTiles
                         + (global_kt >> 2)) * kWout;
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(weight_scale_smem_addr), "l"(tma_weight_scale),
                           "r"(0), "r"(tiled_scale_row),
                           "r"(scale_barrier_addr) : "memory");
                } else {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
                        :: "r"(weight_scale_smem_addr), "l"(tma_weight_scale),
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

    const int warp = mtid / 32;
    const int lane = mtid % 32;
    const int row0 = warp * 16 + lane / 4;
    const int row1 = row0 + 8;
    const int packed_k_offset = (lane % 4) * 4;
    const int column_base = (lane % 4) * 2;
    float accum[kWgmmaGroups][4] = {};

    #if K_W13_K_UNROLL16_SPLIT2
    #pragma unroll (K == 4096 && SplitK <= 2 ? 16 : 4)
    #elif K_W13_K_UNROLL8_SPLIT2
    #pragma unroll (K == 4096 && SplitK <= 2 ? 8 : 4)
    #elif K_ROUTE_K_UNROLL8_SPLIT2
    #pragma unroll (SplitK <= 2 ? 8 : 4)
    #elif K_ROUTE_K_UNROLL8
    #pragma unroll 8
    #elif K_ROUTE_K_UNROLL4
    #pragma unroll 4
    #elif K_ROUTE_K_UNROLL2
    #pragma unroll 2
    #else
    #pragma unroll 1
    #endif
    for (int local_kt = 0; local_kt < kKTilesPerSplit; ++local_kt) {
        const int stage = local_kt % kStages;
        const int global_kt = kt_begin + local_kt;
        const int scale_stage =
            kCompactInterleavedScale
            ? stage
            : !kUseTmaScale
            ? stage
            : (kScaleBuffers == 1
            ? 0
            : (kScaleQuadReuse == 4
               ? ((global_kt >> 2) & 1)
               : stage));
        const int scale_k_base =
            kCompactInterleavedScale ? 0 : (global_kt & 3) * 4;

        if constexpr (DualWgW13) {
            if (local_kt > 0 && math_wg == 0) {
                if (mtid == 0) {
                    mbar_wait(
                        activation_empty_barrier_addr,
                        (local_kt - 1) & 1u);
                }
                // Only WG0 participates.  Its leader observes WG1's
                // WGMMA-complete arrival before any lane overwrites the
                // shared activation tile for the next K128 iteration.
                asm volatile("bar.sync 1,128;" ::: "memory");
            }
        }

        // One uint2 per thread covers the complete 8x128 activation tile.
        if (!DualWgW13 || math_wg == 0) {
            const int token_slot = mtid / 16;
            const int k8 = (mtid % 16) * 8;
            uint2 value = make_uint2(0, 0);
            const int activation_row = activation_rows[token_slot];
            if constexpr (kPredicatedPaddedActivation && IsW13) {
                const int safe_row = activation_row < 0 ? 0 : activation_row;
                value = load_reused_u64_predicated(
                    reinterpret_cast<const uint2*>(
                        activation + static_cast<int64_t>(safe_row) * K
                        + global_kt * kBlockK + k8),
                    activation_row, reused_cache_policy);
            } else if (activation_row >= 0) {
                value = load_reused_u64(reinterpret_cast<const uint2*>(
                    activation + static_cast<int64_t>(activation_row) * K
                    + global_kt * kBlockK + k8), reused_cache_policy);
            }
            *reinterpret_cast<uint2*>(
                activation_smem + token_slot * kBlockK
                + (k8 ^ ((token_slot & 7) << 4))) = value;
        }

        int scale_slot = -1;
        if (!DualWgW13 || math_wg == 0) {
            if constexpr (kDistributedPrep) {
                const int warp_lane = mtid & 31;
                if (warp_lane < 2)
                    scale_slot = (mtid >> 5) * 2 + warp_lane;
            } else if (mtid < kTok) {
                scale_slot = mtid;
            }
        }
        if constexpr (kPredicatedPaddedActivation && IsW13) {
            const int safe_slot = scale_slot < 0 ? 0 : scale_slot;
            const int row = activation_rows[safe_slot];
            const int safe_row = row < 0 ? 0 : row;
            int64_t scale_index;
            if constexpr (!IsW13 && kW2MblockScale) {
                scale_index =
                    (static_cast<int64_t>(safe_row >> 3) * kNumKTiles
                     + global_kt) * kTok + (safe_row & 7);
            } else {
                scale_index =
                    static_cast<int64_t>(safe_row) * kNumKTiles + global_kt;
            }
            const int valid_load = scale_slot < row ? scale_slot : row;
            const float loaded_scale = load_reused_f32_predicated(
                activation_scale + scale_index,
                valid_load, reused_cache_policy);
            store_shared_f32_predicated(
                activation_scale_smem + safe_slot,
                loaded_scale * expert_weight_scale, scale_slot);
        } else if (scale_slot >= 0) {
            const int row = activation_rows[scale_slot];
            if (row >= 0) {
                int64_t scale_index;
                if constexpr (!IsW13 && kW2MblockScale) {
                    scale_index =
                        (static_cast<int64_t>(m_block_idx) * kNumKTiles
                         + global_kt) * kTok + scale_slot;
                } else {
                    scale_index =
                        static_cast<int64_t>(row) * kNumKTiles + global_kt;
                }
                activation_scale_smem[scale_slot] =
                    load_reused_f32(
                        activation_scale + scale_index,
                        reused_cache_policy)
                    * expert_weight_scale;
            } else {
                activation_scale_smem[scale_slot] = 0.0f;
            }
        }
        if constexpr (!kUseTmaScale) {
            for (int i = tid; i < kWout * 4; i += blockDim.x) {
                const int local_n = i >> 2;
                const int k_group = i & 3;
                weight_scale_smem[stage * kScaleStageBytes
                                  + local_n * kScaleRowBytes
                                  + scale_k_base + k_group] = __ldg(
                    weight_scale
                    + static_cast<int64_t>(weight_row + local_n) * (K / 32)
                    + global_kt * 4 + k_group);
            }
        }
        // The long-K W13 path benefits from one polling lane.  Short-K W2
        // retains all-warp waits, which wake the consumer faster in practice.
        if constexpr (kLeaderMbarWait && IsW13) {
            if (mtid == 0)
                mbar_wait(weight_barrier_addr(stage),
                          (local_kt / kStages) & 1u);
        } else {
            mbar_wait(weight_barrier_addr(stage),
                      (local_kt / kStages) & 1u);
        }
        if constexpr (kUseTmaScale && kScaleQuadReuse == 4
                      && kScaleBuffers == 1) {
            if ((local_kt & 3) == 0) {
                if constexpr (kLeaderMbarWait && IsW13) {
                    if (mtid == 0)
                        mbar_wait(scale_barrier_addr, (local_kt >> 2) & 1u);
                } else {
                    mbar_wait(scale_barrier_addr, (local_kt >> 2) & 1u);
                }
            }
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
                weight_smem_addr + stage * kWeightStageStride;
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
                            weight_scale_smem[scale_stage * kScaleStageStride
                                              + group_row0 * kScaleRowBytes
                                              + scale_k_base];
                        const uint32_t exponent1 =
                            weight_scale_smem[scale_stage * kScaleStageStride
                                              + group_row1 * kScaleRowBytes
                                              + scale_k_base];
                        if constexpr (kNormalizedWeightScale) {
                            if constexpr (kNormalizedSharedLut) {
                                weight_lut0 = lut_smem[exponent0];
                                weight_lut1 = lut_smem[exponent1];
                            } else {
                                weight_lut0 =
                                    synth_normalized_e2m1_lut(exponent0);
                                weight_lut1 =
                                    synth_normalized_e2m1_lut(exponent1);
                            }
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
                        weight_scale_smem[scale_stage * kScaleStageStride
                                          + group_row0 * kScaleRowBytes
                                          + scale_k_base + k_step];
                    const uint32_t exponent1 =
                        weight_scale_smem[scale_stage * kScaleStageStride
                                          + group_row1 * kScaleRowBytes
                                          + scale_k_base + k_step];
                    if constexpr (kNormalizedWeightScale) {
                        if constexpr (kNormalizedSharedLut) {
                            weight_lut0 = lut_smem[exponent0];
                            weight_lut1 = lut_smem[exponent1];
                        } else {
                            weight_lut0 =
                                synth_normalized_e2m1_lut(exponent0);
                            weight_lut1 =
                                synth_normalized_e2m1_lut(exponent1);
                        }
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
                            weight_scale_smem[scale_stage * kScaleStageStride
                                              + group_row0 * kScaleRowBytes
                                              + scale_k_base
                                              + next_k_step];
                        const uint32_t next_exponent1 =
                            weight_scale_smem[scale_stage * kScaleStageStride
                                              + group_row1 * kScaleRowBytes
                                              + scale_k_base
                                              + next_k_step];
                        if constexpr (kNormalizedWeightScale) {
                            if constexpr (kNormalizedSharedLut) {
                                next_weight_lut0[group] =
                                    lut_smem[next_exponent0];
                                next_weight_lut1[group] =
                                    lut_smem[next_exponent1];
                            } else {
                                next_weight_lut0[group] =
                                    synth_normalized_e2m1_lut(next_exponent0);
                                next_weight_lut1[group] =
                                    synth_normalized_e2m1_lut(next_exponent1);
                            }
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
            if constexpr (!kMergedWgmmaGroup) {
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (int group = 0; group < kWgmmaGroups; ++group) {
                    #pragma unroll
                    for (int value = 0; value < 4; ++value)
                        ptx::warpgroup_fence_operand(tile[group][value]);
                }
                ptx::warpgroup_wait<0>();
            }
        }
        if constexpr (kMergedWgmmaGroup) {
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (int group = 0; group < kWgmmaGroups; ++group) {
                #pragma unroll
                for (int value = 0; value < 4; ++value)
                    ptx::warpgroup_fence_operand(tile[group][value]);
            }
            ptx::warpgroup_wait<0>();
        }
        if constexpr (DualWgW13) {
            if (math_wg == 1 && mtid == 0)
                mbar_arrive(activation_empty_barrier_addr);
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
                if constexpr (kW2CoalescedStore) {
                    // Each warp owns 16 columns in each N64 accumulator
                    // group.  Keep its eight-route, 32-column slice private
                    // so the later exchange needs only a warp barrier.
                    const int warp_buffer = warp * kTok * 32;
                    const int warp_row = row0 - warp * 16;
                    const int local_n0 = group * 16 + warp_row;
                    const int local_n1 = local_n0 + 8;
                    w2_output_smem[
                        warp_buffer + column_base * 32 + local_n0] =
                        __float2bfloat16(accum[group][0]);
                    w2_output_smem[
                        warp_buffer + column_base * 32 + local_n1] =
                        __float2bfloat16(accum[group][2]);
                    w2_output_smem[
                        warp_buffer + (column_base + 1) * 32 + local_n0] =
                        __float2bfloat16(accum[group][1]);
                    w2_output_smem[
                        warp_buffer + (column_base + 1) * 32 + local_n1] =
                        __float2bfloat16(accum[group][3]);
                } else {
                    auto* route_output =
                        reinterpret_cast<__nv_bfloat16*>(output);
                    if (route0 < max_routes) {
                        route_output[static_cast<int64_t>(route0) * N
                                     + output_n0] =
                            __float2bfloat16(accum[group][0]);
                        route_output[static_cast<int64_t>(route0) * N
                                     + output_n1] =
                            __float2bfloat16(accum[group][2]);
                    }
                    if (route1 < max_routes) {
                        route_output[static_cast<int64_t>(route1) * N
                                     + output_n0] =
                            __float2bfloat16(accum[group][1]);
                        route_output[static_cast<int64_t>(route1) * N
                                     + output_n1] =
                            __float2bfloat16(accum[group][3]);
                    }
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
    if constexpr (!IsW13 && kW2RouteOutput && kW2CoalescedStore) {
        __syncwarp();
        // Each group of four lanes emits one route's 32 columns owned by
        // this warp.  The four 16-byte vectors cover two contiguous N16
        // spans without a CTA-wide synchronization point.
        const int output_slot = lane >> 2;
        const int vector_index = lane & 3;
        const int output_route = route_ids[output_slot];
        if (output_route < max_routes) {
            auto* route_output = reinterpret_cast<__nv_bfloat16*>(output);
            const int local_column = vector_index * 8;
            const int warp_buffer = warp * kTok * 32;
            const uint4 packed = *reinterpret_cast<const uint4*>(
                w2_output_smem + warp_buffer
                + output_slot * 32 + local_column);
            const int output_group = vector_index >> 1;
            const int output_half = vector_index & 1;
            const int output_column =
                n_block_idx * kWout + output_group * 64
                + warp * 16 + output_half * 8;
            *reinterpret_cast<uint4*>(
                route_output + static_cast<int64_t>(output_route) * N
                + output_column) = packed;
        }
    }

    if constexpr (PublishW2Progress) {
        static_assert(!IsW13 && SplitK == 1 && LaunchNTiles == 0
                      && N == 4096 && kWout == 128,
                      "W2 progress publication requires full N4096/WOUT128");
        constexpr int kNumW2Tiles = N / kWout;

        // __syncthreads strongly-happens-before every participating thread
        // resumes.  Each route lane's following device-scope release store
        // therefore publishes every CTA lane's ordinary global output stores
        // to the worker's matching device-scope acquire load.
        __syncthreads();
        if (tid < kTok) {
            const int route = route_ids[tid];
            if (route < max_routes)
                progress_store_release(
                    progress_state + route * kNumW2Tiles + n_block_idx, 1);
        }
    }
}

// Keep the selected standalone launch as a thin wrapper around the task body.
// The same task body is also the compute building block for the single-launch
// TP MegaMoE path, avoiding a second independently maintained GEMM core.
template <int K, int N, int SplitK, bool IsW13, int LaunchNTiles = 0,
          bool PublishW2Progress = false, bool DualWgW13 = false>
__global__ ROUTE_LAUNCH_BOUNDS(IsW13, DualWgW13) void route_gemm(
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
        int32_t* __restrict__ progress_state,
        int max_routes,
        int n_tile_begin) {
    route_gemm_task<K, N, SplitK, IsW13, LaunchNTiles,
                    PublishW2Progress, DualWgW13>(
        &tma_weight, &tma_weight_scale,
        weight, weight_scale, weight_global_scale,
        activation, activation_scale,
        sorted_ids, expert_ids, num_tokens_padded, topk_weights,
        output, global_lut, progress_state, max_routes, n_tile_begin,
        static_cast<int>(blockIdx.x));
}

// Native MegaMoE-style TP-local W13 task.  Two aligned math warpgroups consume
// one N256 block whose physical rows alternate gate/up in eight-row chunks.
// Each warpgroup owns its two weight/activation stages and issues its own bulk
// copies, avoiding the four dedicated producer warps of the first prototype.
// Full K stays inside the CTA, so its epilogue can preserve the public BF16 ->
// SwiGLU -> BF16 -> group-128 FP8 boundary without a split-K workspace or
// completion atomics.
template <int Intermediate>
__global__ __launch_bounds__(256, 1) void paired_w13_fused_kernel(
        const uint8_t* __restrict__ weight,
        const float* __restrict__ weight_global_scale,
        const uint8_t* __restrict__ activation,
        const float* __restrict__ activation_scale,
        const int32_t* __restrict__ sorted_ids,
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ num_tokens_padded,
        float* __restrict__ raw_output,
        __nv_bfloat16* __restrict__ bf16_output,
        uint8_t* __restrict__ quantized_output,
        float* __restrict__ output_scale,
        int max_routes) {
    constexpr int K = 4096;
    constexpr int kNumKTiles = K / kBlockK;
    constexpr int kPairStages = 2;
    constexpr int kMathWGs = 2;
    constexpr int kWeightStageBytes = kWout * (kBlockK / 2);
    constexpr int kScaleStageBytes = kWout * 16;
    constexpr int kCombinedStageBytes =
        kWeightStageBytes + kScaleStageBytes;
    constexpr int kWeightWGBytes = kPairStages * kCombinedStageBytes;
    constexpr int kActivationStageBytes = kTok * kBlockK;
    constexpr int kActivationWGBytes =
        kPairStages * kActivationStageBytes;
    constexpr int kOutputGroups = Intermediate / 128;
    constexpr int kRawN = 2 * Intermediate;
    constexpr int kRawNTiles = kRawN / kWout;
    constexpr int kScaleTiles = kNumKTiles / 4;
    constexpr int kBytesPerNTile =
        kNumKTiles * kWeightStageBytes
        + kScaleTiles * kScaleStageBytes;
    static_assert(kWout == 128,
                  "paired W13 requires two N128 math warpgroups");
    static_assert(Intermediate == 512 || Intermediate == 256);
    static_assert(kRawNTiles == 2 * kOutputGroups);

    const int task_idx = blockIdx.x;
    const int m_block_idx = task_idx / kOutputGroups;
    const int output_group = task_idx - m_block_idx * kOutputGroups;
    if (m_block_idx * kTok >= __ldg(num_tokens_padded))
        return;
    const int expert_idx = __ldg(expert_ids + m_block_idx);
    if (expert_idx < 0)
        return;

    extern __shared__ __align__(1024) uint8_t dynamic_smem[];
    uint8_t* weight_smem = dynamic_smem;
    uint8_t* activation_smem =
        weight_smem + kMathWGs * kWeightWGBytes;
    const uint32_t weight_smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(weight_smem));
    const uint32_t activation_smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(activation_smem));
    const int weight_swizzle_row_offset =
        (weight_smem_addr >> 7) & 3;

    __shared__ __align__(8)
        uint64_t full_barriers[kMathWGs][kPairStages];
    __shared__ int32_t route_ids[kTok];
    __shared__ int32_t activation_rows[kTok];
    __shared__ float
        activation_scale_smem[kMathWGs][kPairStages][kTok];
    __shared__ float expert_scale;
    __shared__ float epilogue_amax[kMathWGs][kTok][32];
    __shared__ float epilogue_scale_inv[kTok];

    const int tid = threadIdx.x;
    const int math_wg = tid >> 7;
    const int mtid = tid & 127;
    if (tid < kTok) {
        const int route = __ldg(sorted_ids + m_block_idx * kTok + tid);
        route_ids[tid] = route;
        activation_rows[tid] = route < max_routes ? route / kTopK : -1;
    }
    if (tid == 0)
        expert_scale = __ldg(weight_global_scale + expert_idx);

    uint32_t full_barrier_addr[kPairStages];
    #pragma unroll
    for (int stage = 0; stage < kPairStages; ++stage) {
        full_barrier_addr[stage] = static_cast<uint32_t>(
            __cvta_generic_to_shared(&full_barriers[math_wg][stage]));
    }
    if (mtid == 0) {
        #pragma unroll
        for (int stage = 0; stage < kPairStages; ++stage)
            mbar_init(full_barrier_addr[stage]);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    float epilogue_value0[kWgmmaGroups] = {};
    float epilogue_value1[kWgmmaGroups] = {};

#if 0
    if (tid < 128) {
        // Producer group.  Named barrier 1 contains exactly these four warps.
        for (int kt = 0; kt < kNumKTiles; ++kt) {
            const int stage = kt & 1;
            if (kt >= kPairStages && tid == 0)
                mbar_wait(empty_barrier_addr[stage],
                          ((kt / kPairStages) - 1) & 1u);
            asm volatile("bar.sync 1,128;" ::: "memory");

            const int token_slot = tid >> 4;
            const int k8 = (tid & 15) * 8;
            uint2 value = make_uint2(0u, 0u);
            const int activation_row = activation_rows[token_slot];
            if (activation_row >= 0) {
                value = *reinterpret_cast<const uint2*>(
                    activation + static_cast<int64_t>(activation_row) * K
                    + kt * kBlockK + k8);
            }
            *reinterpret_cast<uint2*>(
                activation_smem + stage * kActivationStageBytes
                + token_slot * kBlockK
                + (k8 ^ ((token_slot & 7) << 4))) = value;
            if (tid < kTok) {
                const int row = activation_rows[tid];
                activation_scale_smem[stage][tid] = row >= 0
                    ? __ldg(activation_scale
                            + static_cast<int64_t>(row) * kNumKTiles + kt)
                        * expert_scale
                    : 0.0f;
            }
            asm volatile("bar.sync 1,128;" ::: "memory");

            if (tid == 0) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                const int record = kt & 7;
                const bool load_scale =
                    record == 0 || (record == 3 && kt + 1 < kNumKTiles);
                if (load_scale) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                        :: "r"(full_barrier_addr[stage]),
                           "n"(kMathWGs * kCombinedStageBytes));
                } else {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                        :: "r"(full_barrier_addr[stage]),
                           "n"(kMathWGs * kWeightStageBytes));
                }
                const int scales_before =
                    (kt >> 3) * 2
                    + (record >= 1 ? 1 : 0)
                    + (record >= 4 ? 1 : 0);
                const int record_offset =
                    kt * kWeightStageBytes
                    + scales_before * kScaleStageBytes;
                const int copy_bytes = load_scale
                    ? kCombinedStageBytes : kWeightStageBytes;
                #pragma unroll
                for (int math_wg = 0; math_wg < kMathWGs; ++math_wg) {
                    const int raw_n_tile = output_group * 2 + math_wg;
                    const int64_t ntile =
                        static_cast<int64_t>(expert_idx) * kRawNTiles
                        + raw_n_tile;
                    const uint8_t* src =
                        weight + ntile * kBytesPerNTile + record_offset;
                    const uint32_t dst = weight_smem_addr
                        + math_wg * kWeightWGBytes
                        + stage * kCombinedStageBytes;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::"
                        "complete_tx::bytes [%0],[%1],%2,[%3];"
                        :: "r"(dst), "l"(src), "r"(copy_bytes),
                           "r"(full_barrier_addr[stage]) : "memory");
                }
            }
        }
    }
#endif
    {
        const int warp = mtid >> 5;
        const int lane = mtid & 31;
        const int row0 = warp * 16 + lane / 4;
        const int row1 = row0 + 8;
        const int packed_k_offset = (lane & 3) * 4;
        const int route_slot0 = (lane & 3) * 2;
        const int route_slot1 = route_slot0 + 1;
        float accum[kWgmmaGroups][4] = {};
        uint32_t next_packed0[kWgmmaGroups];
        uint32_t next_packed1[kWgmmaGroups];
        uint2 next_lut0[kWgmmaGroups];
        uint2 next_lut1[kWgmmaGroups];

        const auto load_weight_stage = [&](int kt, int stage) {
            if (mtid == 0) {
                const int record = kt & 7;
                const bool load_scale =
                    record == 0 || (record == 3 && kt + 1 < kNumKTiles);
                const int copy_bytes = load_scale
                    ? kCombinedStageBytes : kWeightStageBytes;
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"
                    :: "r"(full_barrier_addr[stage]), "r"(copy_bytes));
                const int scales_before =
                    (kt >> 3) * 2
                    + (record >= 1 ? 1 : 0)
                    + (record >= 4 ? 1 : 0);
                const int record_offset =
                    kt * kWeightStageBytes
                    + scales_before * kScaleStageBytes;
                const int raw_n_tile = output_group * 2 + math_wg;
                const int64_t ntile =
                    static_cast<int64_t>(expert_idx) * kRawNTiles
                    + raw_n_tile;
                const uint8_t* src =
                    weight + ntile * kBytesPerNTile + record_offset;
                const uint32_t dst = weight_smem_addr
                    + math_wg * kWeightWGBytes
                    + stage * kCombinedStageBytes;
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::"
                    "complete_tx::bytes [%0],[%1],%2,[%3];"
                    :: "r"(dst), "l"(src), "r"(copy_bytes),
                       "r"(full_barrier_addr[stage]) : "memory");
            }
        };

        #pragma unroll
        for (int stage = 0; stage < kPairStages; ++stage)
            load_weight_stage(stage, stage);

        for (int kt = 0; kt < kNumKTiles; ++kt) {
            const int stage = kt & 1;
            const int token_slot = mtid >> 4;
            const int k8 = (mtid & 15) * 8;
            uint2 value = make_uint2(0u, 0u);
            const int activation_row = activation_rows[token_slot];
            if (activation_row >= 0) {
                value = *reinterpret_cast<const uint2*>(
                    activation + static_cast<int64_t>(activation_row) * K
                    + kt * kBlockK + k8);
            }
            *reinterpret_cast<uint2*>(
                activation_smem + math_wg * kActivationWGBytes
                + stage * kActivationStageBytes
                + token_slot * kBlockK
                + (k8 ^ ((token_slot & 7) << 4))) = value;
            if (mtid < kTok) {
                const int row = activation_rows[mtid];
                activation_scale_smem[math_wg][stage][mtid] = row >= 0
                    ? __ldg(activation_scale
                            + static_cast<int64_t>(row) * kNumKTiles + kt)
                        * expert_scale
                    : 0.0f;
            }
            if (mtid == 0)
                mbar_wait(full_barrier_addr[stage], (kt >> 1) & 1u);
            if (math_wg == 0)
                asm volatile("bar.sync 2,128;" ::: "memory");
            else
                asm volatile("bar.sync 3,128;" ::: "memory");

            const uint32_t stage_base = weight_smem_addr
                + math_wg * kWeightWGBytes
                + stage * kCombinedStageBytes;
            const uint8_t* scale_ptr = weight_smem
                + math_wg * kWeightWGBytes
                + ((kt >> 2) & 1) * kCombinedStageBytes
                + kWeightStageBytes;
            float tile[kWgmmaGroups][4] = {};
            #pragma unroll
            for (int k_step = 0; k_step < kBlockK / 32; ++k_step) {
                const int common_chunk = k_step ^
                    (((row0 >> 1) + weight_swizzle_row_offset) & 3);
                const uint32_t common_address = stage_base
                    + row0 * (kBlockK / 2)
                    + common_chunk * 16 + packed_k_offset;
                const auto activation_desc = desc_128b(
                    activation_smem_addr + math_wg * kActivationWGBytes
                    + stage * kActivationStageBytes
                    + k_step * 32);
                #pragma unroll
                for (int group = 0; group < kWgmmaGroups; ++group) {
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 4; ++value_idx)
                        ptx::warpgroup_fence_operand(tile[group][value_idx]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (int group = 0; group < kWgmmaGroups; ++group) {
                    const int group_row0 = group * 64 + row0;
                    const int group_row1 = group * 64 + row1;
                    uint32_t packed0;
                    uint32_t packed1;
                    uint2 lut0;
                    uint2 lut1;
                    if (k_step == 0) {
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed0)
                            : "r"(common_address
                                  + group * 64 * (kBlockK / 2)));
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(packed1)
                            : "r"(common_address
                                  + (group * 64 + 8) * (kBlockK / 2)));
                        const uint32_t exponent0 = scale_ptr[
                            group_row0 * 16 + (kt & 3) * 4];
                        const uint32_t exponent1 = scale_ptr[
                            group_row1 * 16 + (kt & 3) * 4];
                        lut0 = synth_normalized_e2m1_lut(exponent0);
                        lut1 = synth_normalized_e2m1_lut(exponent1);
                    } else {
                        packed0 = next_packed0[group];
                        packed1 = next_packed1[group];
                        lut0 = next_lut0[group];
                        lut1 = next_lut1[group];
                    }
                    const uint2 fp8_0 =
                        dequant_weight_word<kMode2Braid>(packed0, lut0);
                    const uint2 fp8_1 =
                        dequant_weight_word<kMode2Braid>(packed1, lut1);
                    if (k_step + 1 < kBlockK / 32) {
                        const int next_step = k_step + 1;
                        const int next_common_chunk = next_step ^
                            (((row0 >> 1) + weight_swizzle_row_offset) & 3);
                        const uint32_t next_common_address = stage_base
                            + row0 * (kBlockK / 2)
                            + next_common_chunk * 16 + packed_k_offset;
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(next_packed0[group])
                            : "r"(next_common_address
                                  + group * 64 * (kBlockK / 2)));
                        asm volatile("ld.shared.b32 %0,[%1];"
                            : "=r"(next_packed1[group])
                            : "r"(next_common_address
                                  + (group * 64 + 8) * (kBlockK / 2)));
                        const uint32_t next_exponent0 = scale_ptr[
                            group_row0 * 16 + (kt & 3) * 4 + next_step];
                        const uint32_t next_exponent1 = scale_ptr[
                            group_row1 * 16 + (kt & 3) * 4 + next_step];
                        next_lut0[group] =
                            synth_normalized_e2m1_lut(next_exponent0);
                        next_lut1[group] =
                            synth_normalized_e2m1_lut(next_exponent1);
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
                    for (int value_idx = 0; value_idx < 4; ++value_idx)
                        ptx::warpgroup_fence_operand(tile[group][value_idx]);
                }
                ptx::warpgroup_wait<0>();
            }
            #pragma unroll
            for (int group = 0; group < kWgmmaGroups; ++group) {
                accum[group][0] += tile[group][0]
                    * activation_scale_smem[math_wg][stage][route_slot0];
                accum[group][1] += tile[group][1]
                    * activation_scale_smem[math_wg][stage][route_slot1];
                accum[group][2] += tile[group][2]
                    * activation_scale_smem[math_wg][stage][route_slot0];
                accum[group][3] += tile[group][3]
                    * activation_scale_smem[math_wg][stage][route_slot1];
            }

            if (math_wg == 0)
                asm volatile("bar.sync 2,128;" ::: "memory");
            else
                asm volatile("bar.sync 3,128;" ::: "memory");
            if (kt + kPairStages < kNumKTiles)
                load_weight_stage(kt + kPairStages, stage);
        }

        const int route0 = route_ids[route_slot0];
        const int route1 = route_ids[route_slot1];
        const int column_in_half_base = warp * 8 + lane / 4;
        #pragma unroll
        for (int group = 0; group < kWgmmaGroups; ++group) {
            const int column_in_group =
                math_wg * 64 + group * 32 + column_in_half_base;
            const int output_column = output_group * 128 + column_in_group;
            if (route0 < max_routes) {
                if (raw_output != nullptr) {
                    raw_output[static_cast<int64_t>(route0) * kRawN
                               + output_column] = accum[group][0];
                    raw_output[static_cast<int64_t>(route0) * kRawN
                               + Intermediate + output_column] =
                        accum[group][2];
                }
                const float gate = __bfloat162float(
                    __float2bfloat16(accum[group][0]));
                const float up = __bfloat162float(
                    __float2bfloat16(accum[group][2]));
                const __nv_bfloat16 result = __float2bfloat16(
                    gate / (1.0f + __expf(-gate)) * up);
                epilogue_value0[group] = __bfloat162float(result);
                if (bf16_output != nullptr)
                    bf16_output[static_cast<int64_t>(route0) * Intermediate
                                + output_column] = result;
            }
            if (route1 < max_routes) {
                if (raw_output != nullptr) {
                    raw_output[static_cast<int64_t>(route1) * kRawN
                               + output_column] = accum[group][1];
                    raw_output[static_cast<int64_t>(route1) * kRawN
                               + Intermediate + output_column] =
                        accum[group][3];
                }
                const float gate = __bfloat162float(
                    __float2bfloat16(accum[group][1]));
                const float up = __bfloat162float(
                    __float2bfloat16(accum[group][3]));
                const __nv_bfloat16 result = __float2bfloat16(
                    gate / (1.0f + __expf(-gate)) * up);
                epilogue_value1[group] = __bfloat162float(result);
                if (bf16_output != nullptr)
                    bf16_output[static_cast<int64_t>(route1) * Intermediate
                                + output_column] = result;
            }
        }
        const int contributor = warp * 8 + lane / 4;
        epilogue_amax[math_wg][route_slot0][contributor] =
            route0 < max_routes
            ? fmaxf(fabsf(epilogue_value0[0]),
                    fabsf(epilogue_value0[1])) : 0.0f;
        epilogue_amax[math_wg][route_slot1][contributor] =
            route1 < max_routes
            ? fmaxf(fabsf(epilogue_value1[0]),
                    fabsf(epilogue_value1[1])) : 0.0f;
    }

    __syncthreads();
    if (tid < 128) {
        const int producer_warp = tid >> 5;
        const int lane = tid & 31;
        #pragma unroll
        for (int route_slot = producer_warp;
             route_slot < kTok; route_slot += 4) {
            float value = fmaxf(
                epilogue_amax[0][route_slot][lane],
                epilogue_amax[1][route_slot][lane]);
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                value = fmaxf(value,
                    __shfl_down_sync(0xffffffffu, value, delta));
            if (lane == 0) {
                const int route = route_ids[route_slot];
                const float scale =
                    fmaxf(value, 1.0e-30f) * (1.0f / 448.0f);
                epilogue_scale_inv[route_slot] = 1.0f / scale;
                if (route < max_routes) {
                    output_scale[static_cast<int64_t>(route) * kOutputGroups
                                 + output_group] = scale;
                }
            }
        }
    }
    __syncthreads();

    {
        const int warp = mtid >> 5;
        const int lane = mtid & 31;
        const int route_slot0 = (lane & 3) * 2;
        const int route_slot1 = route_slot0 + 1;
        const int route0 = route_ids[route_slot0];
        const int route1 = route_ids[route_slot1];
        const int column_in_half_base = warp * 8 + lane / 4;
        #pragma unroll
        for (int group = 0; group < kWgmmaGroups; ++group) {
            const int output_column = output_group * 128
                + math_wg * 64 + group * 32 + column_in_half_base;
            if (route0 < max_routes) {
                quantized_output[
                    static_cast<int64_t>(route0) * Intermediate
                    + output_column] = __nv_fp8_e4m3(
                        epilogue_value0[group]
                        * epilogue_scale_inv[route_slot0]).__x;
            }
            if (route1 < max_routes) {
                quantized_output[
                    static_cast<int64_t>(route1) * Intermediate
                    + output_column] = __nv_fp8_e4m3(
                        epilogue_value1[group]
                        * epilogue_scale_inv[route_slot1]).__x;
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
__device__ __forceinline__ void reduce_swiglu_quant_task(
        const float* __restrict__ partials,
        __nv_bfloat16* __restrict__ activation,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        const int32_t* __restrict__ route_to_sorted,
        const int32_t* __restrict__ topk_ids,
        const float* __restrict__ w2_global_scale,
        int routes,
        int group) {
    static_assert(Intermediate % 128 == 0);
    constexpr int kGroupsPerRoute = Intermediate / 128;
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
    const int sorted_position = (kW2SortedAct || kW2MblockScale)
        ? __ldg(route_to_sorted + route)
        : route;
    const int quantized_row = kW2SortedAct ? sorted_position : route;
    const int quantized_index = quantized_row * Intermediate + column;

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
            float output_scale = group_scale;
            if constexpr (kW2FoldGlobalScale) {
                const int expert = __ldg(topk_ids + route);
                output_scale *= __ldg(w2_global_scale + expert);
            }
            if constexpr (kW2MblockScale) {
                const int mblock = sorted_position >> 3;
                const int slot = sorted_position & 7;
                scale[(mblock * kGroupsPerRoute + group_in_route) * 8
                      + slot] = output_scale;
            } else {
                scale[group] = output_scale;
            }
        }
    }
    __syncthreads();
    quantized[quantized_index] =
        __nv_fp8_e4m3(value / group_scale).__x;
}

template <int Intermediate, int SplitK>
__global__ __launch_bounds__(128) void reduce_swiglu_quant_kernel(
        const float* __restrict__ partials,
        __nv_bfloat16* __restrict__ activation,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        const int32_t* __restrict__ route_to_sorted,
        const int32_t* __restrict__ topk_ids,
        const float* __restrict__ w2_global_scale,
        int routes) {
    reduce_swiglu_quant_task<Intermediate, SplitK>(
        partials, activation, quantized, scale,
        route_to_sorted, topk_ids, w2_global_scale, routes,
        static_cast<int>(blockIdx.x));
}

// Route-only preparation for the serving ABI, where X and its group-128
// scales are already FP8 inputs.  One CTA constructs the exact
// E=256/top-k=6/block-M=8 layout consumed by both routed GEMMs.
__global__ __launch_bounds__(256) void route_align_kernel(
        const int32_t* __restrict__ topk_ids,
        int32_t* __restrict__ sorted_ids,
        int32_t* __restrict__ expert_ids,
        int32_t* __restrict__ num_tokens_padded,
        int32_t* __restrict__ route_to_sorted,
        int routes) {
    constexpr int kExperts = 256;
    using ExpertScan = cub::BlockScan<int, 256>;
    __shared__ int counts[kExperts];
    __shared__ int cursors[kExperts];
    __shared__ int total_padded;
    __shared__ typename ExpertScan::TempStorage scan_storage;

    const int tid = threadIdx.x;
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
            if constexpr (kW2SortedAct || kW2MblockScale)
                route_to_sorted[route] = position;
        }
    }
}

// Legacy diagnostic entry that combines route alignment and BF16 input
// quantization.  It is intentionally not used by the serving benchmark.
__global__ __launch_bounds__(256) void fused_route_quant_kernel(
        const int32_t* __restrict__ topk_ids,
        const __nv_bfloat16* __restrict__ input,
        int32_t* __restrict__ sorted_ids,
        int32_t* __restrict__ expert_ids,
        int32_t* __restrict__ num_tokens_padded,
        uint8_t* __restrict__ quantized,
        float* __restrict__ scale,
        int32_t* __restrict__ route_to_sorted,
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
                if constexpr (kW2SortedAct || kW2MblockScale)
                    route_to_sorted[route] = position;
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

// 128-thread route preparation body for the resident single-launch kernel.
// CTA 0 builds the exact E=256/block-M=8 route pool.  X and X scales are
// caller-owned FP8 inputs, so the following grid barrier only publishes route
// metadata to the W13 phase.
__device__ __forceinline__ void single_launch_route_task(
        const int32_t* __restrict__ topk_ids,
        int32_t* __restrict__ sorted_ids,
        int32_t* __restrict__ expert_ids,
        int32_t* __restrict__ num_tokens_padded,
        int32_t* __restrict__ route_to_sorted,
        int tokens, int linear_block_idx) {
    constexpr int kExperts = 256;
    __shared__ int counts[kExperts];
    __shared__ int cursors[kExperts];
    __shared__ int total_padded;

    const int tid = threadIdx.x;
    const int routes = tokens * kTopK;
    if (linear_block_idx == 0) {
        for (int expert = tid; expert < kExperts; expert += 128)
            counts[expert] = 0;
        __syncthreads();

        for (int route = tid; route < routes; route += 128) {
            const int expert = __ldg(topk_ids + route);
            if (static_cast<unsigned>(expert) < kExperts)
                atomicAdd(counts + expert, 1);
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
            for (int expert = 0; expert < kExperts; ++expert) {
                cursors[expert] = offset;
                const int padded_count = (counts[expert] + 7) & ~7;
                for (int position = offset;
                     position < offset + padded_count; position += 8)
                    expert_ids[position >> 3] = expert;
                offset += padded_count;
            }
            total_padded = offset;
            *num_tokens_padded = offset;
        }
        __syncthreads();

        for (int position = tid; position < total_padded; position += 128)
            sorted_ids[position] = routes;
        __syncthreads();

        for (int route = tid; route < routes; route += 128) {
            const int expert = __ldg(topk_ids + route);
            if (static_cast<unsigned>(expert) < kExperts) {
                const int position = atomicAdd(cursors + expert, 1);
                sorted_ids[position] = route;
                if constexpr (kW2SortedAct || kW2MblockScale)
                    route_to_sorted[route] = position;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int32_t load_acquire_gpu_i32(
        const int32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.acquire.gpu.global.u32 %0,[%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return static_cast<int32_t>(value);
}

__device__ __forceinline__ void store_release_gpu_i32(
        int32_t* pointer, int32_t value) {
    asm volatile(
        "st.release.gpu.global.u32 [%0],%1;"
        : : "l"(pointer), "r"(static_cast<uint32_t>(value)) : "memory");
}

__device__ __forceinline__ int32_t atomic_add_release_gpu_i32(
        int32_t* pointer, int32_t value) {
    uint32_t old;
    asm volatile(
        "atom.release.gpu.global.add.u32 %0,[%1],%2;"
        : "=r"(old)
        : "l"(pointer), "r"(static_cast<uint32_t>(value))
        : "memory");
    return static_cast<int32_t>(old);
}

// Scheduler-only correctness probe for the TP-local interleaved design.
// One lane per persistent CTA claims dynamically bounded W13 tasks.  The last
// W13 tile for an M block publishes that block into a ready queue; W2 tasks
// can only be claimed from published queue entries.  No task bound depends on
// host inspection of the route distribution.
__global__ void interleaved_scheduler_probe_kernel(
        const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ num_tokens_padded,
        int32_t* __restrict__ counters,
        int32_t* __restrict__ readiness,
        int32_t* __restrict__ ready_queue,
        int32_t* __restrict__ ready_valid,
        int32_t* __restrict__ w13_owner,
        int32_t* __restrict__ w13_order,
        int32_t* __restrict__ w2_owner,
        int32_t* __restrict__ w2_mblock,
        int32_t* __restrict__ w2_order,
        int max_mblocks, int w13_tiles, int w2_tiles) {
    if (threadIdx.x != 0)
        return;

    const int num_mblocks = __ldg(num_tokens_padded) / kTok;
    if (blockIdx.x == 0)
        counters[7] = num_mblocks;
    const int total_w13 = num_mblocks * w13_tiles;
    const int total_w2 = num_mblocks * w2_tiles;

    while (true) {
        // Prefer a ready W2 task so W2 follows completed W13 blocks instead
        // of waiting for the complete W13 task space to drain.
        bool claimed_w2 = false;
        while (true) {
            const int next_w2 = atomicAdd(counters + 1, 0);
            const int published = atomicAdd(counters + 2, 0);
            const int queue_slot = next_w2 / w2_tiles;
            if (queue_slot >= published
                    || load_acquire_gpu_i32(ready_valid + queue_slot) == 0)
                break;
            if (atomicCAS(counters + 1, next_w2, next_w2 + 1) != next_w2)
                continue;

            const int n_tile = next_w2 - queue_slot * w2_tiles;
            const int mblock =
                load_acquire_gpu_i32(ready_queue + queue_slot);
            if (static_cast<unsigned>(mblock) >=
                    static_cast<unsigned>(num_mblocks)) {
                atomicAdd(counters + 6, 1);
                atomicAdd(counters + 9, 1);
            } else if (atomicAdd(readiness + mblock, 0) != w13_tiles) {
                atomicAdd(counters + 6, 1);
                atomicAdd(counters + 10, 1);
            }
            w2_owner[next_w2] = blockIdx.x;
            w2_mblock[next_w2] = mblock;
            w2_order[next_w2] = atomicAdd(counters + 3, 1);
            atomicAdd(counters + 5, 1);
            claimed_w2 = true;
            (void)n_tile;
            break;
        }
        if (claimed_w2)
            continue;

        int w13_task = -1;
        while (true) {
            const int next_w13 = atomicAdd(counters + 0, 0);
            if (next_w13 >= total_w13)
                break;
            if (atomicCAS(counters + 0, next_w13, next_w13 + 1)
                    == next_w13) {
                w13_task = next_w13;
                break;
            }
        }
        if (w13_task >= 0) {
            const int mblock = w13_task / w13_tiles;
            if (static_cast<unsigned>(mblock) >=
                    static_cast<unsigned>(max_mblocks)
                    || __ldg(expert_ids + mblock) < 0) {
                atomicAdd(counters + 6, 1);
                atomicAdd(counters + 8, 1);
            }
            w13_owner[w13_task] = blockIdx.x;
            w13_order[w13_task] = atomicAdd(counters + 3, 1);
            __threadfence();
            const int done = atomicAdd(readiness + mblock, 1) + 1;
            atomicAdd(counters + 4, 1);
            if (done == w13_tiles) {
                const int queue_slot = atomicAdd(counters + 2, 1);
                ready_queue[queue_slot] = mblock;
                store_release_gpu_i32(ready_valid + queue_slot, 1);
            } else if (done > w13_tiles) {
                atomicAdd(counters + 6, 1);
                atomicAdd(counters + 11, 1);
            }
            continue;
        }

        if (atomicAdd(counters + 4, 0) == total_w13
                && atomicAdd(counters + 5, 0) == total_w2)
            break;
    }
}

__global__ void cast_bf16_kernel(
        const float* __restrict__ input,
        __nv_bfloat16* __restrict__ output,
        int numel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel)
        output[index] = __float2bfloat16(input[index]);
}

// DeepSeek-V4-Flash has a fixed k=6 and H=4096 local route reduction.
// Unlike the rejected one-CTA-per-token prototype, split each token across
// multiple independent hidden tiles so M=8 still exposes at least 64 CTAs.
// Each BF16 pair has a unique writer and accumulates routes in the same order
// and precision as SGLang's moe_fused_mul_sum Triton kernel.
template <int Threads, int VecPairs>
__global__ __launch_bounds__(Threads) void tiled_k6_reduce_kernel(
        const __nv_bfloat16* __restrict__ input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ output,
        int tokens) {
    constexpr int kHidden = 4096;
    constexpr int kPairs = kHidden / 2;
    constexpr int kTilePairs = Threads * VecPairs;
    static_assert(kPairs % kTilePairs == 0);
    const int token = blockIdx.y;
    if (token >= tokens)
        return;

    const int pair0 = blockIdx.x * kTilePairs + threadIdx.x;
    float2 accum[VecPairs];
    #pragma unroll
    for (int vec = 0; vec < VecPairs; ++vec)
        accum[vec] = make_float2(0.0f, 0.0f);

    const auto* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
    #pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        const float route_weight =
            __ldg(topk_weights + token * kTopK + route) * kRoutedScale;
        const int64_t route_base =
            (static_cast<int64_t>(token) * kTopK + route) * kPairs;
        #pragma unroll
        for (int vec = 0; vec < VecPairs; ++vec) {
            const int pair = pair0 + vec * Threads;
            const float2 value = __bfloat1622float2(input2[route_base + pair]);
            accum[vec].x = fmaf(value.x, route_weight, accum[vec].x);
            accum[vec].y = fmaf(value.y, route_weight, accum[vec].y);
        }
    }

    auto* output2 = reinterpret_cast<__nv_bfloat162*>(output);
    const int64_t output_base = static_cast<int64_t>(token) * kPairs;
    #pragma unroll
    for (int vec = 0; vec < VecPairs; ++vec) {
        const int pair = pair0 + vec * Threads;
        output2[output_base + pair] =
            __floats2bfloat162_rn(accum[vec].x, accum[vec].y);
    }
}

__device__ __forceinline__ uint4 load_relaxed_sys_16b(const void* pointer) {
    uint4 value;
    asm volatile(
        "ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ void store_relaxed_sys_16b(
        void* pointer, const uint4& value) {
    asm volatile(
        "st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
        :
        : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
          "l"(pointer)
        : "memory");
}

__device__ __forceinline__ void store_multimem_16b(
        void* pointer, const uint4& value) {
    const float4 bits = *reinterpret_cast<const float4*>(&value);
    asm volatile(
        "multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
        :
        : "f"(bits.x), "f"(bits.y), "f"(bits.z), "f"(bits.w),
          "l"(pointer)
        : "memory");
}

__device__ __forceinline__ bool word_has_positive_bf16_zero(uint32_t word) {
    return (word & 0xffffu) == 0u || (word & 0xffff0000u) == 0u;
}

__device__ __forceinline__ uint32_t clear_positive_bf16_zero(uint32_t word) {
    if ((word & 0xffffu) == 0u)
        word |= 0x00008000u;
    if ((word & 0xffff0000u) == 0u)
        word |= 0x80000000u;
    return word;
}

// Low-footprint consumers overlap local k6 reduction and multicast stores
// with the unfragmented W2 grid.  Each static token/N-chunk task waits on its
// direct route/tile markers.  The optional inline finish first gates remote
// polling on completion of every local worker, avoiding cross-rank task-order
// cycles while removing the separate finish launch.
template <int Threads, int NumChunks, bool InlineFinish>
__global__ __launch_bounds__(Threads) void progress_k6_mc_push_tp4_kernel(
        const __nv_bfloat16* __restrict__ input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ output,
        int32_t* __restrict__ progress_state,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* __restrict__ push_mc,
        int tokens, int rank, int64_t push_stride) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kPairsPerToken = kHidden / 2;
    constexpr int kNumW2Tiles = 32;
    constexpr int kNumChunks = NumChunks;
    constexpr int kVecsPerChunk = (kHidden / kNumChunks) / 8;
    constexpr int kPairsPerVec = 4;
    static_assert(kNumChunks == 2 || kNumChunks == 4 || kNumChunks == 8);
    static_assert(Threads == kVecsPerChunk);

    const int total_tasks = tokens * kNumChunks;
    int32_t* task_done =
        progress_state + tokens * kTopK * kNumW2Tiles;
    int32_t* worker_done = task_done + 1;
    const int phase = push_counter[0] & 1u;
    const int64_t phase_offset =
        static_cast<int64_t>(phase) * push_stride * kWorld;

    for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        if (threadIdx.x < kTopK * (kNumW2Tiles / kNumChunks)) {
            constexpr int kTilesPerChunk = kNumW2Tiles / kNumChunks;
            const int route_slot = threadIdx.x / kTilesPerChunk;
            const int tile_in_chunk = threadIdx.x % kTilesPerChunk;
            const int token = task / kNumChunks;
            const int chunk = task - token * kNumChunks;
            const int route = token * kTopK + route_slot;
            const int tile = chunk * kTilesPerChunk + tile_in_chunk;
            while (progress_load_acquire(
                       progress_state + route * kNumW2Tiles + tile) == 0) {}
        }
        __syncthreads();

        const int token = task / kNumChunks;
        const int chunk = task - token * kNumChunks;
        const int pair0 = chunk * (kPairsPerToken / kNumChunks)
            + threadIdx.x * kPairsPerVec;
        float2 accum[kPairsPerVec];
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair)
            accum[pair] = make_float2(0.0f, 0.0f);
        const auto* input2 =
            reinterpret_cast<const __nv_bfloat162*>(input);
        #pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const float route_weight =
                __ldg(topk_weights + token * kTopK + route) * kRoutedScale;
            const int64_t route_base =
                (static_cast<int64_t>(token) * kTopK + route)
                * kPairsPerToken;
            #pragma unroll
            for (int pair = 0; pair < kPairsPerVec; ++pair) {
                const float2 value = __bfloat1622float2(
                    input2[route_base + pair0 + pair]);
                accum[pair].x =
                    fmaf(value.x, route_weight, accum[pair].x);
                accum[pair].y =
                    fmaf(value.y, route_weight, accum[pair].y);
            }
        }

        uint4 local_vec;
        uint32_t* local_words = reinterpret_cast<uint32_t*>(&local_vec);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(accum[pair].x, accum[pair].y);
            const uint32_t word =
                *reinterpret_cast<const uint32_t*>(&value);
            local_words[pair] = word == 0u ? 0x00008000u : word;
        }
        const int vec = token * (kHidden / 8)
            + chunk * kVecsPerChunk + threadIdx.x;
        const int64_t source_offset =
            static_cast<int64_t>(rank) * push_stride + phase_offset
            + static_cast<int64_t>(vec) * 16;
        store_multimem_16b(push_mc + source_offset, local_vec);
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(task_done, 1);
    }
    if (threadIdx.x == 0)
        atomicAdd(worker_done, 1);

    if constexpr (InlineFinish) {
        // A grid smaller than the H20 SM count is fully resident once every
        // block has entered.  Wait only for local publication here; unlike a
        // per-task remote wait, this cannot form a cross-rank admission cycle.
        __syncthreads();
        if (threadIdx.x == 0) {
            while (progress_load_acquire(worker_done) != gridDim.x) {}
        }
        __syncthreads();

        uint8_t* peer_base[kWorld] = {push0, push1, push2, push3};
        for (int task = blockIdx.x; task < total_tasks; task += gridDim.x) {
            const int token = task / kNumChunks;
            const int chunk = task - token * kNumChunks;
            const int vec = token * (kHidden / 8)
                + chunk * kVecsPerChunk + threadIdx.x;
            uint4 rank_vec[kWorld];
            const int64_t poll_offset = phase_offset
                + static_cast<int64_t>(vec) * 16;
            while (true) {
                #pragma unroll
                for (int source = 0; source < kWorld; ++source) {
                    rank_vec[source] = load_relaxed_sys_16b(
                        peer_base[rank] + source * push_stride + poll_offset);
                }
                bool has_zero = false;
                #pragma unroll
                for (int source = 0; source < kWorld; ++source) {
                    const uint32_t* words =
                        reinterpret_cast<const uint32_t*>(&rank_vec[source]);
                    #pragma unroll
                    for (int pair = 0; pair < kPairsPerVec; ++pair)
                        has_zero |= words[pair] == 0u;
                }
                if (!has_zero)
                    break;
            }

            uint4 result;
            uint32_t* result_words = reinterpret_cast<uint32_t*>(&result);
            #pragma unroll
            for (int pair = 0; pair < kPairsPerVec; ++pair) {
                float2 sum = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (int source = 0; source < kWorld; ++source) {
                    const uint32_t word =
                        reinterpret_cast<const uint32_t*>(&rank_vec[source])[pair];
                    const __nv_bfloat162 value =
                        *reinterpret_cast<const __nv_bfloat162*>(&word);
                    const float2 value_f32 = __bfloat1622float2(value);
                    if (source == 0)
                        sum = value_f32;
                    else {
                        sum.x += value_f32.x;
                        sum.y += value_f32.y;
                    }
                }
                const __nv_bfloat162 value =
                    __floats2bfloat162_rn(sum.x, sum.y);
                result_words[pair] =
                    *reinterpret_cast<const uint32_t*>(&value);
            }
            reinterpret_cast<uint4*>(output)[vec] = result;

            const uint4 empty = make_uint4(0u, 0u, 0u, 0u);
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                *reinterpret_cast<uint4*>(
                    peer_base[rank] + source * push_stride + poll_offset) = empty;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int counter = blockIdx.x; counter < 78;
                 counter += gridDim.x)
                atomicAdd(push_counter + counter, 1u);
        }
    }
}

template <int Threads>
__global__ __launch_bounds__(Threads) void progress_mc_push_finish_tp4_kernel(
        __nv_bfloat16* __restrict__ output,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        int tokens, int rank, int64_t push_stride) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kVecsPerToken = kHidden / 8;
    constexpr int kPairsPerVec = 4;
    const int num_vecs = tokens * kVecsPerToken;
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_threads = gridDim.x * blockDim.x;
    const int phase = push_counter[blockIdx.x] & 1u;
    uint8_t* peer_base[kWorld] = {push0, push1, push2, push3};
    const int64_t phase_offset =
        static_cast<int64_t>(phase) * push_stride * kWorld;

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        uint4 rank_vec[kWorld];
        const int64_t poll_offset =
            phase_offset + static_cast<int64_t>(vec) * 16;
        while (true) {
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                rank_vec[source] = load_relaxed_sys_16b(
                    peer_base[rank] + source * push_stride + poll_offset);
            }
            bool has_zero = false;
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                const uint32_t* words =
                    reinterpret_cast<const uint32_t*>(&rank_vec[source]);
                #pragma unroll
                for (int pair = 0; pair < kPairsPerVec; ++pair)
                    has_zero |= words[pair] == 0u;
            }
            if (!has_zero)
                break;
        }

        uint4 result;
        uint32_t* result_words = reinterpret_cast<uint32_t*>(&result);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            float2 sum = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                const uint32_t word =
                    reinterpret_cast<const uint32_t*>(&rank_vec[source])[pair];
                const __nv_bfloat162 value =
                    *reinterpret_cast<const __nv_bfloat162*>(&word);
                const float2 value_f32 = __bfloat1622float2(value);
                if (source == 0)
                    sum = value_f32;
                else {
                    sum.x += value_f32.x;
                    sum.y += value_f32.y;
                }
            }
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(sum.x, sum.y);
            result_words[pair] =
                *reinterpret_cast<const uint32_t*>(&value);
        }
        reinterpret_cast<uint4*>(output)[vec] = result;

        const uint4 empty = make_uint4(0u, 0u, 0u, 0u);
        #pragma unroll
        for (int source = 0; source < kWorld; ++source) {
            *reinterpret_cast<uint4*>(
                peer_base[rank] + source * push_stride + poll_offset) = empty;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(push_counter + blockIdx.x, 1u);
}

// Fuse the local fixed-k6 route sum with SGLang's TP4 1shot-push protocol.
// The symmetric slots and per-CTA phase counters are owned by the unchanged
// CustomAllReduceV2 communicator, so stock Humming and this kernel can safely
// alternate on the same communicator and inside separately captured graphs.
template <int Threads, bool UseMulticast, bool Chunked = false>
__device__ __forceinline__ void fused_k6_push_ar_tp4_task(
        const __nv_bfloat16* __restrict__ input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ output,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* push_mc,
        int tokens, int rank, int64_t push_stride,
        int hidden_offset, int hidden_size,
        int linear_block_idx, int linear_grid_dim) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kPairsPerToken = kHidden / 2;
    constexpr int kPairsPerVec = 8 / 2;
    constexpr int kVecsPerToken = kHidden / 8;
    const int vecs_per_token =
        Chunked ? hidden_size / 8 : kVecsPerToken;
    const int num_vecs = tokens * vecs_per_token;
    const int global_tid = linear_block_idx * Threads + threadIdx.x;
    const int global_threads = linear_grid_dim * Threads;
    const int phase = push_counter[linear_block_idx] & 1u;
    uint8_t* peer_base[kWorld] = {push0, push1, push2, push3};
    const int64_t phase_offset =
        static_cast<int64_t>(phase) * push_stride * kWorld;

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const int token = vec / vecs_per_token;
        const int vec_in_token = vec - token * vecs_per_token;
        const int pair0 = (Chunked ? hidden_offset / 2 : 0)
            + vec_in_token * kPairsPerVec;
        float2 accum[kPairsPerVec];
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair)
            accum[pair] = make_float2(0.0f, 0.0f);

        const auto* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
        #pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const float route_weight =
                __ldg(topk_weights + token * kTopK + route) * kRoutedScale;
            const int64_t route_base =
                (static_cast<int64_t>(token) * kTopK + route)
                * kPairsPerToken;
            #pragma unroll
            for (int pair = 0; pair < kPairsPerVec; ++pair) {
                const float2 value = __bfloat1622float2(
                    input2[route_base + pair0 + pair]);
                accum[pair].x =
                    fmaf(value.x, route_weight, accum[pair].x);
                accum[pair].y =
                    fmaf(value.y, route_weight, accum[pair].y);
            }
        }

        uint4 local_vec;
        uint32_t* local_words = reinterpret_cast<uint32_t*>(&local_vec);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(accum[pair].x, accum[pair].y);
            uint32_t word = *reinterpret_cast<const uint32_t*>(&value);
            if constexpr (UseMulticast) {
                // K3's push protocol only needs to distinguish an untouched
                // 4-byte atom from a written one.  If both BF16 lanes are
                // +0, turn one into -0; mixed pairs are already nonzero.
                local_words[pair] = word == 0u ? 0x00008000u : word;
            } else {
                local_words[pair] = clear_positive_bf16_zero(word);
            }
        }

        const int64_t source_offset =
            static_cast<int64_t>(rank) * push_stride + phase_offset
            + static_cast<int64_t>(vec) * 16;
        if constexpr (UseMulticast) {
            store_multimem_16b(push_mc + source_offset, local_vec);
        } else {
            #pragma unroll
            for (int peer = 0; peer < kWorld; ++peer)
                store_relaxed_sys_16b(
                    peer_base[peer] + source_offset, local_vec);
        }

        uint4 rank_vec[kWorld];
        const int64_t poll_phase_offset = phase_offset
            + static_cast<int64_t>(vec) * 16;
        do {
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                rank_vec[source] = load_relaxed_sys_16b(
                    peer_base[rank] + source * push_stride
                    + poll_phase_offset);
            }
            bool has_zero = false;
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                const uint32_t* words =
                    reinterpret_cast<const uint32_t*>(&rank_vec[source]);
                #pragma unroll
                for (int pair = 0; pair < kPairsPerVec; ++pair) {
                    if constexpr (UseMulticast)
                        has_zero |= words[pair] == 0u;
                    else
                        has_zero |= word_has_positive_bf16_zero(words[pair]);
                }
            }
            if (!has_zero)
                break;
        } while (true);

        uint4 result;
        uint32_t* result_words = reinterpret_cast<uint32_t*>(&result);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            float2 sum = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                const uint32_t word =
                    reinterpret_cast<const uint32_t*>(&rank_vec[source])[pair];
                const __nv_bfloat162 value =
                    *reinterpret_cast<const __nv_bfloat162*>(&word);
                const float2 value_f32 = __bfloat1622float2(value);
                if (source == 0) {
                    sum = value_f32;
                } else {
                    sum.x += value_f32.x;
                    sum.y += value_f32.y;
                }
            }
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(sum.x, sum.y);
            result_words[pair] = *reinterpret_cast<const uint32_t*>(&value);
        }
        const int64_t output_vec =
            static_cast<int64_t>(token) * kVecsPerToken
            + (Chunked ? hidden_offset / 8 : 0) + vec_in_token;
        reinterpret_cast<uint4*>(output)[output_vec] = result;

        const uint4 empty = make_uint4(0u, 0u, 0u, 0u);
        #pragma unroll
        for (int source = 0; source < kWorld; ++source) {
            reinterpret_cast<uint4*>(
                peer_base[rank] + source * push_stride + phase_offset)[vec]
                = empty;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(push_counter + linear_block_idx, 1u);
}

// Keep the selected standalone launch as a thin wrapper around the same
// device body used by the forthcoming persistent TP MegaMoE kernel.  Passing
// the logical grid explicitly avoids child launches and lets a resident grid
// reuse the exact validated CustomAllReduceV2 phase-counter protocol.
template <int Threads, bool UseMulticast, bool Chunked = false>
__global__ __launch_bounds__(Threads) void fused_k6_push_ar_tp4_kernel(
        const __nv_bfloat16* __restrict__ input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ output,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* push_mc,
        int tokens, int rank, int64_t push_stride,
        int hidden_offset, int hidden_size) {
    fused_k6_push_ar_tp4_task<Threads, UseMulticast, Chunked>(
        input, topk_weights, output, push_counter,
        push0, push1, push2, push3, push_mc,
        tokens, rank, push_stride, hidden_offset, hidden_size,
        static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
}

template <int Threads>
__device__ __forceinline__ void fused_k6_nvls_pull_tp4_task(
        const __nv_bfloat16* __restrict__ route_input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ symm_input,
        const uint8_t* __restrict__ symm_input_mc,
        __nv_bfloat16* __restrict__ output,
        uint8_t* __restrict__ sem_local,
        uint8_t* __restrict__ sem_mc,
        int tokens, int linear_block_idx, int linear_grid_dim);

// Reusable local-rank grid barrier.  Every CTA first publishes all lanes'
// writes, then one lane contributes to a generation-counted barrier.  Each
// phase owns a separate count/epoch pair, so CUDA-graph replays need no memset
// node and an epoch can advance indefinitely with the stable graph storage.
__device__ __forceinline__ void single_launch_grid_barrier(
        int32_t* __restrict__ state, int phase, int expected_blocks) {
    __shared__ int observed_epoch;
    // A fence is per CUDA thread.  Every lane that produced ordinary global
    // stores must publish its own writes before lane 0 announces CTA arrival.
    // The old lane-0-only fence was sufficient in practice for the staged
    // bring-up, but was not a valid publication primitive for a task DAG.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t* count = state + phase * 2;
        int32_t* epoch = count + 1;
        observed_epoch = load_acquire_gpu_i32(epoch);
        const int arrival = atomicAdd(count, 1);
        if (arrival == expected_blocks - 1) {
            atomicExch(count, 0);
            store_release_gpu_i32(epoch, observed_epoch + 1);
        } else {
            while (load_acquire_gpu_i32(epoch) == observed_epoch) {
            }
        }
    }
    __syncthreads();
}

enum SingleLaunchSchedulerOffset : int {
    kSchedulerNextW13 = 0,
    kSchedulerW13QueueTail = 1,
    kSchedulerNextActivation = 2,
    kSchedulerW2QueueTail = 3,
    kSchedulerNextW2 = 4,
    kSchedulerDoneW13 = 5,
    kSchedulerDoneActivation = 6,
    kSchedulerDoneW2 = 7,
    kSchedulerHeaderWords = 8,
};

// First complete bring-up of the real one-launch TP4 path.  It intentionally
// keeps phase barriers between the already validated kernels before the next
// optimization step introduces the MegaMoE task DAG and W13/W2 overlap.  The
// one global launch nevertheless owns route preparation, intermediate
// requantization, both MXFP4 GEMMs, SwiGLU, fixed-k6 reduction and multicast
// all-reduce.  Input X is already FP8 at this API boundary.
template <int SplitK>
__global__ __launch_bounds__(128, 4) void tp4_megamoe_single_launch_kernel(
        const __grid_constant__ CUtensorMap w13_tma_weight,
        const __grid_constant__ CUtensorMap w13_tma_weight_scale,
        const __grid_constant__ CUtensorMap w2_tma_weight,
        const __grid_constant__ CUtensorMap w2_tma_weight_scale,
        const uint8_t* __restrict__ w13,
        const uint8_t* __restrict__ s13,
        const float* __restrict__ g13,
        const uint8_t* __restrict__ w2,
        const uint8_t* __restrict__ s2,
        const float* __restrict__ g2,
        const uint8_t* __restrict__ qx,
        const float* __restrict__ x_scale,
        const int32_t* __restrict__ topk_ids,
        const float* __restrict__ topk_weights,
        int32_t* __restrict__ sorted_ids,
        int32_t* __restrict__ expert_ids,
        int32_t* __restrict__ num_tokens_padded,
        float* __restrict__ partials,
        __nv_bfloat16* __restrict__ activation,
        uint8_t* __restrict__ qactivation,
        float* __restrict__ activation_scale,
        __nv_bfloat16* __restrict__ down,
        const uint2* __restrict__ lut,
        int32_t* __restrict__ barrier_state,
        int32_t* __restrict__ route_to_sorted,
        __nv_bfloat16* __restrict__ output,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* push_mc,
        __nv_bfloat16* __restrict__ pull_input,
        const uint8_t* __restrict__ pull_input_mc,
        uint8_t* __restrict__ pull_sem_local,
        uint8_t* __restrict__ pull_sem_mc,
        int tokens, int max_mblocks, int rank, int64_t push_stride) {
    constexpr int kIntermediate = 512;
    constexpr int kW13NTiles = (2 * kIntermediate) / kWout;
    constexpr int kW2NTiles = 4096 / kWout;
    const int cta = static_cast<int>(blockIdx.x);
    const int ctas = static_cast<int>(gridDim.x);
    const int routes = tokens * kTopK;

    // barrier_state[0:8] remains the generation-counted grid-barrier slab.
    // The suffix is graph-stable scheduler storage, reset cooperatively by
    // this kernel on every replay: eight counters followed by one W13-done
    // count and two release-published queues per possible routed M block.
    int32_t* scheduler = barrier_state + 8;
    if constexpr (kSingleLaunchInterleaved) {
        const int scheduler_words = kSchedulerHeaderWords + 3 * max_mblocks;
        for (int word = cta * blockDim.x + threadIdx.x;
             word < scheduler_words; word += ctas * blockDim.x)
            scheduler[word] = 0;
    }

    single_launch_route_task(
        topk_ids, sorted_ids, expert_ids, num_tokens_padded,
        route_to_sorted, tokens, cta);
    single_launch_grid_barrier(barrier_state, 0, ctas);

    if constexpr (kSingleLaunchInterleaved) {
        static_assert(SplitK == 2 || SplitK == 4);
        constexpr int kSplitPairs = SplitK / 2;
        constexpr int kW13GroupsPerMblock = kW13NTiles * kSplitPairs;
        constexpr int kActivationGroupsPerRoute = kIntermediate / 128;

        const int num_mblocks = __ldg(num_tokens_padded) / kTok;
        const int total_w13_groups = num_mblocks * kW13GroupsPerMblock;
        const int total_w2_tasks = num_mblocks * kW2NTiles;
        int32_t* w13_done = scheduler + kSchedulerHeaderWords;
        int32_t* activation_queue = w13_done + max_mblocks;
        int32_t* w2_queue = activation_queue + max_mblocks;

        // One block-wide mailbox lets lane 0 perform global scheduling while
        // all 128 lanes remain available to the selected WGMMA/quant task.
        __shared__ int scheduled_kind;
        __shared__ int scheduled_index;
        __shared__ int scheduled_mblock;
        bool w13_exhausted = false;

        while (true) {
            if (threadIdx.x == 0) {
                int kind = -1;
                int index = -1;
                int mblock = -1;

                while (kind < 0) {
                    if (atomicAdd(scheduler + kSchedulerDoneW2, 0)
                            == total_w2_tasks) {
                        kind = 0;
                        break;
                    }

                    // First drain any W2 tile whose activation block has
                    // already been release-published.  The queue encodes
                    // mblock+1 so zero remains the reset/not-ready sentinel.
                    const int next_w2 =
                        atomicAdd(scheduler + kSchedulerNextW2, 0);
                    if (next_w2 < total_w2_tasks) {
                        const int queue_slot = next_w2 / kW2NTiles;
                        const int encoded = load_acquire_gpu_i32(
                            w2_queue + queue_slot);
                        if (encoded != 0
                                && atomicCAS(
                                    scheduler + kSchedulerNextW2,
                                    next_w2, next_w2 + 1) == next_w2) {
                            kind = 3;
                            index = next_w2 - queue_slot * kW2NTiles;
                            mblock = encoded - 1;
                            break;
                        }
                    }

                    // One CTA performs all four group-128 quant groups for
                    // each valid route in a ready mblock.  This keeps every
                    // route's scales complete before any W2 N tile sees it.
                    const int next_activation = atomicAdd(
                        scheduler + kSchedulerNextActivation, 0);
                    if (next_activation < num_mblocks) {
                        const int encoded = load_acquire_gpu_i32(
                            activation_queue + next_activation);
                        if (encoded != 0
                                && atomicCAS(
                                    scheduler + kSchedulerNextActivation,
                                    next_activation,
                                    next_activation + 1)
                                    == next_activation) {
                            kind = 2;
                            index = next_activation;
                            mblock = encoded - 1;
                            break;
                        }
                    }

                    // Pair adjacent split-K slices under one scheduler claim.
                    // M8 still exposes 16 tasks/mblock (typically >390 total)
                    // while halving global claim/publication overhead.
                    if (!w13_exhausted) {
                        const int next_w13 = atomicAdd(
                            scheduler + kSchedulerNextW13, 1);
                        if (next_w13 < total_w13_groups) {
                            kind = 1;
                            index = next_w13;
                            mblock = next_w13 / kW13GroupsPerMblock;
                            break;
                        }
                        w13_exhausted = true;
                    }

                    // All upstream work may be claimed but not yet complete.
                    // Poll with one lane while the remaining lanes sleep at
                    // the mailbox barrier; no resident CTA is stranded on a
                    // specific not-yet-ready downstream task.
                    __nanosleep(64);
                }

                scheduled_kind = kind;
                scheduled_index = index;
                scheduled_mblock = mblock;
            }
            __syncthreads();

            const int kind = scheduled_kind;
            if (kind == 0)
                break;

            if (kind == 1) {
                const int local_group = scheduled_index
                    - scheduled_mblock * kW13GroupsPerMblock;
                const int n_tile = local_group / kSplitPairs;
                const int split_pair = local_group - n_tile * kSplitPairs;
                const int first_task =
                    (scheduled_mblock * kW13NTiles + n_tile) * SplitK
                    + split_pair * 2;
                #pragma unroll
                for (int split_in_pair = 0; split_in_pair < 2;
                     ++split_in_pair) {
                    route_gemm_task<4096, 1024, SplitK, true>(
                        &w13_tma_weight, &w13_tma_weight_scale,
                        w13, s13, g13, qx, x_scale,
                        sorted_ids, expert_ids, num_tokens_padded,
                        topk_weights, partials, lut, nullptr,
                        routes, 0, first_task + split_in_pair);
                    __syncthreads();
                }

                // The final CTA barrier after the paired GEMM calls strongly
                // happens-before lane 0's GPU-scope release atomic.  One
                // release therefore publishes every lane's ordinary global
                // partial store without 128 redundant device-wide fences.
                if (threadIdx.x == 0) {
                    const int done = atomic_add_release_gpu_i32(
                        w13_done + scheduled_mblock, 1) + 1;
                    atomicAdd(scheduler + kSchedulerDoneW13, 1);
                    if (done == kW13GroupsPerMblock) {
                        const int queue_slot = atomicAdd(
                            scheduler + kSchedulerW13QueueTail, 1);
                        store_release_gpu_i32(
                            activation_queue + queue_slot,
                            scheduled_mblock + 1);
                    }
                }
            } else if (kind == 2) {
                #pragma unroll
                for (int route_slot = 0; route_slot < kTok; ++route_slot) {
                    const int route = __ldg(
                        sorted_ids + scheduled_mblock * kTok + route_slot);
                    if (route < routes) {
                        #pragma unroll
                        for (int group_in_route = 0;
                             group_in_route < kActivationGroupsPerRoute;
                             ++group_in_route) {
                            reduce_swiglu_quant_task<kIntermediate, SplitK>(
                                partials, activation, qactivation,
                                activation_scale, route_to_sorted, topk_ids,
                                g2, routes,
                                route * kActivationGroupsPerRoute
                                    + group_in_route);
                            __syncthreads();
                        }
                    }
                }

                // Publish all quantized values/scales with one CTA release,
                // mirroring the W13 readiness protocol above.
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(scheduler + kSchedulerDoneActivation, 1);
                    const int queue_slot = atomicAdd(
                        scheduler + kSchedulerW2QueueTail, 1);
                    store_release_gpu_i32(
                        w2_queue + queue_slot, scheduled_mblock + 1);
                }
            } else {
                const int task =
                    scheduled_mblock * kW2NTiles + scheduled_index;
                route_gemm_task<512, 4096, 1, false>(
                    &w2_tma_weight, &w2_tma_weight_scale,
                    w2, s2, g2, qactivation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, topk_weights,
                    reinterpret_cast<float*>(down), lut, nullptr,
                    routes, 0, task);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(scheduler + kSchedulerDoneW2, 1);
            }
            __syncthreads();
        }

        // This remaining barrier is the only compute-wide phase boundary:
        // every lane publishes W2 route rows before the communication CTAs
        // begin k6 reduction and the TP collective.
        single_launch_grid_barrier(barrier_state, 3, ctas);
    } else {
        const int w13_tasks = max_mblocks * kW13NTiles * SplitK;
        for (int task = cta; task < w13_tasks; task += ctas) {
            route_gemm_task<4096, 1024, SplitK, true>(
                &w13_tma_weight, &w13_tma_weight_scale,
                w13, s13, g13, qx, x_scale,
                sorted_ids, expert_ids, num_tokens_padded, topk_weights,
                partials, lut, nullptr, routes, 0, task);
            __syncthreads();
        }
        single_launch_grid_barrier(barrier_state, 1, ctas);

        const int activation_groups = routes * (kIntermediate / 128);
        for (int group = cta; group < activation_groups; group += ctas) {
            reduce_swiglu_quant_task<kIntermediate, SplitK>(
                partials, activation, qactivation, activation_scale,
                route_to_sorted, topk_ids, g2, routes, group);
            __syncthreads();
        }
        single_launch_grid_barrier(barrier_state, 2, ctas);

        const int w2_tasks = max_mblocks * kW2NTiles;
        for (int task = cta; task < w2_tasks; task += ctas) {
            route_gemm_task<512, 4096, 1, false>(
                &w2_tma_weight, &w2_tma_weight_scale,
                w2, s2, g2, qactivation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, topk_weights,
                reinterpret_cast<float*>(down), lut, nullptr,
                routes, 0, task);
            __syncthreads();
        }
        single_launch_grid_barrier(barrier_state, 3, ctas);
    }

    // M128 exceeds CARv2's one-shot push slab.  Reuse its multicast-bound
    // pull region and semaphore protocol there; smaller messages retain the
    // validated 78-CTA multicast push path.  Extra compute CTAs can retire
    // because kernel completion still waits for every communication CTA.
    if (tokens == 128 && cta < 16) {
        fused_k6_nvls_pull_tp4_task<128>(
            down, topk_weights, pull_input, pull_input_mc, output,
            pull_sem_local, pull_sem_mc, tokens, cta, 16);
    } else if (tokens != 128 && cta < 78) {
        fused_k6_push_ar_tp4_task<128, true>(
            down, topk_weights, output, push_counter,
            push0, push1, push2, push3, push_mc,
            tokens, rank, push_stride, 0, 4096, cta, 78);
    }
}

__device__ __forceinline__ uint4 load_multimem_reduce_bf16_16b(
        const void* pointer) {
    uint4 value;
    asm volatile(
        "multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ uint32_t load_relaxed_sys_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.relaxed.sys.global.u32 %0, [%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t load_acquire_sys_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void multimem_red_add_relaxed_u32(
        uint32_t* pointer) {
    asm volatile(
        "multimem.red.relaxed.sys.global.add.u32 [%0], 1;"
        : : "l"(pointer) : "memory");
}

__device__ __forceinline__ void multimem_red_add_release_u32(
        uint32_t* pointer) {
    asm volatile(
        "multimem.red.release.sys.global.add.u32 [%0], 1;"
        : : "l"(pointer) : "memory");
}

// The per-rank W2 route tensor is allocated in multicast-bound symmetric
// memory.  Use NVLS to reduce each route across TP4 first, then exploit
// linearity to apply the shared route weights and fixed k=6 locally.  This
// replaces both the local k6 materialization and a separate output AR.
template <int Threads>
__global__ __launch_bounds__(Threads) void fused_rank_route_mc_pull_tp4_kernel(
        const uint8_t* __restrict__ route_mc,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ output,
        uint8_t* __restrict__ sem_local,
        uint8_t* __restrict__ sem_mc,
        int tokens) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kVecBytes = 16;
    constexpr int kPairsPerVec = kVecBytes / sizeof(__nv_bfloat162);
    constexpr int kVecsPerToken = kHidden * sizeof(__nv_bfloat16) / kVecBytes;
    constexpr int kSemaphoreBytes = 128;

    uint32_t barrier_current = 0;
    if (threadIdx.x == 0) {
        uint8_t* sem = sem_local + blockIdx.x * kSemaphoreBytes;
        auto* flag = reinterpret_cast<uint32_t*>(sem);
        auto* counter = reinterpret_cast<uint32_t*>(sem + sizeof(uint32_t));
        const uint32_t reserved = atomicAdd(counter, 2 * kWorld);
        barrier_current = reserved + kWorld;
        multimem_red_add_relaxed_u32(reinterpret_cast<uint32_t*>(
            sem_mc + blockIdx.x * kSemaphoreBytes));
        while (load_relaxed_sys_u32(flag) - reserved < kWorld) {
        }
    }
    __syncthreads();

    const int global_tid = blockIdx.x * Threads + threadIdx.x;
    const int global_threads = gridDim.x * Threads;
    const int num_vecs = tokens * kVecsPerToken;
    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const int token = vec / kVecsPerToken;
        const int vec_in_token = vec - token * kVecsPerToken;
        uint4 route_sum[kTopK];
        #pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const int64_t route_vec =
                (static_cast<int64_t>(token) * kTopK + route)
                    * kVecsPerToken
                + vec_in_token;
            route_sum[route] = load_multimem_reduce_bf16_16b(
                route_mc + route_vec * kVecBytes);
        }

        float2 accum[kPairsPerVec];
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair)
            accum[pair] = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const float route_weight =
                __ldg(topk_weights + token * kTopK + route) * kRoutedScale;
            const uint32_t* words =
                reinterpret_cast<const uint32_t*>(&route_sum[route]);
            #pragma unroll
            for (int pair = 0; pair < kPairsPerVec; ++pair) {
                const __nv_bfloat162 value =
                    *reinterpret_cast<const __nv_bfloat162*>(words + pair);
                const float2 value_f32 = __bfloat1622float2(value);
                accum[pair].x =
                    fmaf(value_f32.x, route_weight, accum[pair].x);
                accum[pair].y =
                    fmaf(value_f32.y, route_weight, accum[pair].y);
            }
        }

        uint4 result;
        auto* result_words = reinterpret_cast<uint32_t*>(&result);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(accum[pair].x, accum[pair].y);
            result_words[pair] = *reinterpret_cast<const uint32_t*>(&value);
        }
        reinterpret_cast<uint4*>(output)[vec] = result;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        auto* flag = reinterpret_cast<uint32_t*>(
            sem_local + blockIdx.x * kSemaphoreBytes);
        multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(
            sem_mc + blockIdx.x * kSemaphoreBytes));
        while (load_acquire_sys_u32(flag) - barrier_current < kWorld) {
        }
    }
}

// Block-cooperative one-shot NVLS pull.  Each CTA materializes its disjoint
// local k6 vectors once into the symmetric input, publishes them with a
// release multicast semaphore arrival, then every rank obtains the TP4 sum
// with one multimem reduction load per vector.
template <int Threads>
__device__ __forceinline__ void fused_k6_nvls_pull_tp4_task(
        const __nv_bfloat16* __restrict__ route_input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ symm_input,
        const uint8_t* __restrict__ symm_input_mc,
        __nv_bfloat16* __restrict__ output,
        uint8_t* __restrict__ sem_local,
        uint8_t* __restrict__ sem_mc,
        int tokens, int linear_block_idx, int linear_grid_dim) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kPairsPerToken = kHidden / 2;
    constexpr int kVecBytes = 16;
    constexpr int kPairsPerVec = kVecBytes / sizeof(__nv_bfloat162);
    constexpr int kVecsPerToken = kHidden * sizeof(__nv_bfloat16) / kVecBytes;
    constexpr int kSemaphoreBytes = 128;
    const int global_tid = linear_block_idx * Threads + threadIdx.x;
    const int global_threads = linear_grid_dim * Threads;
    const int num_vecs = tokens * kVecsPerToken;

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const int token = vec / kVecsPerToken;
        const int vec_in_token = vec - token * kVecsPerToken;
        const int pair0 = vec_in_token * kPairsPerVec;
        float2 accum[kPairsPerVec];
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair)
            accum[pair] = make_float2(0.0f, 0.0f);

        const auto* route_input2 =
            reinterpret_cast<const __nv_bfloat162*>(route_input);
        #pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            const float route_weight =
                __ldg(topk_weights + token * kTopK + route) * kRoutedScale;
            const int64_t route_base =
                (static_cast<int64_t>(token) * kTopK + route)
                    * kPairsPerToken;
            #pragma unroll
            for (int pair = 0; pair < kPairsPerVec; ++pair) {
                const float2 value = __bfloat1622float2(
                    route_input2[route_base + pair0 + pair]);
                accum[pair].x =
                    fmaf(value.x, route_weight, accum[pair].x);
                accum[pair].y =
                    fmaf(value.y, route_weight, accum[pair].y);
            }
        }

        uint4 local_value;
        auto* words = reinterpret_cast<uint32_t*>(&local_value);
        #pragma unroll
        for (int pair = 0; pair < kPairsPerVec; ++pair) {
            const __nv_bfloat162 value =
                __floats2bfloat162_rn(accum[pair].x, accum[pair].y);
            words[pair] = *reinterpret_cast<const uint32_t*>(&value);
        }
        reinterpret_cast<uint4*>(symm_input)[vec] = local_value;
    }

    __syncthreads();
    uint32_t barrier_current = 0;
    if (threadIdx.x == 0) {
        uint8_t* sem = sem_local + linear_block_idx * kSemaphoreBytes;
        auto* flag = reinterpret_cast<uint32_t*>(sem);
        auto* counter = reinterpret_cast<uint32_t*>(sem + sizeof(uint32_t));
        const uint32_t reserved = atomicAdd(counter, 2 * kWorld);
        barrier_current = reserved + kWorld;
        // The CTA barrier orders every producer lane's global stores before
        // this release arrival; acquire polling on all ranks publishes them.
        multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(
            sem_mc + linear_block_idx * kSemaphoreBytes));
        while (load_acquire_sys_u32(flag) - reserved < kWorld) {
        }
    }
    __syncthreads();

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const uint4 sum = load_multimem_reduce_bf16_16b(
            symm_input_mc + static_cast<int64_t>(vec) * kVecBytes);
        reinterpret_cast<uint4*>(output)[vec] = sum;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        auto* flag = reinterpret_cast<uint32_t*>(
            sem_local + linear_block_idx * kSemaphoreBytes);
        multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(
            sem_mc + linear_block_idx * kSemaphoreBytes));
        while (load_acquire_sys_u32(flag) - barrier_current < kWorld) {
        }
    }
}

template <int Threads>
__global__ __launch_bounds__(Threads) void fused_k6_nvls_pull_tp4_kernel(
        const __nv_bfloat16* __restrict__ route_input,
        const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ symm_input,
        const uint8_t* __restrict__ symm_input_mc,
        __nv_bfloat16* __restrict__ output,
        uint8_t* __restrict__ sem_local,
        uint8_t* __restrict__ sem_mc,
        int tokens) {
    fused_k6_nvls_pull_tp4_task<Threads>(
        route_input, topk_weights, symm_input, symm_input_mc,
        output, sem_local, sem_mc, tokens,
        static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
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

template <int K, int N, int SplitK, bool IsW13, int LaunchNTiles = 0,
          bool PublishW2Progress = false, bool DualWgW13 = false>
void launch_route_gemm(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int max_routes,
        int n_tile_begin = 0, int32_t* progress_state = nullptr) {
    static CUtensorMap weight_descriptor;
    static CUtensorMap scale_descriptor;
    static void* last_weight_pointer = nullptr;
    static void* last_scale_pointer = nullptr;
    if (last_weight_pointer != weight.data_ptr()
            || last_scale_pointer != weight_scale.data_ptr()) {
        weight_descriptor = make_weight_desc(
            weight.data_ptr(), K, weight.numel());
        if constexpr (K >= 512 || kCompactInterleavedScale) {
            if constexpr (kInterleavedBulkCopy) {
                // Neither tensor map is consumed by the linear interleaved
                // path, but keep a valid by-value descriptor argument.
                scale_descriptor = weight_descriptor;
            } else {
                scale_descriptor = make_weight_scale_desc(
                    weight_scale.data_ptr(), K, weight_scale.numel());
            }
        }
        last_weight_pointer = weight.data_ptr();
        last_scale_pointer = weight_scale.data_ptr();
    }
    const int max_m_blocks = expert_ids.numel();
    constexpr int launch_n_tiles =
        LaunchNTiles == 0 ? N / kWout : LaunchNTiles;
    constexpr int math_wgs = DualWgW13 ? 2 : 1;
    static_assert(launch_n_tiles % math_wgs == 0);
    TORCH_CHECK(n_tile_begin >= 0
                    && n_tile_begin + launch_n_tiles <= N / kWout,
                "invalid output-N tile range");
    const int grid = max_m_blocks * (launch_n_tiles / math_wgs) * SplitK;
    constexpr bool use_tma_scale = K >= 512 || kCompactInterleavedScale;
    constexpr int effective_scale_buffers =
        use_tma_scale ? kScaleBuffers : kStages;
    constexpr bool interleaved_scale =
        kInterleavedBulkCopy && use_tma_scale;
    constexpr int scale_row_bytes =
        kCompactInterleavedScale ? 4 : 16;
    constexpr int dynamic_smem_bytes =
        math_wgs * (interleaved_scale
            ? kStages * kWout * ((kBlockK / 2) + scale_row_bytes)
            : kStages * kWout * (kBlockK / 2)
                + effective_scale_buffers * kWout * scale_row_bytes)
        + kTok * kBlockK
        + ((!IsW13 && kW2RouteOutput && kW2CoalescedStore)
            ? kTok * kWout * static_cast<int>(sizeof(__nv_bfloat16))
            : 0);
    if constexpr (IsW13 && kW13MaxSmemCarveout) {
        const cudaError_t carveout_result = cudaFuncSetAttribute(
            route_gemm<K, N, SplitK, IsW13, LaunchNTiles,
                       PublishW2Progress, DualWgW13>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        TORCH_CHECK(
            carveout_result == cudaSuccess,
            "failed to set maximum W13 shared-memory carveout: ",
            cudaGetErrorString(carveout_result));
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    route_gemm<K, N, SplitK, IsW13, LaunchNTiles,
               PublishW2Progress, DualWgW13><<<
        grid, math_wgs * 128, dynamic_smem_bytes, stream>>>(
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
        progress_state,
        max_routes,
        n_tile_begin);
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
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 1024, 4, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 1024, 4, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else if (split_k == 2) {
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 1024, 2, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 1024, 2, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 1024, 1, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 1024, 1, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4) {
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 512, 4, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 512, 4, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else if (split_k == 2) {
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 512, 2, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 512, 2, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        } else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            if constexpr (kW13DualWgSplit)
                launch_route_gemm<4096, 512, 1, true, 0, false, true>(
                    weight, weight_scale, weight_global_scale,
                    activation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded, partials,
                    partials, lut, routes);
            else launch_route_gemm<4096, 512, 1, true>(
                weight, weight_scale, weight_global_scale,
                activation, activation_scale,
                sorted_ids, expert_ids, num_tokens_padded, partials,
                partials, lut, routes);
        }
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
}

void run_w13_paired_impl(
        torch::Tensor weight, torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor raw_output,
        torch::Tensor bf16_output, torch::Tensor quantized_output,
        torch::Tensor output_scale, int intermediate) {
    TORCH_CHECK(kWout == 128,
                "paired W13 requires V4_WOUT=128");
    TORCH_CHECK(kNormalizedWeightScale && kBulkWeightCopy
                    && kInterleavedBulkCopy && kMode2Braid,
                "paired W13 requires normalized Mode2 interleaved bulk weights");
    TORCH_CHECK(intermediate == 512 || intermediate == 256,
                "intermediate must be 512 (TP4) or 256 (TP8)");
    TORCH_CHECK(weight.scalar_type() == torch::kUInt8
                    && activation.scalar_type() == torch::kUInt8,
                "paired W13 weight/activation must be uint8");
    TORCH_CHECK(weight_global_scale.scalar_type() == torch::kFloat32
                    && activation_scale.scalar_type() == torch::kFloat32
                    && output_scale.scalar_type() == torch::kFloat32,
                "paired W13 scales must be float32");
    TORCH_CHECK(quantized_output.scalar_type() == torch::kUInt8,
                "paired W13 quantized output must be uint8");
    TORCH_CHECK(raw_output.numel() == 0
                    || raw_output.scalar_type() == torch::kFloat32,
                "paired W13 debug output must be empty or float32");
    TORCH_CHECK(bf16_output.numel() == 0
                    || bf16_output.scalar_type() == torch::kBFloat16,
                "paired W13 activation output must be empty or bfloat16");
    TORCH_CHECK(activation.dim() == 2 && activation.size(1) == 4096,
                "paired W13 activation must have K=4096");
    TORCH_CHECK(quantized_output.numel()
                    == activation.size(0) * 6 * intermediate,
                "paired W13 quantized output shape mismatch");
    TORCH_CHECK(output_scale.numel()
                    == activation.size(0) * 6 * (intermediate / 128),
                "paired W13 output scale shape mismatch");
    const int max_routes = activation.size(0) * 6;
    const int output_groups = intermediate / 128;
    const int grid = expert_ids.numel() * output_groups;
    constexpr int dynamic_smem_bytes =
        2 * 2 * (kWout * (kBlockK / 2) + kWout * 16)
        + 2 * 2 * kTok * kBlockK;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (intermediate == 512) {
        cudaFuncSetAttribute(paired_w13_fused_kernel<512>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
        paired_w13_fused_kernel<512><<<
            grid, 256, dynamic_smem_bytes, stream>>>(
            weight.data_ptr<uint8_t>(),
            weight_global_scale.data_ptr<float>(),
            activation.data_ptr<uint8_t>(),
            activation_scale.data_ptr<float>(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
            num_tokens_padded.data_ptr<int32_t>(),
            raw_output.numel() ? raw_output.data_ptr<float>() : nullptr,
            bf16_output.numel()
                ? reinterpret_cast<__nv_bfloat16*>(bf16_output.data_ptr())
                : nullptr,
            quantized_output.data_ptr<uint8_t>(),
            output_scale.data_ptr<float>(), max_routes);
    } else {
        cudaFuncSetAttribute(paired_w13_fused_kernel<256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
        paired_w13_fused_kernel<256><<<
            grid, 256, dynamic_smem_bytes, stream>>>(
            weight.data_ptr<uint8_t>(),
            weight_global_scale.data_ptr<float>(),
            activation.data_ptr<uint8_t>(),
            activation_scale.data_ptr<float>(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
            num_tokens_padded.data_ptr<int32_t>(),
            raw_output.numel() ? raw_output.data_ptr<float>() : nullptr,
            bf16_output.numel()
                ? reinterpret_cast<__nv_bfloat16*>(bf16_output.data_ptr())
                : nullptr,
            quantized_output.data_ptr<uint8_t>(),
            output_scale.data_ptr<float>(), max_routes);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

void run_w2_progress(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut,
        torch::Tensor progress_state, int intermediate) {
    TORCH_CHECK(intermediate == 512,
                "W2 progress prototype currently supports TP4 Is=512");
    TORCH_CHECK(kWout == 128 && kW2RouteOutput,
                "W2 progress requires WOUT128 route output");
    TORCH_CHECK(progress_state.scalar_type() == torch::kInt32
                    && progress_state.is_cuda()
                    && progress_state.is_contiguous(),
                "W2 progress state must be contiguous CUDA int32");
    const int tokens = topk_weights.numel() / kTopK;
    TORCH_CHECK(progress_state.numel() >= tokens * 192 + 2,
                "W2 progress state is too small");
    launch_route_gemm<512, 4096, 1, false, 0, true>(
        weight, weight_scale, weight_global_scale,
        activation, activation_scale,
        sorted_ids, expert_ids, num_tokens_padded, topk_weights,
        output, lut, topk_weights.numel(), 0,
        progress_state.data_ptr<int32_t>());
}

void run_w2_chunk(
        torch::Tensor weight, torch::Tensor weight_scale,
        torch::Tensor weight_global_scale,
        torch::Tensor activation, torch::Tensor activation_scale,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor lut, int intermediate,
        int chunks, int chunk_idx) {
    TORCH_CHECK(chunks == 2 || chunks == 4 || chunks == 8,
                "W2 pipeline chunks must be 2,4,8");
    TORCH_CHECK(chunk_idx >= 0 && chunk_idx < chunks,
                "W2 pipeline chunk index is out of range");
    const int ntiles = 4096 / kWout;
    const int tiles_per_chunk = ntiles / chunks;
    const int n_tile_begin = chunk_idx * tiles_per_chunk;

#define LAUNCH_W2_CHUNK(K_VALUE, TILES_VALUE)                              \
    launch_route_gemm<K_VALUE, 4096, 1, false, TILES_VALUE>(               \
        weight, weight_scale, weight_global_scale,                         \
        activation, activation_scale, sorted_ids, expert_ids,             \
        num_tokens_padded, topk_weights, output, lut,                      \
        topk_weights.numel(), n_tile_begin)

    if (intermediate == 512) {
        if (chunks == 2)
            LAUNCH_W2_CHUNK(512, 16);
        else if (chunks == 4)
            LAUNCH_W2_CHUNK(512, 8);
        else
            LAUNCH_W2_CHUNK(512, 4);
    } else if (intermediate == 256) {
        if (chunks == 2)
            LAUNCH_W2_CHUNK(256, 16);
        else if (chunks == 4)
            LAUNCH_W2_CHUNK(256, 8);
        else
            LAUNCH_W2_CHUNK(256, 4);
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
#undef LAUNCH_W2_CHUNK
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
        else if (split_k == 2)
            launch_reduce_swiglu<512, 2>(partials, output, routes);
        else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            launch_reduce_swiglu<512, 1>(partials, output, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4)
            launch_reduce_swiglu<256, 4>(partials, output, routes);
        else if (split_k == 2)
            launch_reduce_swiglu<256, 2>(partials, output, routes);
        else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            launch_reduce_swiglu<256, 1>(partials, output, routes);
        }
    } else {
        TORCH_CHECK(false, "intermediate must be 512 (TP4) or 256 (TP8)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Intermediate, int SplitK>
void launch_reduce_swiglu_quant(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale,
        torch::Tensor route_to_sorted, torch::Tensor topk_ids,
        torch::Tensor w2_global_scale, int routes) {
    constexpr int threads = 128;
    constexpr int groups_per_route = Intermediate / 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* activation_ptr = activation.numel()
        ? reinterpret_cast<__nv_bfloat16*>(activation.data_ptr())
        : nullptr;
    reduce_swiglu_quant_kernel<Intermediate, SplitK><<<
        routes * groups_per_route, threads, 0, stream>>>(
        partials.data_ptr<float>(), activation_ptr,
        quantized.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        route_to_sorted.data_ptr<int32_t>(),
        topk_ids.data_ptr<int32_t>(), w2_global_scale.data_ptr<float>(),
        routes);
}

void reduce_swiglu_quant(
        torch::Tensor partials, torch::Tensor activation,
        torch::Tensor quantized, torch::Tensor scale,
        int intermediate, int split_k, torch::Tensor route_to_sorted,
        torch::Tensor topk_ids, torch::Tensor w2_global_scale) {
    const int routes = partials.size(1);
    TORCH_CHECK(!(kW2SortedAct || kW2MblockScale)
                    || route_to_sorted.numel() == routes,
                "sorted W2 activation/scale requires one position per route");
    TORCH_CHECK(!kW2FoldGlobalScale
                    || (topk_ids.numel() == routes
                        && w2_global_scale.numel() == 256),
                "folded W2 scale requires route experts and 256 globals");
    if (intermediate == 512) {
        if (split_k == 4)
            launch_reduce_swiglu_quant<512, 4>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
        else if (split_k == 2)
            launch_reduce_swiglu_quant<512, 2>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
        else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            launch_reduce_swiglu_quant<512, 1>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
        }
    } else if (intermediate == 256) {
        if (split_k == 4)
            launch_reduce_swiglu_quant<256, 4>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
        else if (split_k == 2)
            launch_reduce_swiglu_quant<256, 2>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
        else {
            TORCH_CHECK(split_k == 1, "split_k must be 1, 2, or 4");
            launch_reduce_swiglu_quant<256, 1>(
                partials, activation, quantized, scale, route_to_sorted,
                topk_ids, w2_global_scale, routes);
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

void tiled_k6_reduce(
        torch::Tensor input, torch::Tensor topk_weights,
        torch::Tensor output, int mode) {
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "route input must be bfloat16");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                "topk_weights must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
                "reduction output must be bfloat16");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "route input must be contiguous CUDA");
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.is_contiguous(),
                "topk_weights must be contiguous CUDA");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "reduction output must be contiguous CUDA");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096,
                "route input must have shape [M*6,4096]");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == 4096,
                "reduction output must have shape [M,4096]");
    TORCH_CHECK(input.size(0) == output.size(0) * 6,
                "route input row count must equal M*6");
    TORCH_CHECK(topk_weights.numel() == output.size(0) * 6,
                "topk_weights must have shape [M,6]");
    const int tokens = output.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (mode == 1) {
        tiled_k6_reduce_kernel<128, 1><<<dim3(16, tokens), 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), tokens);
    } else if (mode == 2) {
        tiled_k6_reduce_kernel<128, 2><<<dim3(8, tokens), 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), tokens);
    } else if (mode == 3) {
        tiled_k6_reduce_kernel<256, 1><<<dim3(8, tokens), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), tokens);
    } else if (mode == 4) {
        tiled_k6_reduce_kernel<256, 2><<<dim3(4, tokens), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), tokens);
    } else {
        TORCH_CHECK(false, "tiled k6 reduce mode must be 1,2,3,4");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitK>
void launch_tp4_megamoe_single(
        const CUtensorMap& w13_descriptor,
        const CUtensorMap& w2_descriptor,
        torch::Tensor w13, torch::Tensor s13, torch::Tensor g13,
        torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
        torch::Tensor qx, torch::Tensor x_scale, torch::Tensor topk_ids,
        torch::Tensor topk_weights, torch::Tensor sorted_ids,
        torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
        torch::Tensor partials,
        torch::Tensor activation, torch::Tensor qactivation,
        torch::Tensor activation_scale, torch::Tensor down,
        torch::Tensor lut, torch::Tensor barrier_state,
        torch::Tensor route_to_sorted, torch::Tensor output,
        torch::Tensor push_counter, torch::Tensor push0,
        torch::Tensor push1, torch::Tensor push2, torch::Tensor push3,
        torch::Tensor pull_input, torch::Tensor pull_sem_local,
        int rank, int64_t push_stride, int64_t push_mc_ptr,
        int64_t pull_input_mc_ptr, int64_t pull_sem_mc_ptr,
        int requested_ctas_per_sm) {
    constexpr int dynamic_smem_bytes =
        kStages * kWout * ((kBlockK / 2) + 4) + kTok * kBlockK;
    const cudaError_t attr_result = cudaFuncSetAttribute(
        tp4_megamoe_single_launch_kernel<SplitK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes);
    TORCH_CHECK(attr_result == cudaSuccess,
                "failed to set single-launch dynamic shared memory: ",
                cudaGetErrorString(attr_result));

    int active_per_sm = 0;
    const cudaError_t occupancy_result =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_per_sm, tp4_megamoe_single_launch_kernel<SplitK>,
            128, dynamic_smem_bytes);
    TORCH_CHECK(occupancy_result == cudaSuccess && active_per_sm > 0,
                "single-launch occupancy query failed: ",
                cudaGetErrorString(occupancy_result));
    int device = -1;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    TORCH_CHECK(properties.multiProcessorCount == 78,
                "single-launch TP4 push protocol currently requires 78-SM H20");
    TORCH_CHECK(requested_ctas_per_sm > 0,
                "requested single-launch CTAs/SM must be positive");
    const int selected_ctas_per_sm = std::min(
        requested_ctas_per_sm, active_per_sm);
    const int resident_grid =
        properties.multiProcessorCount * selected_ctas_per_sm;
    TORCH_CHECK(resident_grid >= 78,
                "single-launch kernel cannot keep the 78 communication CTAs resident");

    const auto stream = at::cuda::getCurrentCUDAStream();
    tp4_megamoe_single_launch_kernel<SplitK><<<
        resident_grid, 128, dynamic_smem_bytes, stream>>>(
        w13_descriptor, w13_descriptor, w2_descriptor, w2_descriptor,
        w13.data_ptr<uint8_t>(), s13.data_ptr<uint8_t>(),
        g13.data_ptr<float>(),
        w2.data_ptr<uint8_t>(), s2.data_ptr<uint8_t>(), g2.data_ptr<float>(),
        qx.data_ptr<uint8_t>(), x_scale.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_padded.data_ptr<int32_t>(), partials.data_ptr<float>(),
        activation.numel()
            ? reinterpret_cast<__nv_bfloat16*>(activation.data_ptr())
            : nullptr,
        qactivation.data_ptr<uint8_t>(), activation_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(down.data_ptr()),
        reinterpret_cast<const uint2*>(lut.data_ptr<uint8_t>()),
        barrier_state.data_ptr<int32_t>(),
        route_to_sorted.numel()
            ? route_to_sorted.data_ptr<int32_t>()
            : nullptr,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(push_counter.data_ptr()),
        push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
        push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(push_mc_ptr),
        reinterpret_cast<__nv_bfloat16*>(pull_input.data_ptr()),
        reinterpret_cast<const uint8_t*>(pull_input_mc_ptr),
        pull_sem_local.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(pull_sem_mc_ptr),
        output.size(0), expert_ids.numel(), rank, push_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_tp4_megamoe_single_launch(
        torch::Tensor w13, torch::Tensor s13, torch::Tensor g13,
        torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
        torch::Tensor qx, torch::Tensor x_scale, torch::Tensor topk_ids,
        torch::Tensor topk_weights, torch::Tensor sorted_ids,
        torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
        torch::Tensor partials,
        torch::Tensor activation, torch::Tensor qactivation,
        torch::Tensor activation_scale, torch::Tensor down,
        torch::Tensor lut, torch::Tensor barrier_state,
        torch::Tensor route_to_sorted, torch::Tensor output,
        torch::Tensor push_counter, torch::Tensor push0,
        torch::Tensor push1, torch::Tensor push2, torch::Tensor push3,
        torch::Tensor pull_input, torch::Tensor pull_sem_local,
        int rank, int64_t push_stride, int64_t push_mc_ptr,
        int64_t pull_input_mc_ptr, int64_t pull_sem_mc_ptr,
        int split_k, int requested_ctas_per_sm) {
    TORCH_CHECK(kWout == 128 && kTiledWeightLayout && kBulkWeightCopy
                    && kInterleavedBulkCopy && kCompactInterleavedScale
                    && kMode2Braid && kNormalizedWeightScale
                    && kW2RouteOutput && !kW2SortedAct && !kW2MblockScale
                    && !kW2FoldGlobalScale && !kW2CoalescedStore,
                "single-launch TP4 currently requires the selected default layout");
    TORCH_CHECK(split_k == 2 || split_k == 4,
                "single-launch TP4 split-K must be 2 or 4");
    TORCH_CHECK(qx.scalar_type() == torch::kUInt8
                    && qx.is_cuda() && qx.is_contiguous()
                    && qx.dim() == 2 && qx.size(1) == 4096,
                "single-launch X must be contiguous FP8 storage [M,4096]");
    const int tokens = qx.size(0);
    TORCH_CHECK(tokens == 8 || tokens == 16 || tokens == 32
                    || tokens == 64 || tokens == 128,
                "single-launch TP4 supports M=8,16,32,64,128");
    const int routes = tokens * kTopK;
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32
                    && topk_ids.numel() == routes,
                "single-launch topk_ids must be int32 [M,6]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32
                    && topk_weights.numel() == routes,
                "single-launch topk_weights must be float32 [M,6]");
    TORCH_CHECK(w13.scalar_type() == torch::kUInt8
                    && s13.scalar_type() == torch::kUInt8
                    && w2.scalar_type() == torch::kUInt8
                    && s2.scalar_type() == torch::kUInt8,
                "single-launch weights/scales must use packed uint8 storage");
    TORCH_CHECK(g13.scalar_type() == torch::kFloat32 && g13.numel() == 256
                    && g2.scalar_type() == torch::kFloat32 && g2.numel() == 256,
                "single-launch normalized expert scales must have 256 entries");
    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32
                    && x_scale.is_cuda() && x_scale.is_contiguous()
                    && x_scale.numel() == tokens * 32,
                "single-launch X scale must be contiguous FP32 [M,32]");
    TORCH_CHECK(partials.scalar_type() == torch::kFloat32
                    && partials.numel() >= 4LL * routes * 1024,
                "single-launch W13 partial workspace is too small");
    TORCH_CHECK(qactivation.scalar_type() == torch::kUInt8
                    && qactivation.numel() == routes * 512
                    && activation_scale.scalar_type() == torch::kFloat32
                    && activation_scale.numel() == routes * 4,
                "single-launch activation quantization workspace mismatch");
    TORCH_CHECK(down.scalar_type() == torch::kBFloat16
                    && down.numel() == static_cast<int64_t>(routes) * 4096,
                "single-launch W2 route workspace shape mismatch");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.numel() == static_cast<int64_t>(tokens) * 4096,
                "single-launch output shape mismatch");
    TORCH_CHECK(sorted_ids.scalar_type() == torch::kInt32
                    && expert_ids.scalar_type() == torch::kInt32
                    && num_tokens_padded.scalar_type() == torch::kInt32
                    && num_tokens_padded.numel() == 1,
                "single-launch route workspaces must be int32");
    const int64_t scheduler_words = kSingleLaunchInterleaved
        ? kSchedulerHeaderWords + 3LL * expert_ids.numel()
        : 0;
    TORCH_CHECK(barrier_state.scalar_type() == torch::kInt32
                    && barrier_state.is_cuda()
                    && barrier_state.numel() >= 8 + scheduler_words,
                "single-launch barrier/scheduler state is too small");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4
                    && push_counter.numel() == 78,
                "single-launch TP4 requires the 78-entry CARv2 push counter");
    TORCH_CHECK(rank >= 0 && rank < 4 && push_mc_ptr != 0,
                "single-launch TP4 requires rank and multicast symmetric VA");
    TORCH_CHECK(tokens == 128
                    || push_stride >= output.numel() * output.element_size(),
                "single-launch push workspace stride is too small");
    for (const auto& workspace : {push0, push1, push2, push3}) {
        TORCH_CHECK(workspace.scalar_type() == torch::kUInt8
                        && workspace.is_cuda() && workspace.is_contiguous(),
                    "single-launch push workspaces must be contiguous CUDA uint8");
    }
    TORCH_CHECK(pull_input.scalar_type() == torch::kBFloat16
                    && pull_input.is_cuda() && pull_input.is_contiguous()
                    && pull_input.sizes() == output.sizes(),
                "single-launch pull input must match output [M,4096]");
    TORCH_CHECK(pull_sem_local.scalar_type() == torch::kUInt8
                    && pull_sem_local.is_cuda()
                    && pull_sem_local.numel() >= 16 * 128,
                "single-launch pull semaphore slab is too small");
    TORCH_CHECK(pull_input_mc_ptr != 0 && pull_sem_mc_ptr != 0,
                "single-launch TP4 requires multicast pull/semaphore VAs");

    static CUtensorMap w13_descriptor;
    static CUtensorMap w2_descriptor;
    static void* last_w13 = nullptr;
    static void* last_w2 = nullptr;
    if (last_w13 != w13.data_ptr()) {
        w13_descriptor = make_weight_desc(w13.data_ptr(), 4096, w13.numel());
        last_w13 = w13.data_ptr();
    }
    if (last_w2 != w2.data_ptr()) {
        w2_descriptor = make_weight_desc(w2.data_ptr(), 512, w2.numel());
        last_w2 = w2.data_ptr();
    }

    if (split_k == 4) {
        launch_tp4_megamoe_single<4>(
            w13_descriptor, w2_descriptor,
            w13, s13, g13, w2, s2, g2, qx, x_scale, topk_ids, topk_weights,
            sorted_ids, expert_ids, num_tokens_padded, partials,
            activation, qactivation, activation_scale, down, lut,
            barrier_state, route_to_sorted, output, push_counter,
            push0, push1, push2, push3, pull_input, pull_sem_local,
            rank, push_stride, push_mc_ptr,
            pull_input_mc_ptr, pull_sem_mc_ptr, requested_ctas_per_sm);
    } else {
        launch_tp4_megamoe_single<2>(
            w13_descriptor, w2_descriptor,
            w13, s13, g13, w2, s2, g2, qx, x_scale, topk_ids, topk_weights,
            sorted_ids, expert_ids, num_tokens_padded, partials,
            activation, qactivation, activation_scale, down, lut,
            barrier_state, route_to_sorted, output, push_counter,
            push0, push1, push2, push3, pull_input, pull_sem_local,
            rank, push_stride, push_mc_ptr,
            pull_input_mc_ptr, pull_sem_mc_ptr, requested_ctas_per_sm);
    }
}

void fused_k6_push_ar_tp4(
        torch::Tensor input, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor push_counter,
        torch::Tensor push0, torch::Tensor push1,
        torch::Tensor push2, torch::Tensor push3,
        int rank, int64_t push_stride, int64_t push_mc_ptr) {
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "route input must be bfloat16");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                "topk_weights must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
                "all-reduce output must be bfloat16");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "route input must be contiguous CUDA");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "all-reduce output must be contiguous CUDA");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096,
                "route input must have shape [M*6,4096]");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == 4096,
                "all-reduce output must have shape [M,4096]");
    TORCH_CHECK(input.size(0) == output.size(0) * 6,
                "route input row count must equal M*6");
    TORCH_CHECK(topk_weights.numel() == output.size(0) * 6,
                "topk_weights must have shape [M,6]");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4,
                "push counter must be a CUDA uint32 tensor");
    TORCH_CHECK(push_counter.numel() == 78,
                "TP4 H20 push counter must contain 78 CTA counters");
    TORCH_CHECK(rank >= 0 && rank < 4, "TP4 rank must be in [0,4)");
    TORCH_CHECK(push_stride >= output.numel() * output.element_size(),
                "push workspace stride is too small");
    for (const auto& workspace : {push0, push1, push2, push3}) {
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(),
                    "push workspaces must be contiguous CUDA tensors");
        TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
                    "push workspaces must be uint8");
    }
    const int tokens = output.size(0);
    const bool use_multicast = push_mc_ptr != 0;
    TORCH_CHECK(
        tokens == 8 || tokens == 16 || tokens == 32
            || (use_multicast && tokens == 64),
        "fused push supports M=8,16,32 and multicast M=64");
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* counter = reinterpret_cast<uint32_t*>(push_counter.data_ptr());
    auto* push_mc = reinterpret_cast<uint8_t*>(push_mc_ptr);
    if (use_multicast && tokens == 64) {
        fused_k6_push_ar_tp4_kernel<512, true><<<78, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,
            push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
            push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
            push_mc, tokens, rank, push_stride, 0, 4096);
    } else if (use_multicast && tokens <= 16) {
        fused_k6_push_ar_tp4_kernel<128, true><<<78, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,
            push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
            push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
            push_mc, tokens, rank, push_stride, 0, 4096);
    } else if (use_multicast) {
        fused_k6_push_ar_tp4_kernel<256, true><<<78, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,
            push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
            push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
            push_mc, tokens, rank, push_stride, 0, 4096);
    } else if (tokens <= 16) {
        fused_k6_push_ar_tp4_kernel<128, false><<<78, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,
            push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
            push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
            nullptr, tokens, rank, push_stride, 0, 4096);
    } else {
        fused_k6_push_ar_tp4_kernel<256, false><<<78, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,
            push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
            push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
            nullptr, tokens, rank, push_stride, 0, 4096);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_k6_push_ar_tp4_chunk(
        torch::Tensor input, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor push_counter,
        torch::Tensor push0, torch::Tensor push1,
        torch::Tensor push2, torch::Tensor push3,
        int rank, int64_t push_stride, int64_t push_mc_ptr,
        int chunks, int chunk_idx, int active_blocks) {
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "route input must be bfloat16");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                "topk_weights must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
                "all-reduce output must be bfloat16");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "route input must be contiguous CUDA");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "all-reduce output must be contiguous CUDA");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096,
                "route input must have shape [M*6,4096]");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == 4096,
                "all-reduce output must have shape [M,4096]");
    TORCH_CHECK(input.size(0) == output.size(0) * 6,
                "route input row count must equal M*6");
    TORCH_CHECK(topk_weights.numel() == output.size(0) * 6,
                "topk_weights must have shape [M,6]");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4
                    && push_counter.numel() == 78,
                "TP4 H20 push counter must contain 78 CUDA uint32 values");
    TORCH_CHECK(rank >= 0 && rank < 4, "TP4 rank must be in [0,4)");
    TORCH_CHECK(push_mc_ptr != 0,
                "pipelined W2 all-reduce requires multicast memory");
    TORCH_CHECK(chunks == 2 || chunks == 4 || chunks == 8,
                "pipeline chunks must be 2,4,8");
    TORCH_CHECK(chunk_idx >= 0 && chunk_idx < chunks,
                "pipeline chunk index is out of range");
    TORCH_CHECK(active_blocks > 0 && active_blocks <= 78,
                "pipeline all-reduce block count must be in [1,78]");
    for (const auto& workspace : {push0, push1, push2, push3}) {
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous()
                        && workspace.scalar_type() == torch::kUInt8,
                    "push workspaces must be contiguous CUDA uint8 tensors");
    }

    const int tokens = output.size(0);
    const int hidden_size = 4096 / chunks;
    const int hidden_offset = chunk_idx * hidden_size;
    TORCH_CHECK(push_stride >=
                    static_cast<int64_t>(tokens) * hidden_size
                        * output.element_size(),
                "push workspace stride is too small for pipeline chunk");
    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_k6_push_ar_tp4_kernel<256, true, true><<<
        active_blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(push_counter.data_ptr()),
        push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
        push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(push_mc_ptr), tokens, rank, push_stride,
        hidden_offset, hidden_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void progress_k6_mc_push_tp4(
        torch::Tensor input, torch::Tensor topk_weights, torch::Tensor output,
        torch::Tensor progress_state, torch::Tensor push_counter,
        torch::Tensor push0, torch::Tensor push1,
        torch::Tensor push2, torch::Tensor push3,
        int64_t push_mc_ptr, int rank, int64_t push_stride,
        int chunks, int active_blocks, bool inline_finish) {
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16
                    && input.is_cuda() && input.is_contiguous(),
                "progress route input must be contiguous CUDA bfloat16");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 4096,
                "progress route input must have shape [M*6,4096]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32
                    && topk_weights.is_cuda()
                    && topk_weights.is_contiguous(),
                "progress topk weights must be contiguous CUDA float32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.is_cuda() && output.is_contiguous()
                    && output.dim() == 2 && output.size(1) == 4096,
                "progress output must be contiguous CUDA [M,4096] bfloat16");
    const int tokens = topk_weights.numel() / kTopK;
    TORCH_CHECK(input.size(0) == tokens * kTopK,
                "progress route rows must equal M*6");
    TORCH_CHECK(progress_state.scalar_type() == torch::kInt32
                    && progress_state.is_cuda()
                    && progress_state.is_contiguous()
                    && progress_state.numel() >= tokens * 192 + 2,
                "progress state must contain at least M*192+2 int32 values");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4
                    && push_counter.numel() == 78,
                "TP4 H20 push counter must contain 78 CUDA uint32 values");
    TORCH_CHECK(push_mc_ptr != 0 && rank >= 0 && rank < 4,
                "progress multicast push requires TP4 multicast memory");
    TORCH_CHECK(push_stride >= input.numel() / kTopK
                    * input.element_size(),
                "progress push workspace stride is too small");
    TORCH_CHECK(chunks == 2 || chunks == 4 || chunks == 8,
                "progress chunks must be 2,4,8");
    TORCH_CHECK(active_blocks > 0 && active_blocks <= 64,
                "progress worker count must be in [1,64]");
    for (const auto& workspace : {push0, push1, push2, push3}) {
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous()
                        && workspace.scalar_type() == torch::kUInt8,
                    "push workspaces must be contiguous CUDA uint8 tensors");
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_PROGRESS_WORKER(THREADS, CHUNKS, INLINE)                    \
    progress_k6_mc_push_tp4_kernel<THREADS, CHUNKS, INLINE><<<            \
        active_blocks, THREADS, 0, stream>>>(                             \
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),         \
        topk_weights.data_ptr<float>(),                                   \
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),              \
        progress_state.data_ptr<int32_t>(),                               \
        reinterpret_cast<uint32_t*>(push_counter.data_ptr()),             \
        push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),              \
        push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),              \
        reinterpret_cast<uint8_t*>(push_mc_ptr),                          \
        tokens, rank, push_stride)
    if (inline_finish) {
        if (chunks == 2)
            LAUNCH_PROGRESS_WORKER(256, 2, true);
        else if (chunks == 4)
            LAUNCH_PROGRESS_WORKER(128, 4, true);
        else
            LAUNCH_PROGRESS_WORKER(64, 8, true);
    } else {
        if (chunks == 2)
            LAUNCH_PROGRESS_WORKER(256, 2, false);
        else if (chunks == 4)
            LAUNCH_PROGRESS_WORKER(128, 4, false);
        else
            LAUNCH_PROGRESS_WORKER(64, 8, false);
    }
#undef LAUNCH_PROGRESS_WORKER
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void progress_mc_push_finish_tp4(
        torch::Tensor output, torch::Tensor push_counter,
        torch::Tensor push0, torch::Tensor push1,
        torch::Tensor push2, torch::Tensor push3,
        int rank, int64_t push_stride) {
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.is_cuda() && output.is_contiguous(),
                "progress all-reduce output must be contiguous CUDA bfloat16");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == 4096,
                "progress all-reduce output must have shape [M,4096]");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4
                    && push_counter.numel() == 78,
                "TP4 H20 push counter must contain 78 CUDA uint32 values");
    TORCH_CHECK(rank >= 0 && rank < 4,
                "progress all-reduce rank must be in [0,4)");
    TORCH_CHECK(push_stride >= output.numel() * output.element_size(),
                "progress push workspace stride is too small");
    for (const auto& workspace : {push0, push1, push2, push3}) {
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous()
                        && workspace.scalar_type() == torch::kUInt8,
                    "push workspaces must be contiguous CUDA uint8 tensors");
    }
    const int tokens = output.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto* counter = reinterpret_cast<uint32_t*>(push_counter.data_ptr());
#define LAUNCH_PROGRESS_FINISH(THREADS)                                    \
    progress_mc_push_finish_tp4_kernel<THREADS><<<78, THREADS, 0, stream>>>(\
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), counter,      \
        push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),              \
        push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),              \
        tokens, rank, push_stride)
    if (tokens <= 16)
        LAUNCH_PROGRESS_FINISH(128);
    else if (tokens <= 32)
        LAUNCH_PROGRESS_FINISH(256);
    else
        LAUNCH_PROGRESS_FINISH(512);
#undef LAUNCH_PROGRESS_FINISH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_rank_route_mc_pull_tp4(
        torch::Tensor route_input, torch::Tensor topk_weights,
        torch::Tensor output, torch::Tensor sem_local,
        int64_t route_mc_ptr, int64_t sem_mc_ptr, int active_blocks) {
    TORCH_CHECK(route_input.scalar_type() == torch::kBFloat16
                    && route_input.is_cuda() && route_input.is_contiguous(),
                "symmetric route input must be contiguous CUDA bfloat16");
    TORCH_CHECK(route_input.dim() == 2 && route_input.size(1) == 4096,
                "symmetric route input must have shape [M*6,4096]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32
                    && topk_weights.is_cuda()
                    && topk_weights.is_contiguous(),
                "topk_weights must be contiguous CUDA float32");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.is_cuda() && output.is_contiguous(),
                "output must be contiguous CUDA bfloat16");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == 4096,
                "output must have shape [M,4096]");
    TORCH_CHECK(route_input.size(0) == output.size(0) * 6
                    && topk_weights.numel() == output.size(0) * 6,
                "route/topk shapes do not match output M");
    TORCH_CHECK(sem_local.scalar_type() == torch::kUInt8
                    && sem_local.is_cuda() && sem_local.is_contiguous(),
                "local semaphore slab must be contiguous CUDA uint8");
    TORCH_CHECK(route_mc_ptr != 0 && sem_mc_ptr != 0,
                "rank-route pull requires multicast route and semaphore VAs");
    TORCH_CHECK(active_blocks > 0 && active_blocks <= 64
                    && sem_local.numel() >= active_blocks * 128,
                "rank-route pull block count exceeds semaphore capacity");
    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_rank_route_mc_pull_tp4_kernel<256><<<
        active_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(route_mc_ptr),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        sem_local.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(sem_mc_ptr),
        output.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_k6_nvls_pull_tp4(
        torch::Tensor route_input, torch::Tensor topk_weights,
        torch::Tensor symm_input, torch::Tensor output,
        torch::Tensor sem_local, int64_t symm_input_mc_ptr,
        int64_t sem_mc_ptr, int active_blocks) {
    TORCH_CHECK(route_input.scalar_type() == torch::kBFloat16
                    && route_input.is_cuda() && route_input.is_contiguous(),
                "route input must be contiguous CUDA bfloat16");
    TORCH_CHECK(route_input.dim() == 2 && route_input.size(1) == 4096,
                "route input must have shape [M*6,4096]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32
                    && topk_weights.is_cuda()
                    && topk_weights.is_contiguous(),
                "topk_weights must be contiguous CUDA float32");
    TORCH_CHECK(symm_input.scalar_type() == torch::kBFloat16
                    && symm_input.is_cuda() && symm_input.is_contiguous(),
                "symmetric input must be contiguous CUDA bfloat16");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.is_cuda() && output.is_contiguous(),
                "output must be contiguous CUDA bfloat16");
    TORCH_CHECK(symm_input.sizes() == output.sizes()
                    && output.dim() == 2 && output.size(1) == 4096,
                "symmetric input/output must have shape [M,4096]");
    TORCH_CHECK(route_input.size(0) == output.size(0) * 6
                    && topk_weights.numel() == output.size(0) * 6,
                "route/topk shapes do not match output M");
    TORCH_CHECK(sem_local.scalar_type() == torch::kUInt8
                    && sem_local.is_cuda() && sem_local.is_contiguous(),
                "local semaphore slab must be contiguous CUDA uint8");
    TORCH_CHECK(symm_input_mc_ptr != 0 && sem_mc_ptr != 0,
                "k6 NVLS pull requires multicast input/semaphore VAs");
    TORCH_CHECK(active_blocks > 0 && active_blocks <= 64
                    && sem_local.numel() >= active_blocks * 128,
                "k6 NVLS pull block count exceeds semaphore capacity");
    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_k6_nvls_pull_tp4_kernel<256><<<active_blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(route_input.data_ptr()),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(symm_input.data_ptr()),
        reinterpret_cast<const uint8_t*>(symm_input_mc_ptr),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        sem_local.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(sem_mc_ptr),
        output.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_align(
        torch::Tensor topk_ids, torch::Tensor sorted_ids,
        torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
        torch::Tensor route_to_sorted) {
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32
                    && topk_ids.is_cuda() && topk_ids.is_contiguous()
                    && topk_ids.dim() == 2 && topk_ids.size(1) == 6,
                "topk_ids must be contiguous CUDA int32 [M,6]");
    TORCH_CHECK(sorted_ids.scalar_type() == torch::kInt32
                    && sorted_ids.is_cuda() && sorted_ids.is_contiguous(),
                "sorted_ids must be contiguous CUDA int32");
    TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32
                    && expert_ids.is_cuda() && expert_ids.is_contiguous(),
                "expert_ids must be contiguous CUDA int32");
    TORCH_CHECK(num_tokens_padded.scalar_type() == torch::kInt32
                    && num_tokens_padded.is_cuda()
                    && num_tokens_padded.numel() == 1,
                "num_tokens_padded must be one CUDA int32");
    const int routes = topk_ids.numel();
    TORCH_CHECK(!(kW2SortedAct || kW2MblockScale)
                    || route_to_sorted.numel() == routes,
                "sorted W2 activation/scale requires one position per route");
    const auto stream = at::cuda::getCurrentCUDAStream();
    route_align_kernel<<<1, 256, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), sorted_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(), num_tokens_padded.data_ptr<int32_t>(),
        route_to_sorted.numel() ? route_to_sorted.data_ptr<int32_t>() : nullptr,
        routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_route_quant(
        torch::Tensor topk_ids, torch::Tensor input,
        torch::Tensor sorted_ids, torch::Tensor expert_ids,
        torch::Tensor num_tokens_padded, torch::Tensor quantized,
        torch::Tensor scale, torch::Tensor route_to_sorted) {
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
    TORCH_CHECK(!(kW2SortedAct || kW2MblockScale)
                    || route_to_sorted.numel() == routes,
                "sorted W2 activation/scale requires one position per route");
    const int blocks = input.size(0) * 16;
    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_route_quant_kernel<<<blocks, 256, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_padded.data_ptr<int32_t>(),
        quantized.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        route_to_sorted.data_ptr<int32_t>(), routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void interleaved_scheduler_probe(
        torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
        torch::Tensor counters, torch::Tensor readiness,
        torch::Tensor ready_queue, torch::Tensor ready_valid,
        torch::Tensor w13_owner, torch::Tensor w13_order,
        torch::Tensor w2_owner, torch::Tensor w2_mblock,
        torch::Tensor w2_order, int w13_tiles, int w2_tiles,
        int num_sms) {
    TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32,
                "expert_ids must be int32");
    TORCH_CHECK(num_tokens_padded.scalar_type() == torch::kInt32,
                "num_tokens_padded must be int32");
    TORCH_CHECK(counters.scalar_type() == torch::kInt32
                    && counters.numel() >= 12,
                "counters must contain at least twelve int32 values");
    TORCH_CHECK(readiness.scalar_type() == torch::kInt32
                    && ready_queue.scalar_type() == torch::kInt32
                    && ready_valid.scalar_type() == torch::kInt32,
                "scheduler state tensors must be int32");
    TORCH_CHECK(w13_owner.scalar_type() == torch::kInt32
                    && w13_order.scalar_type() == torch::kInt32
                    && w2_owner.scalar_type() == torch::kInt32
                    && w2_mblock.scalar_type() == torch::kInt32
                    && w2_order.scalar_type() == torch::kInt32,
                "scheduler trace tensors must be int32");
    TORCH_CHECK(w13_tiles > 0 && w2_tiles > 0 && num_sms > 0,
                "scheduler geometry must be positive");
    const int max_mblocks = readiness.numel();
    TORCH_CHECK(ready_queue.numel() >= max_mblocks
                    && ready_valid.numel() >= max_mblocks,
                "ready queue capacity is too small");
    TORCH_CHECK(w13_owner.numel() >= max_mblocks * w13_tiles
                    && w13_order.numel() >= max_mblocks * w13_tiles,
                "W13 trace capacity is too small");
    TORCH_CHECK(w2_owner.numel() >= max_mblocks * w2_tiles
                    && w2_mblock.numel() >= max_mblocks * w2_tiles
                    && w2_order.numel() >= max_mblocks * w2_tiles,
                "W2 trace capacity is too small");
    const auto stream = at::cuda::getCurrentCUDAStream();
    interleaved_scheduler_probe_kernel<<<num_sms, 32, 0, stream>>>(
        expert_ids.data_ptr<int32_t>(),
        num_tokens_padded.data_ptr<int32_t>(),
        counters.data_ptr<int32_t>(), readiness.data_ptr<int32_t>(),
        ready_queue.data_ptr<int32_t>(), ready_valid.data_ptr<int32_t>(),
        w13_owner.data_ptr<int32_t>(), w13_order.data_ptr<int32_t>(),
        w2_owner.data_ptr<int32_t>(), w2_mblock.data_ptr<int32_t>(),
        w2_order.data_ptr<int32_t>(), max_mblocks, w13_tiles, w2_tiles);
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
void run_w13_paired_impl(
    torch::Tensor weight, torch::Tensor weight_global_scale,
    torch::Tensor activation, torch::Tensor activation_scale,
    torch::Tensor sorted_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_padded, torch::Tensor raw_output,
    torch::Tensor bf16_output, torch::Tensor quantized_output,
    torch::Tensor output_scale, int intermediate);
void run_w2(torch::Tensor weight, torch::Tensor weight_scale,
            torch::Tensor weight_global_scale,
            torch::Tensor activation, torch::Tensor activation_scale,
            torch::Tensor sorted_ids, torch::Tensor expert_ids,
            torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
            torch::Tensor output, torch::Tensor lut, int intermediate);
void run_w2_progress(
    torch::Tensor weight, torch::Tensor weight_scale,
    torch::Tensor weight_global_scale,
    torch::Tensor activation, torch::Tensor activation_scale,
    torch::Tensor sorted_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_padded, torch::Tensor topk_weights,
    torch::Tensor output, torch::Tensor lut,
    torch::Tensor progress_state, int intermediate);
void run_w2_chunk(torch::Tensor weight, torch::Tensor weight_scale,
                  torch::Tensor weight_global_scale,
                  torch::Tensor activation, torch::Tensor activation_scale,
                  torch::Tensor sorted_ids, torch::Tensor expert_ids,
                  torch::Tensor num_tokens_padded,
                  torch::Tensor topk_weights, torch::Tensor output,
                  torch::Tensor lut, int intermediate,
                  int chunks, int chunk_idx);
void reduce_swiglu(torch::Tensor partials, torch::Tensor output,
                   int intermediate, int split_k);
void reduce_swiglu_quant(torch::Tensor partials, torch::Tensor activation,
                         torch::Tensor quantized, torch::Tensor scale,
                         int intermediate, int split_k,
                         torch::Tensor route_to_sorted,
                         torch::Tensor topk_ids,
                         torch::Tensor w2_global_scale);
void cast_bf16(torch::Tensor input, torch::Tensor output);
void tiled_k6_reduce(torch::Tensor input, torch::Tensor topk_weights,
                     torch::Tensor output, int mode);
void run_tp4_megamoe_single_launch(
    torch::Tensor w13, torch::Tensor s13, torch::Tensor g13,
    torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
    torch::Tensor qx, torch::Tensor x_scale, torch::Tensor topk_ids,
    torch::Tensor topk_weights, torch::Tensor sorted_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
    torch::Tensor partials,
    torch::Tensor activation, torch::Tensor qactivation,
    torch::Tensor activation_scale, torch::Tensor down,
    torch::Tensor lut, torch::Tensor barrier_state,
    torch::Tensor route_to_sorted, torch::Tensor output,
    torch::Tensor push_counter, torch::Tensor push0,
    torch::Tensor push1, torch::Tensor push2, torch::Tensor push3,
    torch::Tensor pull_input, torch::Tensor pull_sem_local,
    int rank, int64_t push_stride, int64_t push_mc_ptr,
    int64_t pull_input_mc_ptr, int64_t pull_sem_mc_ptr,
    int split_k, int requested_ctas_per_sm);
void fused_k6_push_ar_tp4(
    torch::Tensor input, torch::Tensor topk_weights,
    torch::Tensor output, torch::Tensor push_counter,
    torch::Tensor push0, torch::Tensor push1,
    torch::Tensor push2, torch::Tensor push3,
    int rank, int64_t push_stride, int64_t push_mc_ptr);
void fused_k6_push_ar_tp4_chunk(
    torch::Tensor input, torch::Tensor topk_weights,
    torch::Tensor output, torch::Tensor push_counter,
    torch::Tensor push0, torch::Tensor push1,
    torch::Tensor push2, torch::Tensor push3,
    int rank, int64_t push_stride, int64_t push_mc_ptr,
    int chunks, int chunk_idx, int active_blocks);
void progress_k6_mc_push_tp4(
    torch::Tensor input, torch::Tensor topk_weights, torch::Tensor output,
    torch::Tensor progress_state, torch::Tensor push_counter,
    torch::Tensor push0, torch::Tensor push1,
    torch::Tensor push2, torch::Tensor push3,
    int64_t push_mc_ptr, int rank, int64_t push_stride,
    int chunks, int active_blocks, bool inline_finish);
void progress_mc_push_finish_tp4(
    torch::Tensor output, torch::Tensor push_counter,
    torch::Tensor push0, torch::Tensor push1,
    torch::Tensor push2, torch::Tensor push3,
    int rank, int64_t push_stride);
void fused_rank_route_mc_pull_tp4(
    torch::Tensor route_input, torch::Tensor topk_weights,
    torch::Tensor output, torch::Tensor sem_local,
    int64_t route_mc_ptr, int64_t sem_mc_ptr, int active_blocks);
void fused_k6_nvls_pull_tp4(
    torch::Tensor route_input, torch::Tensor topk_weights,
    torch::Tensor symm_input, torch::Tensor output,
    torch::Tensor sem_local, int64_t symm_input_mc_ptr,
    int64_t sem_mc_ptr, int active_blocks);
void route_align(torch::Tensor topk_ids, torch::Tensor sorted_ids,
                 torch::Tensor expert_ids,
                 torch::Tensor num_tokens_padded,
                 torch::Tensor route_to_sorted);
void fused_route_quant(torch::Tensor topk_ids, torch::Tensor input,
                       torch::Tensor sorted_ids, torch::Tensor expert_ids,
                       torch::Tensor num_tokens_padded,
                       torch::Tensor quantized, torch::Tensor scale,
                       torch::Tensor route_to_sorted);
void interleaved_scheduler_probe(
    torch::Tensor expert_ids, torch::Tensor num_tokens_padded,
    torch::Tensor counters, torch::Tensor readiness,
    torch::Tensor ready_queue, torch::Tensor ready_valid,
    torch::Tensor w13_owner, torch::Tensor w13_order,
    torch::Tensor w2_owner, torch::Tensor w2_mblock,
    torch::Tensor w2_order, int w13_tiles, int w2_tiles, int num_sms);
void braid_mode2(torch::Tensor weight);
"""


_EXTENSION_CONFIG = (
          f"wo{WOUT}_lr{LUT_ROWS}_"
          f"sr{SCALE_QUAD_REUSE}_sb{SCALE_BUFFERS}_"
          f"st{WEIGHT_STAGES}_"
          f"ws{WEIGHT_SWIZZLE}_wca{int(WEIGHT_COMMON_ADDRESS)}_"
          f"dh{int(DEQUANT_DP4A_HI)}_dl{int(DEQUANT_DP4A_LO)}_"
          f"dsl{int(DEQUANT_SYNTH_LUT)}_"
          f"nws{int(NORMALIZED_WEIGHT_SCALE)}_"
          f"nsl{int(NORMALIZED_SHARED_LUT)}_"
          f"ael{int(ACTIVATION_EVICT_LAST)}_"
          f"ppa{int(PREDICATED_PADDED_ACTIVATION)}_"
          f"twl{int(TILED_WEIGHT_LAYOUT)}_"
          f"bwc{int(BULK_WEIGHT_COPY)}_tcs{int(TMA_CTA_SCOPE)}_"
          f"wef{int(WEIGHT_EVICT_FIRST)}_"
          f"wph{int(WEIGHT_POLICY_HOIST)}_"
          f"wpc{int(WEIGHT_POLICY_CONSTANT)}_"
          f"w2ne{int(W2_NO_WEIGHT_EVICT_FIRST)}_"
          f"ibc{int(INTERLEAVED_BULK_COPY)}_"
          f"cis{int(COMPACT_INTERLEAVED_SCALE)}_"
          f"m2{int(MODE2_BRAID)}_"
          f"ro{int(W2_ROUTE_OUTPUT)}_sa{int(W2_SORTED_ACT)}_"
          f"ms{int(W2_MBLOCK_SCALE)}_"
          f"fg{int(W2_FOLD_GLOBAL_SCALE)}_"
          f"cs{int(W2_COALESCED_STORE)}_"
          f"w2gl{int(W2_GLOBAL_LUT)}_"
          f"w2pf{int(W2_S2R_PREFETCH)}_w13pf{int(W13_S2R_PREFETCH)}_"
          f"lmw{int(LEADER_MBAR_WAIT)}_dba{int(DIRECT_BARRIER_ADDR)}_"
          f"ku2{int(ROUTE_K_UNROLL2)}_ku4{int(ROUTE_K_UNROLL4)}_"
          f"ku8{int(ROUTE_K_UNROLL8)}_"
          f"ku8s2{int(ROUTE_K_UNROLL8_SPLIT2)}_"
          f"w13ku8s2{int(W13_K_UNROLL8_SPLIT2)}_"
          f"w13ku16s2{int(W13_K_UNROLL16_SPLIT2)}_"
          f"dp{int(W13_DISTRIBUTED_PREP)}_w2dp{int(W2_DISTRIBUTED_PREP)}_"
          f"dwg{int(W13_DUAL_WG_SPLIT)}_"
          f"w13mg{int(W13_MERGED_WGMMA_GROUP)}_"
          f"sidag{int(SINGLE_LAUNCH_INTERLEAVED)}_"
          f"mb{MIN_BLOCKS_PER_SM}_w13lb10{int(W13_LAUNCH_BOUND_10)}_"
          f"w13msc{int(W13_MAX_SMEM_CARVEOUT)}_v163rel")
_EXTENSION_NAME = (
    f"v4tp_{hashlib.sha1(_EXTENSION_CONFIG.encode()).hexdigest()[:20]}_v163rel"
)

_ext = load_inline(
    name=_EXTENSION_NAME,
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=[
        "run_w13_impl", "run_w13_paired_impl", "run_w2", "run_w2_progress",
        "run_w2_chunk",
        "reduce_swiglu", "reduce_swiglu_quant",
        "cast_bf16",
        "tiled_k6_reduce",
        "run_tp4_megamoe_single_launch",
        "fused_k6_push_ar_tp4",
        "fused_k6_push_ar_tp4_chunk",
        "progress_k6_mc_push_tp4",
        "progress_mc_push_finish_tp4",
        "fused_rank_route_mc_pull_tp4",
        "fused_k6_nvls_pull_tp4",
        "route_align",
        "fused_route_quant",
        "interleaved_scheduler_probe",
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
        f"-DK_NORMALIZED_SHARED_LUT={int(NORMALIZED_SHARED_LUT)}",
        f"-DK_ACTIVATION_EVICT_LAST={int(ACTIVATION_EVICT_LAST)}",
        f"-DK_PREDICATED_PADDED_ACTIVATION={int(PREDICATED_PADDED_ACTIVATION)}",
        f"-DK_TILED_WEIGHT_LAYOUT={int(TILED_WEIGHT_LAYOUT)}",
        f"-DK_BULK_WEIGHT_COPY={int(BULK_WEIGHT_COPY)}",
        f"-DK_TMA_CTA_SCOPE={int(TMA_CTA_SCOPE)}",
        f"-DK_WEIGHT_EVICT_FIRST={int(WEIGHT_EVICT_FIRST)}",
        f"-DK_WEIGHT_POLICY_HOIST={int(WEIGHT_POLICY_HOIST)}",
        f"-DK_WEIGHT_POLICY_CONSTANT={int(WEIGHT_POLICY_CONSTANT)}",
        f"-DK_W2_NO_WEIGHT_EVICT_FIRST={int(W2_NO_WEIGHT_EVICT_FIRST)}",
        f"-DK_INTERLEAVED_BULK_COPY={int(INTERLEAVED_BULK_COPY)}",
        f"-DK_COMPACT_INTERLEAVED_SCALE={int(COMPACT_INTERLEAVED_SCALE)}",
        f"-DK_MODE2_BRAID={int(MODE2_BRAID)}",
        f"-DK_W2_GLOBAL_LUT={int(W2_GLOBAL_LUT)}",
        f"-DK_W2_S2R_PREFETCH={int(W2_S2R_PREFETCH)}",
        f"-DK_W13_S2R_PREFETCH={int(W13_S2R_PREFETCH)}",
        f"-DK_LEADER_MBAR_WAIT={int(LEADER_MBAR_WAIT)}",
        f"-DK_DIRECT_BARRIER_ADDR={int(DIRECT_BARRIER_ADDR)}",
        f"-DK_ROUTE_K_UNROLL2={int(ROUTE_K_UNROLL2)}",
        f"-DK_ROUTE_K_UNROLL4={int(ROUTE_K_UNROLL4)}",
        f"-DK_ROUTE_K_UNROLL8={int(ROUTE_K_UNROLL8)}",
        f"-DK_ROUTE_K_UNROLL8_SPLIT2={int(ROUTE_K_UNROLL8_SPLIT2)}",
        f"-DK_W13_K_UNROLL8_SPLIT2={int(W13_K_UNROLL8_SPLIT2)}",
        f"-DK_W13_K_UNROLL16_SPLIT2={int(W13_K_UNROLL16_SPLIT2)}",
        f"-DK_W13_DISTRIBUTED_PREP={int(W13_DISTRIBUTED_PREP)}",
        f"-DK_W13_DUAL_WG_SPLIT={int(W13_DUAL_WG_SPLIT)}",
        f"-DK_W2_DISTRIBUTED_PREP={int(W2_DISTRIBUTED_PREP)}",
        f"-DK_W13_MERGED_WGMMA_GROUP={int(W13_MERGED_WGMMA_GROUP)}",
        f"-DK_W2_ROUTE_OUTPUT={int(W2_ROUTE_OUTPUT)}",
        f"-DK_W2_SORTED_ACT={int(W2_SORTED_ACT)}",
        f"-DK_W2_MBLOCK_SCALE={int(W2_MBLOCK_SCALE)}",
        f"-DK_W2_FOLD_GLOBAL_SCALE={int(W2_FOLD_GLOBAL_SCALE)}",
        f"-DK_W2_COALESCED_STORE={int(W2_COALESCED_STORE)}",
        f"-DK_SINGLE_LAUNCH_INTERLEAVED={int(SINGLE_LAUNCH_INTERLEAVED)}",
        f"-DK_MIN_BLOCKS_PER_SM={MIN_BLOCKS_PER_SM}",
        f"-DK_W13_LAUNCH_BOUND_10={int(W13_LAUNCH_BOUND_10)}",
        f"-DK_W13_MAX_SMEM_CARVEOUT={int(W13_MAX_SMEM_CARVEOUT)}",
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


def pair_gate_up_weight_layout(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave W13 gate/up rows in eight-channel chunks at model load."""
    if weight.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError("paired W13 layout expects rank-three tensors")
    experts, raw_channels, packed_k = weight.shape
    if raw_channels % 16:
        raise ValueError("paired W13 output width must be divisible by 16")
    if weight_scale.shape[:2] != (experts, raw_channels):
        raise ValueError("paired W13 weight/scale shapes do not match")
    intermediate = raw_channels // 2
    weight = (
        weight.view(experts, 2, intermediate // 8, 8, packed_k)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view_as(weight)
    )
    scale_k = weight_scale.shape[-1]
    weight_scale = (
        weight_scale.view(experts, 2, intermediate // 8, 8, scale_k)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view_as(weight_scale)
    )
    return weight, weight_scale


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
    if COMPACT_INTERLEAVED_SCALE:
        if weight_scale.shape[-1] != ktiles * 4:
            raise ValueError("E8M0 scale shape does not match group size 32")
        tiled_scale = (
            weight_scale.view(experts, ntiles, WOUT, ktiles, 4)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    elif weight_scale.shape[-1] >= 16:
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
    if INTERLEAVED_BULK_COPY and COMPACT_INTERLEAVED_SCALE:
        weight_tiles = tiled_weight.view(experts, ntiles, ktiles, -1)
        scale_tiles = tiled_scale.view(experts, ntiles, ktiles, -1)
        records: list[torch.Tensor] = []
        for kt in range(ktiles):
            records.append(weight_tiles[:, :, kt])
            records.append(scale_tiles[:, :, kt])
        tiled_weight = torch.cat(records, dim=-1).contiguous()
        expected_bytes = ktiles * WOUT * ((128 // 2) + 4)
        if tiled_weight.shape[-1] != expected_bytes:
            raise RuntimeError(
                "compact interleaved MXFP4 tile packing is incomplete"
            )
        tiled_scale = torch.empty(0, dtype=torch.uint8, device=weight.device)
    elif INTERLEAVED_BULK_COPY and weight_scale.shape[-1] >= 16:
        weight_tiles = tiled_weight.view(experts, ntiles, ktiles, -1)
        scale_tiles = tiled_scale.view(
            experts, ntiles, weight_scale.shape[-1] // 16, -1
        )
        records: list[torch.Tensor] = []
        for kt in range(ktiles):
            records.append(weight_tiles[:, :, kt])
            if kt % 8 == 0:
                records.append(scale_tiles[:, :, kt // 4])
            elif kt % 8 == 3 and kt + 1 < ktiles:
                records.append(scale_tiles[:, :, (kt + 1) // 4])
        tiled_weight = torch.cat(records, dim=-1).contiguous()
        expected_bytes = (
            ktiles * WOUT * (128 // 2)
            + (ktiles // 4) * WOUT * 16
        )
        if tiled_weight.shape[-1] != expected_bytes:
            raise RuntimeError("interleaved MXFP4 tile packing is incomplete")
        tiled_scale = torch.empty(0, dtype=torch.uint8, device=weight.device)
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
    if split_k not in (1, 2, 4):
        raise ValueError("W13 split_k must be 1, 2, or 4")
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


def run_w13_paired(
    weight: torch.Tensor,
    weight_global_scale: torch.Tensor,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    raw_output: torch.Tensor,
    bf16_output: torch.Tensor,
    quantized_output: torch.Tensor,
    output_scale: torch.Tensor,
    intermediate: int,
) -> None:
    if WOUT != 128:
        raise ValueError("paired W13 requires V4_WOUT=128")
    _ext.run_w13_paired_impl(
        weight,
        weight_global_scale,
        activation,
        activation_scale,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        raw_output,
        bf16_output,
        quantized_output,
        output_scale,
        intermediate,
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


def run_w2_progress(
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
    progress_state: torch.Tensor,
    intermediate: int,
) -> None:
    _ext.run_w2_progress(
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
        progress_state,
        intermediate,
    )


def run_w2_chunk(
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
    chunks: int,
    chunk_idx: int,
) -> None:
    if WOUT != 128:
        raise ValueError("pipelined W2 currently requires V4_WOUT=128")
    if chunks not in (2, 4, 8) or not 0 <= chunk_idx < chunks:
        raise ValueError("invalid W2 pipeline chunk")
    _ext.run_w2_chunk(
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
        chunks,
        chunk_idx,
    )


def reduce_swiglu(
    partials: torch.Tensor,
    output: torch.Tensor,
    intermediate: int,
    split_k: int | None = None,
) -> None:
    if split_k is None:
        split_k = select_w13_split_k(partials.size(1))
    if split_k not in (1, 2, 4):
        raise ValueError("W13 split_k must be 1, 2, or 4")
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
    route_to_sorted: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
) -> None:
    if split_k is None:
        split_k = select_w13_split_k(partials.size(1))
    if split_k not in (1, 2, 4):
        raise ValueError("W13 split_k must be 1, 2, or 4")
    if route_to_sorted is None:
        if W2_NEEDS_ROUTE_MAP:
            raise ValueError(
                "sorted W2 activation/scale requires route_to_sorted"
            )
        route_to_sorted = torch.empty(
            0, dtype=torch.int32, device=partials.device
        )
    if topk_ids is None:
        if W2_FOLD_GLOBAL_SCALE:
            raise ValueError("folded W2 scale requires topk_ids")
        topk_ids = torch.empty(0, dtype=torch.int32, device=partials.device)
    if w2_global_scale is None:
        if W2_FOLD_GLOBAL_SCALE:
            raise ValueError("folded W2 scale requires w2_global_scale")
        w2_global_scale = torch.empty(
            0, dtype=torch.float32, device=partials.device
        )
    _ext.reduce_swiglu_quant(
        partials,
        activation,
        quantized,
        scale,
        intermediate,
        split_k,
        route_to_sorted,
        topk_ids,
        w2_global_scale,
    )


def cast_bf16(input: torch.Tensor, output: torch.Tensor) -> None:
    _ext.cast_bf16(input, output)


def interleaved_scheduler_probe(
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    counters: torch.Tensor,
    readiness: torch.Tensor,
    ready_queue: torch.Tensor,
    ready_valid: torch.Tensor,
    w13_owner: torch.Tensor,
    w13_order: torch.Tensor,
    w2_owner: torch.Tensor,
    w2_mblock: torch.Tensor,
    w2_order: torch.Tensor,
    w13_tiles: int,
    w2_tiles: int,
    num_sms: int,
) -> None:
    _ext.interleaved_scheduler_probe(
        expert_ids,
        num_tokens_padded,
        counters,
        readiness,
        ready_queue,
        ready_valid,
        w13_owner,
        w13_order,
        w2_owner,
        w2_mblock,
        w2_order,
        w13_tiles,
        w2_tiles,
        num_sms,
    )


def tiled_k6_reduce(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    mode: int,
) -> None:
    if mode not in (1, 2, 3, 4):
        raise ValueError("tiled k6 reduce mode must be one of 1,2,3,4")
    _ext.tiled_k6_reduce(input, topk_weights, output, mode)


def run_tp4_megamoe_single_launch(
    w13: torch.Tensor,
    s13: torch.Tensor,
    g13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    g2: torch.Tensor,
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    partials: torch.Tensor,
    activation: torch.Tensor,
    qactivation: torch.Tensor,
    activation_scale: torch.Tensor,
    down: torch.Tensor,
    lut: torch.Tensor,
    barrier_state: torch.Tensor,
    route_to_sorted: torch.Tensor,
    output: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pull_input: torch.Tensor,
    pull_sem_local: torch.Tensor,
    rank: int,
    push_stride: int,
    push_mc_ptr: int,
    pull_input_mc_ptr: int,
    pull_sem_mc_ptr: int,
    split_k: int,
) -> None:
    """Run FP8-input TP4 MoE and multicast all-reduce in one launch."""
    _ext.run_tp4_megamoe_single_launch(
        w13,
        s13,
        g13,
        w2,
        s2,
        g2,
        qx.view(torch.uint8),
        x_scale,
        topk_ids,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        partials,
        activation,
        qactivation.view(torch.uint8),
        activation_scale,
        down,
        lut,
        barrier_state,
        route_to_sorted,
        output,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        pull_input,
        pull_sem_local,
        rank,
        push_stride,
        push_mc_ptr,
        pull_input_mc_ptr,
        pull_sem_mc_ptr,
        split_k,
        SINGLE_LAUNCH_CTAS_PER_SM,
    )


def fused_k6_push_ar_tp4(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    push_stride: int,
    push_mc_ptr: int = 0,
) -> None:
    _ext.fused_k6_push_ar_tp4(
        input,
        topk_weights,
        output,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        rank,
        push_stride,
        push_mc_ptr,
    )


def fused_k6_push_ar_tp4_chunk(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    push_stride: int,
    push_mc_ptr: int,
    chunks: int,
    chunk_idx: int,
    active_blocks: int,
) -> None:
    _ext.fused_k6_push_ar_tp4_chunk(
        input,
        topk_weights,
        output,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        rank,
        push_stride,
        push_mc_ptr,
        chunks,
        chunk_idx,
        active_blocks,
    )


def progress_k6_mc_push_tp4(
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    progress_state: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    push_mc_ptr: int,
    rank: int,
    push_stride: int,
    active_blocks: int,
) -> None:
    _ext.progress_k6_mc_push_tp4(
        input,
        topk_weights,
        output,
        progress_state,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        push_mc_ptr,
        rank,
        push_stride,
        W2_PROGRESS_CHUNKS,
        active_blocks,
        W2_PROGRESS_INLINE_FINISH,
    )


def progress_mc_push_finish_tp4(
    output: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
    push_stride: int,
) -> None:
    _ext.progress_mc_push_finish_tp4(
        output,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        rank,
        push_stride,
    )


def fused_rank_route_mc_pull_tp4(
    route_input: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    sem_local: torch.Tensor,
    route_mc_ptr: int,
    sem_mc_ptr: int,
    active_blocks: int,
) -> None:
    _ext.fused_rank_route_mc_pull_tp4(
        route_input,
        topk_weights,
        output,
        sem_local,
        route_mc_ptr,
        sem_mc_ptr,
        active_blocks,
    )


def fused_k6_nvls_pull_tp4(
    route_input: torch.Tensor,
    topk_weights: torch.Tensor,
    symm_input: torch.Tensor,
    output: torch.Tensor,
    sem_local: torch.Tensor,
    symm_input_mc_ptr: int,
    sem_mc_ptr: int,
    active_blocks: int,
) -> None:
    _ext.fused_k6_nvls_pull_tp4(
        route_input,
        topk_weights,
        symm_input,
        output,
        sem_local,
        symm_input_mc_ptr,
        sem_mc_ptr,
        active_blocks,
    )


def route_align(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    route_to_sorted: torch.Tensor | None = None,
) -> None:
    """Build routed-MoE block metadata without quantizing the FP8 input."""
    if route_to_sorted is None:
        if W2_NEEDS_ROUTE_MAP:
            raise ValueError(
                "sorted W2 activation/scale requires route_to_sorted"
            )
        route_to_sorted = torch.empty(
            0, dtype=torch.int32, device=topk_ids.device
        )
    _ext.route_align(
        topk_ids,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        route_to_sorted,
    )


def fused_route_quant(
    topk_ids: torch.Tensor,
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    route_to_sorted: torch.Tensor | None = None,
) -> None:
    if route_to_sorted is None:
        if W2_NEEDS_ROUTE_MAP:
            raise ValueError(
                "sorted W2 activation/scale requires route_to_sorted"
            )
        route_to_sorted = torch.empty(
            0, dtype=torch.int32, device=input.device
        )
    _ext.fused_route_quant(
        topk_ids,
        input,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        quantized,
        scale,
        route_to_sorted,
    )


def braid_mode2_(weight: torch.Tensor) -> torch.Tensor:
    """Offline in-place Mode2 sign/magnitude braid; excluded from inference."""
    _ext.braid_mode2(weight)
    return weight
