"""Native one-CTA-per-SM MXFP4 MegaMoE kernel for V4 Flash TP.

The timed kernel starts from caller-provided FP8-E4M3 activations and FP32
group-128 activation scales.  Input quantization and router computation are
model/runtime work outside this module.  FC1, SwiGLU plus the intermediate
FP8 requantization, FC2, fixed-k6 combine, and TP communication execute in one
CUDA launch.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


HIDDEN = 4096
NUM_EXPERTS = 256
TOP_K = 6
MAX_TOKENS = 128
BLOCK_M = 8
MAX_POOL_TOKENS = 3072
PADDED_SF_POOL_TOKENS = (MAX_POOL_TOKENS // BLOCK_M) * 128

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext_v4_tp")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
os.environ.setdefault("MAX_JOBS", "2")

_include_candidates = (
    Path("/lustre/raplab/client/xutingz/fac/DeepGEMM/deep_gemm/include"),
    Path("/home/xutingz/fac/DeepGEMM/deep_gemm/include"),
)
DEEP_GEMM_INCLUDE = next((path for path in _include_candidates if path.exists()), None)
if DEEP_GEMM_INCLUDE is None:
    raise FileNotFoundError("Cannot locate the read-only DeepGEMM include tree")
REPO_INCLUDE = Path(__file__).resolve().parent


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _workspace_metadata_bytes() -> int:
    """Mirror deep_gemm::layout::Workspace for local EP-rank one."""
    num_ranks = 1
    num_max_recv_tokens_per_expert = num_ranks * MAX_TOKENS
    generic_pool_tokens = _align(
        num_ranks * MAX_TOKENS * min(TOP_K, NUM_EXPERTS)
        + NUM_EXPERTS * (192 - 1),
        384,
    )
    generic_pool_blocks = generic_pool_tokens // BLOCK_M
    num_bytes = 128
    num_bytes += NUM_EXPERTS * 8 * 2
    num_bytes += NUM_EXPERTS * 8
    num_bytes += _align(generic_pool_blocks, 2) * 4
    num_bytes += generic_pool_blocks * 8
    num_bytes += (
        NUM_EXPERTS
        * num_ranks
        * num_max_recv_tokens_per_expert
        * 4
    )
    num_bytes += generic_pool_tokens * 12
    return _align(num_bytes, 16)


@dataclass
class NativeWorkspace:
    storage: torch.Tensor
    qx: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    l1_acts: torch.Tensor
    l1_acts_sf: torch.Tensor
    l2_acts: torch.Tensor
    l2_acts_sf: torch.Tensor

    def load_inputs(
        self,
        qx: torch.Tensor,
        x_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Populate registered input views before graph capture/timing."""
        m = qx.size(0)
        self.qx[:m].copy_(qx)
        self.x_scale[:m].copy_(x_scale)
        self.topk_ids[:m].copy_(topk_ids.to(torch.int64))
        self.topk_weights[:m].copy_(topk_weights)


def allocate_workspace(
    intermediate_per_rank: int, device: torch.device | str
) -> NativeWorkspace:
    if intermediate_per_rank not in (256, 512):
        raise ValueError("native V4 TP workspace supports TP8/TP4 I=256/512")

    offset = _workspace_metadata_bytes()
    fields: dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]] = {}

    def reserve(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        nonlocal offset
        element_size = torch.empty((), dtype=dtype).element_size()
        count = 1
        for dim in shape:
            count *= dim
        fields[name] = (offset, count * element_size, dtype, shape)
        offset += count * element_size

    reserve("qx", (MAX_TOKENS, HIDDEN), torch.float8_e4m3fn)
    reserve("x_scale", (MAX_TOKENS, HIDDEN // 128), torch.float32)
    reserve("topk_ids", (MAX_TOKENS, TOP_K), torch.int64)
    reserve("topk_weights", (MAX_TOKENS, TOP_K), torch.float32)
    reserve("l1_acts", (MAX_POOL_TOKENS, HIDDEN), torch.float8_e4m3fn)
    reserve(
        "l1_acts_sf",
        (HIDDEN // 128, PADDED_SF_POOL_TOKENS),
        torch.float32,
    )
    # This four-byte-per-row buffer is part of the body layout even though it
    # is not consumed by the host launcher directly.
    reserve("l1_topk_weights", (MAX_POOL_TOKENS,), torch.float32)
    reserve(
        "l2_acts",
        (MAX_POOL_TOKENS, intermediate_per_rank),
        torch.float8_e4m3fn,
    )
    # The body reserves physical per-64 capacity and uses its first per-128
    # half for BM8/BN256.
    reserve(
        "l2_acts_sf",
        (intermediate_per_rank // 16, PADDED_SF_POOL_TOKENS),
        torch.float32,
    )
    reserve(
        "combine",
        (TOP_K, MAX_TOKENS, HIDDEN),
        torch.bfloat16,
    )

    storage = torch.zeros((offset,), dtype=torch.uint8, device=device)

    def view(name: str) -> torch.Tensor:
        byte_offset, num_bytes, dtype, shape = fields[name]
        return storage.narrow(0, byte_offset, num_bytes).view(dtype).view(shape)

    return NativeWorkspace(
        storage=storage,
        qx=view("qx"),
        x_scale=view("x_scale"),
        topk_ids=view("topk_ids"),
        topk_weights=view("topk_weights"),
        l1_acts=view("l1_acts"),
        l1_acts_sf=view("l1_acts_sf"),
        l2_acts=view("l2_acts"),
        l2_acts_sf=view("l2_acts_sf"),
    )


def _interleave_l1(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
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


def _scale_to_tile_major(scale: torch.Tensor) -> torch.Tensor:
    experts, rows, groups = scale.shape
    block_n, block_k, group_size = 256, 128, 32
    groups_per_k_block = block_k // group_size
    if rows % block_n or groups % groups_per_k_block:
        raise ValueError("MXFP4 scale shape is not divisible by N256/K128")
    tile_major = (
        scale.view(
            experts,
            rows // block_n,
            block_n,
            groups // groups_per_k_block,
            groups_per_k_block,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return tile_major.repeat_interleave(2, dim=-1).contiguous()


def _fuse_packed_and_scale(
    packed: torch.Tensor, scale_tile_major: torch.Tensor
) -> torch.Tensor:
    experts, rows, packed_k = packed.shape
    block_n, block_k = 256, 128
    k_blocks = packed_k // (block_k // 2)
    fused = torch.zeros(
        (experts, rows // block_n, k_blocks, block_n, 80),
        dtype=torch.uint8,
        device=packed.device,
    )
    packed_tile = (
        packed.view(experts, rows // block_n, block_n, k_blocks, 64)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    fused[..., :64] = packed_tile
    fused[..., 64:72] = scale_tile_major
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(experts, rows, k_blocks * 80)
        .contiguous()
    )


def _braid_mode2_signs(fused_weight: torch.Tensor) -> torch.Tensor:
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
    braided_nibbles = magnitudes | (braided_signs << 3)
    fused_rows[..., :64] = (
        braided_nibbles[..., 0::2] | (braided_nibbles[..., 1::2] << 4)
    ).reshape(experts, rows, storage_k // 80, 64)
    return fused_rows.view(experts, rows, storage_k).contiguous()


def transform_weights(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the fused 80-byte Mode2 rows once at model-load time."""
    if any(t.dtype != torch.uint8 or t.ndim != 3 for t in (w13, s13, w2, s2)):
        raise TypeError("native MXFP4 weights/scales must be rank-three uint8")
    w13_il = _interleave_l1(w13)
    s13_il = _interleave_l1(s13)
    native_w13 = _braid_mode2_signs(
        _fuse_packed_and_scale(w13_il, _scale_to_tile_major(s13_il))
    )
    native_w2 = _braid_mode2_signs(
        _fuse_packed_and_scale(w2.contiguous(), _scale_to_tile_major(s2))
    )
    return native_w13, native_w2


_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <cuda.h>
#include <cuda_bf16.h>

#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

using namespace deep_gemm;

namespace deep_gemm {

__device__ __forceinline__ uint4 native_load_relaxed_sys_16b(
        const void* pointer) {
    uint4 value;
    asm volatile(
        "ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ void native_store_multimem_16b(
        void* pointer, const uint4& value) {
    const float4 bits = *reinterpret_cast<const float4*>(&value);
    asm volatile(
        "multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
        :
        : "f"(bits.x), "f"(bits.y), "f"(bits.z), "f"(bits.w),
          "l"(pointer)
        : "memory");
}

__device__ __forceinline__ uint4 native_load_multimem_reduce_bf16_16b(
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

__device__ __forceinline__ uint32_t native_load_relaxed_sys_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.relaxed.sys.global.u32 %0, [%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t native_load_acquire_sys_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void native_multimem_red_add_relaxed_u32(
        uint32_t* pointer) {
    asm volatile(
        "multimem.red.relaxed.sys.global.add.u32 [%0], 1;"
        : : "l"(pointer) : "memory");
}

__device__ __forceinline__ void native_multimem_red_add_release_u32(
        uint32_t* pointer) {
    asm volatile(
        "multimem.red.release.sys.global.add.u32 [%0], 1;"
        : : "l"(pointer) : "memory");
}

template <int kThreads>
__device__ __forceinline__ void native_tp4_multicast_push(
        const __nv_bfloat16* __restrict__ local_output,
        __nv_bfloat16* __restrict__ output,
        uint32_t* __restrict__ push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* __restrict__ push_mc,
        const uint32_t num_tokens, const int rank, const int64_t push_stride,
        const int linear_block_idx, const int linear_grid_dim) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kVecsPerToken = kHidden / 8;
    constexpr float kRoutedScale = 1.5f;
    const int global_tid = linear_block_idx * kThreads + threadIdx.x;
    const int global_threads = linear_grid_dim * kThreads;
    const int num_vecs = static_cast<int>(num_tokens) * kVecsPerToken;
    const int phase = push_counter[linear_block_idx] & 1u;
    const int64_t phase_offset =
        static_cast<int64_t>(phase) * push_stride * kWorld;
    uint8_t* peer_base[kWorld] = {push0, push1, push2, push3};

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const uint4 local_bits = reinterpret_cast<const uint4*>(local_output)[vec];
        const auto* local_pairs =
            reinterpret_cast<const __nv_bfloat162*>(&local_bits);
        uint4 scaled;
        auto* scaled_words = reinterpret_cast<uint32_t*>(&scaled);
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            float2 value = __bfloat1622float2(local_pairs[pair]);
            value.x *= kRoutedScale;
            value.y *= kRoutedScale;
            const __nv_bfloat162 casted =
                __floats2bfloat162_rn(value.x, value.y);
            uint32_t word = *reinterpret_cast<const uint32_t*>(&casted);
            scaled_words[pair] = word == 0u ? 0x00008000u : word;
        }

        const int64_t source_offset =
            static_cast<int64_t>(rank) * push_stride + phase_offset
            + static_cast<int64_t>(vec) * 16;
        native_store_multimem_16b(push_mc + source_offset, scaled);

        uint4 rank_vec[kWorld];
        const int64_t poll_offset =
            phase_offset + static_cast<int64_t>(vec) * 16;
        while (true) {
            bool missing = false;
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                rank_vec[source] = native_load_relaxed_sys_16b(
                    peer_base[rank] + source * push_stride + poll_offset);
                const auto* words =
                    reinterpret_cast<const uint32_t*>(&rank_vec[source]);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair)
                    missing |= words[pair] == 0u;
            }
            if (!missing)
                break;
        }

        uint4 sum_bits;
        auto* sum_words = reinterpret_cast<uint32_t*>(&sum_bits);
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            float2 sum = make_float2(0.0f, 0.0f);
            #pragma unroll
            for (int source = 0; source < kWorld; ++source) {
                const auto* source_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&rank_vec[source]);
                const float2 value = __bfloat1622float2(source_pairs[pair]);
                sum.x += value.x;
                sum.y += value.y;
            }
            const __nv_bfloat162 casted =
                __floats2bfloat162_rn(sum.x, sum.y);
            sum_words[pair] = *reinterpret_cast<const uint32_t*>(&casted);
        }
        reinterpret_cast<uint4*>(output)[vec] = sum_bits;

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

template <int kThreads>
__device__ __forceinline__ void native_tp4_nvls_pull(
        const __nv_bfloat16* __restrict__ local_output,
        __nv_bfloat16* __restrict__ symm_input,
        const uint8_t* __restrict__ symm_input_mc,
        __nv_bfloat16* __restrict__ output,
        uint8_t* __restrict__ sem_local,
        uint8_t* __restrict__ sem_mc,
        const uint32_t num_tokens,
        const int linear_block_idx, const int linear_grid_dim) {
    constexpr int kWorld = 4;
    constexpr int kHidden = 4096;
    constexpr int kVecsPerToken = kHidden / 8;
    constexpr int kSemaphoreBytes = 128;
    constexpr float kRoutedScale = 1.5f;
    const int global_tid = linear_block_idx * kThreads + threadIdx.x;
    const int global_threads = linear_grid_dim * kThreads;
    const int num_vecs = static_cast<int>(num_tokens) * kVecsPerToken;

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        const uint4 local_bits = reinterpret_cast<const uint4*>(local_output)[vec];
        const auto* local_pairs =
            reinterpret_cast<const __nv_bfloat162*>(&local_bits);
        uint4 scaled;
        auto* words = reinterpret_cast<uint32_t*>(&scaled);
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            float2 value = __bfloat1622float2(local_pairs[pair]);
            value.x *= kRoutedScale;
            value.y *= kRoutedScale;
            const __nv_bfloat162 casted =
                __floats2bfloat162_rn(value.x, value.y);
            words[pair] = *reinterpret_cast<const uint32_t*>(&casted);
        }
        reinterpret_cast<uint4*>(symm_input)[vec] = scaled;
    }

    __syncthreads();
    uint32_t barrier_current = 0;
    if (threadIdx.x == 0) {
        uint8_t* sem = sem_local + linear_block_idx * kSemaphoreBytes;
        auto* flag = reinterpret_cast<uint32_t*>(sem);
        auto* counter = reinterpret_cast<uint32_t*>(sem + sizeof(uint32_t));
        const uint32_t reserved = atomicAdd(counter, 2 * kWorld);
        barrier_current = reserved + kWorld;
        native_multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(
            sem_mc + linear_block_idx * kSemaphoreBytes));
        while (native_load_acquire_sys_u32(flag) - reserved < kWorld) {}
    }
    __syncthreads();

    for (int vec = global_tid; vec < num_vecs; vec += global_threads) {
        reinterpret_cast<uint4*>(output)[vec] =
            native_load_multimem_reduce_bf16_16b(
                symm_input_mc + static_cast<int64_t>(vec) * 16);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        auto* flag = reinterpret_cast<uint32_t*>(
            sem_local + linear_block_idx * kSemaphoreBytes);
        native_multimem_red_add_release_u32(reinterpret_cast<uint32_t*>(
            sem_mc + linear_block_idx * kSemaphoreBytes));
        while (native_load_acquire_sys_u32(flag) - barrier_current < kWorld) {}
    }
}

template <int kIntermediate>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
v4_flash_tp4_native_megamoe_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<1> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        __nv_bfloat16* output,
        uint32_t* push_counter,
        uint8_t* push0, uint8_t* push1, uint8_t* push2, uint8_t* push3,
        uint8_t* push_mc,
        __nv_bfloat16* pull_input,
        const uint8_t* pull_input_mc,
        uint8_t* pull_sem_local,
        uint8_t* pull_sem_mc,
        const int rank,
        const int64_t push_stride,
        const bool enable_tp) {
    constexpr uint32_t kNumMaxTokensPerRank = 128;
    constexpr uint32_t kNumExpertsPerWave = 32;
    constexpr uint32_t kNumSMs = 78;
    constexpr uint32_t kNumRanks = 1;
    constexpr uint32_t kNumExperts = 256;
    constexpr uint32_t BLOCK_M = 8;
    constexpr uint32_t BLOCK_N = 256;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumMaxPoolTokens = 3072;
    constexpr uint32_t kNumPaddedSFPoolTokens = 49152;
    constexpr uint32_t kNumStages = 4;
    constexpr float kActivationClamp = cute::numeric_limits<float>::infinity();
    constexpr bool kFastMath = true;
    constexpr bool kSwapABRequested = true;
    constexpr bool kSingleActiveDispatchWarp = true;
    // transform_weights() materializes the Mode2 sign/magnitude braid once at
    // model load, so select the matching braided row decoder in the body.
    constexpr bool kUseMode2RowDecoder = false;
    constexpr bool kUseInterleavedScheduler = true;
    constexpr uint32_t kHidden = 4096;
    constexpr uint32_t kIntermediateHidden = kIntermediate;
    constexpr uint32_t kNumTopk = 6;
    constexpr uint32_t kNumDispatchThreads = 64;
    constexpr uint32_t kNumNonEpilogueThreads = 64;
    constexpr uint32_t kNumEpilogueThreads = 256;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include "v4_flash_tp_native_body.inl"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000)
    // The last combine store is asynchronous.  Drain it before the local-rank
    // grid barrier transfers ownership of y to the communication tail.
    ptx::tma_store_wait<0>();
    comm::grid_sync<kNumSMs, 2>(
        workspace, sm_idx, thread_idx, []() { __syncthreads(); });
    if (!enable_tp)
        return;

    const auto* local_output = reinterpret_cast<const __nv_bfloat16*>(y);
    if (num_tokens == 128) {
        constexpr int kPullBlocks = 64;
        if (sm_idx < kPullBlocks) {
            native_tp4_nvls_pull<384>(
                local_output, pull_input, pull_input_mc, output,
                pull_sem_local, pull_sem_mc, num_tokens,
                static_cast<int>(sm_idx), kPullBlocks);
        }
    } else {
        native_tp4_multicast_push<384>(
            local_output, output, push_counter,
            push0, push1, push2, push3, push_mc,
            num_tokens, rank, push_stride,
            static_cast<int>(sm_idx), static_cast<int>(kNumSMs));
    }
#endif
}

}  // namespace deep_gemm

CUtensorMap native_make_desc(
        void* pointer,
        CUtensorMapDataType dtype,
        uint64_t inner,
        uint64_t outer,
        uint32_t box_inner,
        uint32_t box_outer,
        uint64_t outer_stride_bytes,
        CUtensorMapSwizzle swizzle) {
    CUtensorMap descriptor;
    const cuuint64_t global_dims[2] = {inner, outer};
    const cuuint64_t global_strides[1] = {outer_stride_bytes};
    const cuuint32_t box_dims[2] = {box_inner, box_outer};
    const cuuint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &descriptor, dtype, 2, pointer,
        global_dims, global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS,
                "native cuTensorMapEncodeTiled failed: ", result);
    return descriptor;
}

void run_native_tp4(
        torch::Tensor workspace,
        torch::Tensor l1_acts,
        torch::Tensor l1_acts_sf,
        torch::Tensor l2_acts,
        torch::Tensor l2_acts_sf,
        torch::Tensor w13,
        torch::Tensor w2,
        torch::Tensor local_output,
        torch::Tensor output,
        torch::Tensor push_counter,
        torch::Tensor push0,
        torch::Tensor push1,
        torch::Tensor push2,
        torch::Tensor push3,
        torch::Tensor pull_input,
        torch::Tensor pull_sem_local,
        int64_t push_mc_ptr,
        int64_t pull_input_mc_ptr,
        int64_t pull_sem_mc_ptr,
        int rank,
        int64_t push_stride,
        int tokens,
        int intermediate,
        bool enable_tp) {
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8
                    && workspace.is_cuda() && workspace.is_contiguous(),
                "native workspace must be contiguous CUDA uint8");
    TORCH_CHECK(tokens == 8 || tokens == 16 || tokens == 32
                    || tokens == 64 || tokens == 128,
                "native TP4 supports M=8,16,32,64,128");
    TORCH_CHECK(intermediate == 512,
                "native TP4 requires intermediate_per_rank=512");
    TORCH_CHECK(w13.scalar_type() == torch::kUInt8 && w13.is_contiguous()
                    && w13.sizes() == torch::IntArrayRef({256, 1024, 2560}),
                "native W13 must be uint8 [256,1024,2560]");
    TORCH_CHECK(w2.scalar_type() == torch::kUInt8 && w2.is_contiguous()
                    && w2.sizes() == torch::IntArrayRef({256, 4096, 320}),
                "native W2 must be uint8 [256,4096,320]");
    TORCH_CHECK(local_output.scalar_type() == torch::kBFloat16
                    && local_output.numel() == static_cast<int64_t>(tokens) * 4096,
                "native local output must be BF16 [M,4096]");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16
                    && output.numel() == local_output.numel(),
                "native final output must be BF16 [M,4096]");
    TORCH_CHECK(push_counter.is_cuda() && push_counter.element_size() == 4
                    && push_counter.numel() >= 78,
                "native TP4 needs at least 78 CARv2 push counters");
    TORCH_CHECK(!enable_tp || (rank >= 0 && rank < 4 && push_mc_ptr != 0),
                "native TP4 requires a valid rank and multicast push VA");
    TORCH_CHECK(!enable_tp || (pull_input_mc_ptr != 0 && pull_sem_mc_ptr != 0),
                "native TP4 requires multicast pull/semaphore VAs");

    static void* last_workspace = nullptr;
    static void* last_w13 = nullptr;
    static void* last_w2 = nullptr;
    static CUtensorMap tensor_map_l1_acts;
    static CUtensorMap tensor_map_l1_acts_sf;
    static CUtensorMap tensor_map_l1_weights;
    static CUtensorMap tensor_map_l1_output;
    static CUtensorMap tensor_map_l2_acts;
    static CUtensorMap tensor_map_l2_acts_sf;
    static CUtensorMap tensor_map_l2_weights;
    if (last_workspace != workspace.data_ptr()
            || last_w13 != w13.data_ptr()
            || last_w2 != w2.data_ptr()) {
        tensor_map_l1_acts = native_make_desc(
            l1_acts.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            4096, 3072, 128, 8, 4096,
            CU_TENSOR_MAP_SWIZZLE_128B);
        tensor_map_l1_acts_sf = native_make_desc(
            l1_acts_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            49152, 32, 8, 1, 49152 * sizeof(float),
            CU_TENSOR_MAP_SWIZZLE_NONE);
        tensor_map_l1_weights = native_make_desc(
            w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2560, 256 * 1024, 80, 256, 2560,
            CU_TENSOR_MAP_SWIZZLE_NONE);
        tensor_map_l1_output = native_make_desc(
            l2_acts.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            512, 3072, 128, 8, 512,
            CU_TENSOR_MAP_SWIZZLE_NONE);
        tensor_map_l2_acts = native_make_desc(
            l2_acts.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            512, 3072, 128, 8, 512,
            CU_TENSOR_MAP_SWIZZLE_128B);
        tensor_map_l2_acts_sf = native_make_desc(
            l2_acts_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            49152, 4, 8, 1, 49152 * sizeof(float),
            CU_TENSOR_MAP_SWIZZLE_NONE);
        tensor_map_l2_weights = native_make_desc(
            w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            320, 256 * 4096, 80, 256, 320,
            CU_TENSOR_MAP_SWIZZLE_NONE);
        last_workspace = workspace.data_ptr();
        last_w13 = w13.data_ptr();
        last_w2 = w2.data_ptr();
    }

    std::array<int64_t, 1> ptrs = {
        reinterpret_cast<int64_t>(workspace.data_ptr<uint8_t>())};
    const layout::SymBuffer<1> sym_buffer(ptrs, 0);
    auto kernel = v4_flash_tp4_native_megamoe_impl<512>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448));
    const auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<78, 384, 232448, stream>>>(
        local_output.data_ptr(), nullptr, static_cast<uint32_t>(tokens),
        sym_buffer,
        tensor_map_l1_acts, tensor_map_l1_acts_sf,
        tensor_map_l1_weights, tensor_map_l1_output,
        tensor_map_l2_acts, tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(push_counter.data_ptr()),
        push0.data_ptr<uint8_t>(), push1.data_ptr<uint8_t>(),
        push2.data_ptr<uint8_t>(), push3.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(push_mc_ptr),
        reinterpret_cast<__nv_bfloat16*>(pull_input.data_ptr()),
        reinterpret_cast<const uint8_t*>(pull_input_mc_ptr),
        pull_sem_local.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(pull_sem_mc_ptr),
        rank, push_stride, enable_tp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP = r"""
void run_native_tp4(
    torch::Tensor workspace,
    torch::Tensor l1_acts,
    torch::Tensor l1_acts_sf,
    torch::Tensor l2_acts,
    torch::Tensor l2_acts_sf,
    torch::Tensor w13,
    torch::Tensor w2,
    torch::Tensor local_output,
    torch::Tensor output,
    torch::Tensor push_counter,
    torch::Tensor push0,
    torch::Tensor push1,
    torch::Tensor push2,
    torch::Tensor push3,
    torch::Tensor pull_input,
    torch::Tensor pull_sem_local,
    int64_t push_mc_ptr,
    int64_t pull_input_mc_ptr,
    int64_t pull_sem_mc_ptr,
    int rank,
    int64_t push_stride,
    int tokens,
    int intermediate,
    bool enable_tp);
"""

_SOURCE_HASH = hashlib.sha1((_CPP + _CUDA).encode()).hexdigest()[:20]
_ext = load_inline(
    name=f"v4tp_native_megamoe_{_SOURCE_HASH}",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["run_native_tp4"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode",
        "arch=compute_90a,code=sm_90a",
        "-std=c++20",
        "-lineinfo",
        f"-I{DEEP_GEMM_INCLUDE}",
        f"-I{REPO_INCLUDE}",
    ],
    extra_ldflags=["-lcuda"],
    verbose=os.environ.get("V4_VERBOSE_BUILD", "0") == "1",
)


def run_tp4(
    workspace: NativeWorkspace,
    native_w13: torch.Tensor,
    native_w2: torch.Tensor,
    local_output: torch.Tensor,
    output: torch.Tensor,
    push_counter: torch.Tensor,
    push_workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pull_input: torch.Tensor,
    pull_sem_local: torch.Tensor,
    push_mc_ptr: int,
    pull_input_mc_ptr: int,
    pull_sem_mc_ptr: int,
    rank: int,
    push_stride: int,
    tokens: int,
) -> None:
    _ext.run_native_tp4(
        workspace.storage,
        workspace.l1_acts.view(torch.uint8),
        workspace.l1_acts_sf,
        workspace.l2_acts.view(torch.uint8),
        workspace.l2_acts_sf,
        native_w13,
        native_w2,
        local_output,
        output,
        push_counter,
        push_workspaces[0],
        push_workspaces[1],
        push_workspaces[2],
        push_workspaces[3],
        pull_input,
        pull_sem_local,
        push_mc_ptr,
        pull_input_mc_ptr,
        pull_sem_mc_ptr,
        rank,
        push_stride,
        tokens,
        512,
        True,
    )


def run_local(
    workspace: NativeWorkspace,
    native_w13: torch.Tensor,
    native_w2: torch.Tensor,
    local_output: torch.Tensor,
    tokens: int,
) -> None:
    """Diagnostic entry that executes the same body but skips the TP tail."""
    device = local_output.device
    dummy_counter = torch.zeros((78,), dtype=torch.int32, device=device)
    dummy_bytes = workspace.storage[:128]
    _ext.run_native_tp4(
        workspace.storage,
        workspace.l1_acts.view(torch.uint8),
        workspace.l1_acts_sf,
        workspace.l2_acts.view(torch.uint8),
        workspace.l2_acts_sf,
        native_w13,
        native_w2,
        local_output,
        local_output,
        dummy_counter,
        dummy_bytes,
        dummy_bytes,
        dummy_bytes,
        dummy_bytes,
        local_output,
        dummy_bytes,
        0,
        0,
        0,
        0,
        0,
        tokens,
        512,
        False,
    )
