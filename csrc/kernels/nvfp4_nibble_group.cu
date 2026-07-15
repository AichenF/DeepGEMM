#include "nvfp4_nibble_group.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>

namespace deep_gemm::mega {
namespace {

constexpr std::uint64_t kGroupedNibbleLayoutMarker = 0x21213176474e4744ull; // "DGNGv1!!"

__global__ void group_nvfp4_nibbles_kernel(std::uint8_t* data,
                                            const std::int64_t num_payload_words) {
    const auto word_idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word_idx >= num_payload_words)
        return;

    // Each fused BK128 row block is 80 bytes: 64 bytes of FP4 payload followed
    // by 8 bytes of scale and 8 bytes of padding. One thread converts a 4-byte
    // payload word, so 16 consecutive threads cover one payload block.
    const auto row_block_idx = word_idx >> 4;
    const auto word_in_payload = word_idx & 15;
    auto* marker_ptr = reinterpret_cast<std::uint64_t*>(
        data + row_block_idx * 80 + 72);
    const auto warp_mask = __activemask();
    const int lane_in_warp = threadIdx.x & 31;
    int row_is_already_grouped = 0;
    if ((lane_in_warp & 15) == 0)
        row_is_already_grouped = *marker_ptr == kGroupedNibbleLayoutMarker;
    // A row is exactly one aligned half-warp. Its leader samples the old marker
    // and broadcasts it before any lane can publish the new one. This makes a
    // sequential direct launch idempotent; the public wrapper additionally
    // serializes concurrent callers.
    row_is_already_grouped = __shfl_sync(
        warp_mask, row_is_already_grouped, lane_in_warp & 16);
    if (row_is_already_grouped)
        return;

    auto* word_ptr = reinterpret_cast<std::uint32_t*>(
        data + row_block_idx * 80 + word_in_payload * sizeof(std::uint32_t));
    const std::uint32_t packed = *word_ptr;

    // Input bytes are [lo_i | hi_i << 4] for i=0..3. The grouped decoder wants
    // [hi0|hi1<<4, hi2|hi3<<4, lo0|lo1<<4, lo2|lo3<<4].
    const std::uint32_t high_nibbles =
        ((packed >> 4)  & 0x0000000fu) |
        ((packed >> 8)  & 0x000000f0u) |
        ((packed >> 12) & 0x00000f00u) |
        ((packed >> 16) & 0x0000f000u);
    const std::uint32_t low_nibbles =
        ( packed        & 0x0000000fu) |
        ((packed >> 4)  & 0x000000f0u) |
        ((packed >> 8)  & 0x00000f00u) |
        ((packed >> 12) & 0x0000f000u);
    *word_ptr = high_nibbles | (low_nibbles << 16);
    // Publish the row marker only after all 16 payload words are visible.
    __syncwarp(__activemask());
    if (word_in_payload == 0)
        *marker_ptr = kGroupedNibbleLayoutMarker;
}

} // namespace

void nvfp4_group_nibbles_inplace_sm90(const torch::Tensor& fused_weight) {
    TORCH_CHECK(fused_weight.is_cuda(), "NVFP4 nibble grouping requires a CUDA tensor");
    TORCH_CHECK(fused_weight.scalar_type() == torch::kUInt8,
                "NVFP4 nibble grouping requires torch.uint8 weights");
    TORCH_CHECK(fused_weight.dim() == 3, "NVFP4 fused weights must be 3-dimensional");
    TORCH_CHECK(fused_weight.is_contiguous(), "NVFP4 fused weights must be contiguous");
    TORCH_CHECK(fused_weight.size(2) % 80 == 0,
                "NVFP4 fused weight storage width must be divisible by 80 bytes");
    TORCH_CHECK(fused_weight.storage_offset() == 0 and
                    fused_weight.nbytes() == fused_weight.storage().nbytes(),
                "raw NVFP4 nibble grouping requires a dedicated full storage; "
                "use nvfp4_group_nibbles_for_mega_moe_sm90 for managed aliases");

    c10::cuda::CUDAGuard device_guard(fused_weight.device());
    const std::int64_t num_row_blocks =
        fused_weight.size(0) * fused_weight.size(1) * (fused_weight.size(2) / 80);
    const std::int64_t num_payload_words = num_row_blocks * 16;
    if (num_payload_words == 0)
        return;

    constexpr int kThreads = 256;
    const std::int64_t num_blocks = (num_payload_words + kThreads - 1) / kThreads;
    TORCH_CHECK(num_blocks <= std::numeric_limits<unsigned int>::max(),
                "NVFP4 nibble grouping tensor is too large for a CUDA launch");
    const auto stream = at::cuda::getCurrentCUDAStream(fused_weight.device().index());
    group_nvfp4_nibbles_kernel<<<static_cast<unsigned int>(num_blocks), kThreads, 0, stream>>>(
        fused_weight.data_ptr<std::uint8_t>(), num_payload_words);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace deep_gemm::mega
