#pragma once

#include <torch/extension.h>

namespace deep_gemm::mega {

// Losslessly regroup the FP4 nibbles in the 64-byte payload of every fused
// 80-byte NVFP4 row block. Scale bytes are untouched; the otherwise-unused
// padding word records persistent grouped-layout metadata.
void nvfp4_group_nibbles_inplace_sm90(const torch::Tensor& fused_weight);

} // namespace deep_gemm::mega
