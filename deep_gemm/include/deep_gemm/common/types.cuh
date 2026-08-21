#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    // NVFP4: both operands are packed E2M1 with per-16-element E4M3 scale factors
    NVFP4       = 2,
};

constexpr CUTLASS_HOST_DEVICE int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        // NVFP4 is sub-byte and has no integer byte size. Bit-aware MegaMoE
        // helpers handle it explicitly; generic descriptor paths must not.
        case MmaKind::NVFP4:    return 0;
        default: return 0;
    }
}

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

} // namespace deep_gemm
