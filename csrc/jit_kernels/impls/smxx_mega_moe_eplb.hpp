#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

#include <deep_gemm/layout/mega_moe_eplb.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

class MegaMoEEPLBPlanRuntime final: public LaunchRuntime<MegaMoEEPLBPlanRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, num_helper_slots;
        layout::SymBuffer<> workspace;
        const int64_t* topk_idx;
        int num_tokens, num_topk;
        int block_m, open_gate_blocks;
        unsigned generation;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_plan_kernel<
        {}, {}, {}, {}
    >);
}};
)", args.num_ranks, args.num_experts, args.num_helper_slots, args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.workspace, args.topk_idx,
            static_cast<uint32_t>(args.num_tokens), static_cast<uint32_t>(args.num_topk),
            static_cast<uint32_t>(args.block_m), static_cast<uint32_t>(args.open_gate_blocks),
            static_cast<uint32_t>(args.generation)));
    }
};

class MegaMoEEPLBRemapRuntime final: public LaunchRuntime<MegaMoEEPLBRemapRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, num_helper_slots;
        uint32_t* workspace;
        int rank_idx;
        int64_t* topk_idx;
        int num_tokens, num_topk, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_remap_kernel<
        {}, {}, {}, {}
    >);
}};
)", args.num_ranks, args.num_experts, args.num_helper_slots, args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.workspace, static_cast<uint32_t>(args.rank_idx), args.topk_idx,
            static_cast<uint32_t>(args.num_tokens), static_cast<uint32_t>(args.num_topk),
            static_cast<uint32_t>(args.block_m)));
    }
};

class MegaMoEEPLBCopyWeightsRuntime final: public LaunchRuntime<MegaMoEEPLBCopyWeightsRuntime> {
public:
    struct Args {
        int num_ranks;
        layout::SymBuffer<> arena;
        eplb::CopyPlanes planes;
        const int2* copy_list;
        const int32_t* copy_count;
        int num_home_experts;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_copy_weights_kernel<
        {}, {}
    >);
}};
)", args.num_ranks, args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.arena, args.planes, args.copy_list, args.copy_count,
            static_cast<uint32_t>(args.num_home_experts)));
    }
};

} // namespace deep_gemm
