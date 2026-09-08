#pragma once

#include <torch/python.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

class SM120FP8FP4RoutedMoERuntime final:
    public LaunchRuntime<SM120FP8FP4RoutedMoERuntime> {
public:
    struct Argument {
        alignas(8) std::array<std::byte, 8> bytes{};

        template <typename T>
        explicit Argument(const T& value) {
            static_assert(sizeof(T) <= sizeof(bytes));
            std::memcpy(bytes.data(), &value, sizeof(T));
        }
    };

    struct Args {
        std::vector<Argument> arguments;
        LaunchArgs launch_args;
        bool enable_phase_trace;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define DG_SM120_ROUTED_MOE_ENABLE_PHASE_TRACE {}
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe.cuh>
)", args.enable_phase_trace ? 1 : 0);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        std::vector<void*> pointers;
        pointers.reserve(args.arguments.size());
        for (auto& argument: args.arguments)
            pointers.push_back(argument.bytes.data());
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, pointers.data()));
    }
};

namespace detail {

static torch::Tensor require_cuda_tensor(
    const pybind11::dict& arguments,
    const char* name,
    int device,
    at::ScalarType scalar_type,
    std::int64_t minimum_numel,
    std::size_t alignment = 1,
    bool exact_numel = false) {
    if (not arguments.contains(name))
        throw std::invalid_argument(std::string("missing SM120 routed MoE tensor: ") + name);
    auto tensor = pybind11::cast<torch::Tensor>(arguments[name]);
    if (not tensor.is_cuda() or not tensor.is_contiguous() or tensor.get_device() != device)
        throw std::invalid_argument(
            std::string("SM120 routed MoE tensor must be contiguous on the current CUDA device: ") +
            name);
    if (tensor.scalar_type() != scalar_type)
        throw std::invalid_argument(
            std::string("SM120 routed MoE tensor has an invalid dtype: ") + name);
    if ((exact_numel and tensor.numel() != minimum_numel) or
        (not exact_numel and tensor.numel() < minimum_numel))
        throw std::invalid_argument(
            std::string("SM120 routed MoE tensor is too small: ") + name);
    if (alignment == 0 or
        reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % alignment != 0)
        throw std::invalid_argument(
            std::string("SM120 routed MoE tensor has insufficient alignment: ") + name);
    return tensor;
}

static std::int64_t require_integer(const pybind11::dict& arguments, const char* name) {
    if (not arguments.contains(name))
        DG_HOST_UNREACHABLE(std::string("missing SM120 routed MoE argument: ") + name);
    return pybind11::cast<std::int64_t>(arguments[name]);
}

} // namespace detail

static void sm120_fp8_fp4_routed_moe(
    const pybind11::dict& arguments,
    int rank,
    int world_size,
    int active_rows,
    std::uint32_t epoch,
    int grid_ctas,
    float activation_clamp,
    bool fast_math,
    bool enable_phase_trace,
    bool drain_only = false) {
    using Shape = sm120_routed_moe::Shape;
    constexpr int kThreads = Shape::kThreads;
    constexpr int kDynamicSharedMemory = Shape::kDynamicSharedMemoryBytes;

    DG_HOST_ASSERT(world_size == Shape::kWorldSize);
    DG_HOST_ASSERT(rank >= 0 and rank < world_size);
    DG_HOST_ASSERT(active_rows >= 1 and active_rows <= Shape::kMaxRows);
    DG_HOST_ASSERT(grid_ctas >= world_size and grid_ctas <= Shape::kMaxGridCTAs);
    DG_HOST_ASSERT(activation_clamp == sm120_routed_moe::Shape::kActivationClamp);
    DG_HOST_ASSERT(fast_math == sm120_routed_moe::Shape::kFastMath);

    const int device = at::cuda::current_device();
    std::vector<SM120FP8FP4RoutedMoERuntime::Argument> packed;
    packed.reserve(91);

    const auto append_tensor = [&](
        const char* name,
        at::ScalarType scalar_type,
        std::int64_t minimum_numel,
        std::size_t alignment = 1,
        bool exact_numel = false) {
        const auto tensor = detail::require_cuda_tensor(
            arguments, name, device, scalar_type, minimum_numel, alignment, exact_numel);
        packed.emplace_back(tensor.data_ptr());
    };
    const auto append_handle = [&](const char* name) {
        const auto value = reinterpret_cast<void*>(detail::require_integer(arguments, name));
        packed.emplace_back(value);
    };

    using Communication = sm120_routed_moe::CommunicationLayout;
    using Codec = sm120_routed_moe::ResultCodecLayout;
    using Workspace = sm120_routed_moe::WorkspaceLayout;
    constexpr std::int64_t kTmaDescriptorBytes = 128;
    constexpr std::int64_t kExpertsPerRank = Shape::kExpertsPerRank;
    constexpr std::int64_t kPeerExperts = Shape::kWorldSize * kExpertsPerRank;
    constexpr std::int64_t kChunkEntries =
        Shape::kWorldSize * Communication::kDispatchChunks;
    constexpr std::int64_t kResultChunkEntries =
        Shape::kWorldSize * Codec::kMaxChunksPerPeer;
    constexpr std::int64_t kW1TilesPerTask =
        (Shape::kIntermediate * 2) / Shape::kMmaTileN;
    constexpr std::int64_t kW2TilesPerTask = Shape::kHidden / Shape::kMmaTileN;

    for (const char* name: {
             "W1_A", "W1_B", "W1_SFA", "W1_SFB", "W1_D",
             "W2_A", "W2_B", "W2_SFA", "W2_SFB", "W2_D"})
        append_tensor(name, torch::kUInt8, kTmaDescriptorBytes, 128, true);

    append_tensor(
        "intermediate_fp8", torch::kUInt8,
        static_cast<std::int64_t>(Workspace::kPoolRows) * Shape::kIntermediate, 16);
    append_tensor(
        "intermediate_sfa_u8", torch::kUInt8,
        static_cast<std::int64_t>(Shape::kIntermediate / 128) *
            Workspace::kPoolRows * sizeof(std::uint32_t),
        16);
    append_tensor("requant_groups_done", torch::kInt32, 1, 4);
    append_tensor(
        "w2_warp_done", torch::kInt32,
        static_cast<std::int64_t>(Workspace::kMaxTasks) * kW2TilesPerTask, 4);
    append_tensor("w2_tiles_completed", torch::kInt32, 1, 4);
    append_tensor("topk_idx_i32", torch::kInt32,
                  static_cast<std::int64_t>(active_rows) * Shape::kTopK * 2, 8);
    append_tensor("topk_weights", torch::kFloat32,
                  static_cast<std::int64_t>(active_rows) * Shape::kTopK, 4);
    append_tensor("x_fp8_i32", torch::kInt32,
                  static_cast<std::int64_t>(active_rows) * Shape::kHidden / 4, 16);
    append_tensor("x_sf_i32", torch::kInt32,
                  static_cast<std::int64_t>(active_rows) * Shape::kHidden / 128, 16);
    append_tensor("owner_record_counts", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("owner_route_counts", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("owner_minexp_record_base", torch::kInt32, kPeerExperts, 4);
    append_tensor("owner_minexp_record_cursor", torch::kInt32, kPeerExperts, 4);
    append_tensor("sorted_record_token", torch::kInt32,
                  static_cast<std::int64_t>(Shape::kWorldSize) * Shape::kMaxRows, 4);
    append_tensor("sorted_record_route_base", torch::kInt32,
                  static_cast<std::int64_t>(Shape::kWorldSize) * Shape::kMaxRows, 4);
    append_tensor("route_result_index", torch::kInt32,
                  static_cast<std::int64_t>(active_rows) * Shape::kTopK, 4);
    append_tensor("protocol_error", torch::kInt32, 1, 4);
    if (enable_phase_trace) {
        append_tensor(
            "phase_timestamps", torch::kUInt64,
            sm120_routed_moe::TraceLayout::kTimestampCount, 8);
        append_tensor("peer_phase_timestamps", torch::kUInt64, Shape::kWorldSize, 8);
    } else {
        packed.emplace_back(static_cast<void*>(nullptr));
        packed.emplace_back(static_cast<void*>(nullptr));
    }
    append_tensor("w2_task_counter", torch::kUInt32, Workspace::kMaxTasks, 4);
    append_tensor("w1_task_counter", torch::kUInt32, Workspace::kMaxTasks, 4);
    append_tensor("dispatch_chunk_scatter_counter", torch::kUInt32, kChunkEntries, 4);
    append_tensor("pull_chunk_arrived", torch::kUInt32, kChunkEntries, 4);
    append_tensor("result_owner_progress", torch::kUInt32, Shape::kWorldSize, 4);
    append_tensor("dispatch_chunk_targets", torch::kInt32, kChunkEntries, 4);
    append_tensor("pipeline_claim_cursor", torch::kInt32, 1, 4);
    append_tensor("combine_claim_cursor", torch::kInt32, 1, 4);
    append_tensor(
        "pipeline_tile_mailbox", torch::kInt32, Workspace::kMailboxEntries, 4);
    append_tensor("task_gate_packed", torch::kInt32, Workspace::kMaxTasks, 4);
    append_tensor("result_chunk_total", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("result_chunk_tally", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("result_ovf_cursor", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("signal_base_scratch", torch::kUInt64, Shape::kWorldSize, 8);
    append_tensor("dispatch_chunk_signal_base_scratch", torch::kUInt64, kChunkEntries, 8);
    append_tensor("result_signal_base_scratch", torch::kUInt64, Shape::kWorldSize, 8);
    append_tensor("ack_signal_base_scratch", torch::kUInt64, Shape::kWorldSize + 1, 8);
    append_tensor("final_output", torch::kBFloat16,
                  static_cast<std::int64_t>(active_rows) * Shape::kHidden, 16);
    append_tensor("pool_fp8_u32", torch::kInt32,
                  static_cast<std::int64_t>(Workspace::kPoolRows) * Shape::kHidden / 4, 16);
    append_tensor("pool_sf_u32", torch::kInt32,
                  static_cast<std::int64_t>(Shape::kHidden / 128) * Workspace::kPoolRows, 16);
    append_tensor("routing_weight_pool", torch::kFloat32, Workspace::kPoolRows, 4);
    append_tensor("meta_source_rank", torch::kInt32, Workspace::kPoolRows, 4);
    append_tensor("meta_result_index", torch::kInt32, Workspace::kPoolRows, 4);
    append_tensor("expert_counts", torch::kInt32, kExpertsPerRank, 4);
    append_tensor("owner_expert_route_counts", torch::kInt32, kPeerExperts, 4);
    append_tensor("source_route_sum", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("source_expert_counts", torch::kInt32, kPeerExperts, 4);
    append_tensor("expert_source_base", torch::kInt32, kPeerExperts, 4);
    append_tensor("expert_source_offsets", torch::kInt32, kPeerExperts, 4);
    append_tensor("source_expert_prefix", torch::kInt32, kPeerExperts, 4);
    append_tensor("source_record_counts", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("source_route_counts", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("source_active_rows", torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("expert_row_offsets", torch::kInt32, kExpertsPerRank, 4);
    for (const char* name: {
             "task_local_expert", "task_pool_row"})
        append_tensor(name, torch::kInt32, Workspace::kMaxTasks, 4);
    for (const char* name: {
             "total_valid_routes", "total_m_tasks", "histogram_done", "prefix_done"})
        append_tensor(name, torch::kInt32, 1, 4);
    append_tensor(
        "w1_warp_done", torch::kInt32,
        static_cast<std::int64_t>(Workspace::kMaxTasks) * kW1TilesPerTask, 4);
    append_tensor("w1_tiles_completed", torch::kInt32, 1, 4);

    packed.emplace_back(rank);
    packed.emplace_back(world_size);
    packed.emplace_back(active_rows);
    packed.emplace_back(epoch);
    packed.emplace_back(drain_only);
    append_handle("gin_device_communicator");

    for (const char* name: {
             "dispatch_header_out", "dispatch_header_out_window",
             "dispatch_payload_out", "dispatch_payload_out_window",
             "dispatch_header_inbox", "dispatch_header_inbox_window",
             "dispatch_payload_inbox", "dispatch_payload_inbox_window",
             "result_out", "result_out_window", "result_inbox", "result_inbox_window"}) {
        const std::string argument_name(name);
        if (argument_name.size() >= 7 and
            argument_name.compare(argument_name.size() - 7, 7, "_window") == 0)
            append_handle(name);
        else
            append_tensor(
                name, torch::kUInt8,
                argument_name.find("header") != std::string::npos ?
                    Communication::kHeaderWindowBytes :
                    (argument_name.find("payload") != std::string::npos ?
                         Communication::kPayloadWindowBytes :
                         Communication::kResultWindowBytes),
                16, true);
    }
    append_handle("ack_out_window");
    append_handle("ack_inbox_window");

    SM120FP8FP4RoutedMoERuntime::Args compile_args{
        {}, LaunchArgs(grid_ctas, kThreads), enable_phase_trace};
    const auto runtime = compiler->build(
        "sm120_fp8_fp4_routed_moe",
        SM120FP8FP4RoutedMoERuntime::generate(compile_args));
    SM120FP8FP4RoutedMoERuntime::Args launch_args{
        std::move(packed),
        LaunchArgs(grid_ctas, kThreads, kDynamicSharedMemory, 1, false, true),
        enable_phase_trace};
    SM120FP8FP4RoutedMoERuntime::launch(runtime, launch_args);
}

} // namespace deep_gemm
