#pragma once

#include <torch/python.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM120RoutedMoEArgument {
    alignas(8) std::array<std::byte, 8> bytes{};

    template <typename T>
    explicit SM120RoutedMoEArgument(const T& value) {
        static_assert(sizeof(T) <= sizeof(bytes));
        std::memcpy(bytes.data(), &value, sizeof(T));
    }
};

static void launch_sm120_routed_moe_kernel(
    const KernelHandle& kernel,
    const LaunchConfigHandle& config,
    std::vector<SM120RoutedMoEArgument>& arguments) {
    std::vector<void*> pointers;
    pointers.reserve(arguments.size());
    for (auto& argument: arguments)
        pointers.push_back(argument.bytes.data());
    DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, pointers.data()));
}

class SM120FP8FP4RoutedMoERuntime final:
    public LaunchRuntime<SM120FP8FP4RoutedMoERuntime> {
public:
    using Argument = SM120RoutedMoEArgument;

    struct Args {
        std::vector<Argument> arguments;
        LaunchArgs launch_args;
        int world_size;
        bool unroll_k_loops;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.world_size == 4 or args.world_size == 8);
        return fmt::format(R"(
#define DG_SM120_ROUTED_MOE_WORLD_SIZE {}
#define DG_SM120_ROUTED_MOE_KERNEL sm120_fp8_fp4_routed_moe_ep{}_impl
#define DG_SM120_ROUTED_MOE_K_LOOP_UNROLL {}
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe.cuh>
)", args.world_size, args.world_size, args.unroll_k_loops ? 2 : 1);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        launch_sm120_routed_moe_kernel(kernel, config, args.arguments);
    }
};

class SM120FP8FP4RoutedMoEDrainRuntime final:
    public LaunchRuntime<SM120FP8FP4RoutedMoEDrainRuntime> {
public:
    using Argument = SM120FP8FP4RoutedMoERuntime::Argument;

    struct Args {
        std::vector<Argument> arguments;
        LaunchArgs launch_args;
        int world_size;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.world_size == 4 or args.world_size == 8);
        return fmt::format(R"(
#define DG_SM120_ROUTED_MOE_WORLD_SIZE {}
#define DG_SM120_ROUTED_MOE_DRAIN_KERNEL sm120_fp8_fp4_routed_moe_ep{}_drain_impl
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe_drain.cuh>
)", args.world_size, args.world_size);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        launch_sm120_routed_moe_kernel(kernel, config, args.arguments);
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

static std::shared_ptr<KernelRuntime> prepare_sm120_fp8_fp4_routed_moe_drain(
    int world_size) {
    SM120FP8FP4RoutedMoEDrainRuntime::Args compile_args{
        {}, LaunchArgs(1, 32), world_size};
    return compiler->build(
        fmt::format("sm120_fp8_fp4_routed_moe_ep{}_drain", world_size),
        SM120FP8FP4RoutedMoEDrainRuntime::generate(compile_args));
}

static std::shared_ptr<KernelRuntime> prepare_sm120_fp8_fp4_routed_moe(
    int world_size,
    int active_rows) {
    SM120FP8FP4RoutedMoERuntime::Args compile_args{
        {}, LaunchArgs(1, 1), world_size, world_size == 4 and active_rows == 2048};
    auto runtime = compiler->build(
        fmt::format(
            "sm120_fp8_fp4_routed_moe_ep{}_u{}",
            world_size,
            compile_args.unroll_k_loops ? 2 : 1),
        SM120FP8FP4RoutedMoERuntime::generate(compile_args));
    (void)prepare_sm120_fp8_fp4_routed_moe_drain(world_size);
    return runtime;
}

template <int WorldSize>
static void sm120_fp8_fp4_routed_moe(
    const pybind11::dict& arguments,
    int rank,
    int active_rows,
    std::uint32_t epoch,
    int grid_ctas,
    float activation_clamp,
    bool fast_math,
    bool drain_only = false) {
    static_assert(WorldSize == 4 or WorldSize == 8);
    using Shape = sm120_routed_moe::ShapeT<WorldSize>;
    using Communication = sm120_routed_moe::CommunicationLayoutT<WorldSize>;
    using Codec = sm120_routed_moe::ResultCodecLayoutT<WorldSize>;
    using Workspace = sm120_routed_moe::WorkspaceLayoutT<WorldSize>;

    DG_HOST_ASSERT(rank >= 0 and rank < Shape::kWorldSize);
    DG_HOST_ASSERT(active_rows >= 1 and active_rows <= Shape::kMaxRows);
    DG_HOST_ASSERT(grid_ctas >= Shape::kWorldSize and grid_ctas <= Shape::kMaxGridCTAs);
    DG_HOST_ASSERT(activation_clamp == Shape::kActivationClamp);
    DG_HOST_ASSERT(fast_math == Shape::kFastMath);

    const int device = at::cuda::current_device();
    if (drain_only) {
        std::vector<SM120FP8FP4RoutedMoEDrainRuntime::Argument> drain_arguments;
        drain_arguments.reserve(2);
        const auto communicator = reinterpret_cast<void*>(
            detail::require_integer(arguments, "gin_device_communicator"));
        drain_arguments.emplace_back(communicator);
        const auto ack_signal_base_scratch = detail::require_cuda_tensor(
            arguments,
            "ack_signal_base_scratch",
            device,
            torch::kUInt64,
            Shape::kWorldSize + 1,
            8);
        drain_arguments.emplace_back(ack_signal_base_scratch.data_ptr());
        SM120FP8FP4RoutedMoEDrainRuntime::Args drain_args{
            std::move(drain_arguments), LaunchArgs(1, 32), WorldSize};
        SM120FP8FP4RoutedMoEDrainRuntime::launch(
            prepare_sm120_fp8_fp4_routed_moe_drain(WorldSize), drain_args);
        return;
    }

    std::vector<SM120FP8FP4RoutedMoERuntime::Argument> packed;
    packed.reserve(112);

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

    constexpr std::int64_t kTensorMapBytes = 128;
    constexpr std::int64_t kPeerExperts = Shape::kExperts;
    constexpr std::int64_t kChunkEntries =
        Shape::kWorldSize * Communication::kDispatchChunks;
    constexpr std::int64_t kResultChunkEntries =
        Shape::kWorldSize * Codec::kMaxChunksPerPeer;
    constexpr std::int64_t kTilesPerTask = Shape::kHidden / Shape::kMmaTileN;

    append_tensor("W1_A", torch::kUInt8, kTensorMapBytes, 128, true);
    if constexpr (WorldSize == 4)
        append_tensor("W1_A64", torch::kUInt8, kTensorMapBytes, 128, true);
    for (const char* name: {"W1_B", "W1_SFA"})
        append_tensor(name, torch::kUInt8, kTensorMapBytes, 128, true);
    if constexpr (WorldSize == 4)
        append_tensor("W1_SFA64", torch::kUInt8, kTensorMapBytes, 128, true);
    for (const char* name: {"W1_SFB", "W1_D"})
        append_tensor(name, torch::kUInt8, kTensorMapBytes, 128, true);

    append_tensor("W2_A", torch::kUInt8, kTensorMapBytes, 128, true);
    if constexpr (WorldSize == 4)
        append_tensor("W2_A64", torch::kUInt8, kTensorMapBytes, 128, true);
    for (const char* name: {"W2_B", "W2_SFA"})
        append_tensor(name, torch::kUInt8, kTensorMapBytes, 128, true);
    if constexpr (WorldSize == 4)
        append_tensor("W2_SFA64", torch::kUInt8, kTensorMapBytes, 128, true);
    for (const char* name: {"W2_SFB", "W2_D"})
        append_tensor(name, torch::kUInt8, kTensorMapBytes, 128, true);

    append_tensor(
        "intermediate_fp8", torch::kUInt8,
        static_cast<std::int64_t>(Workspace::kPoolRows) * Shape::kIntermediate, 16);
    append_tensor(
        "intermediate_sfa_u8", torch::kUInt8,
        static_cast<std::int64_t>(Shape::kIntermediate / 128) *
            Workspace::kPoolRows * sizeof(std::uint32_t), 16);
    append_tensor("requant_groups_done", torch::kInt32, 1, 4);
    append_tensor("w2_warp_done", torch::kInt32,
                  static_cast<std::int64_t>(Workspace::kMaxTasks) * kTilesPerTask, 4);
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
    append_tensor("phase_timestamps", torch::kUInt64,
                  sm120_routed_moe::TraceLayout::kTimestampCount, 8);
    append_tensor("peer_phase_timestamps", torch::kUInt64, Shape::kWorldSize, 8);
    append_tensor("w2_task_counter", torch::kUInt32, Workspace::kMaxTasks, 4);
    append_tensor("w1_task_counter", torch::kUInt32, Workspace::kMaxTasks, 4);
    append_tensor("dispatch_chunk_scatter_counter", torch::kUInt32, kChunkEntries, 4);
    append_tensor("pull_chunk_arrived", torch::kUInt32, kChunkEntries, 4);
    append_tensor("result_owner_ready", torch::kUInt32, Shape::kWorldSize, 4);
    append_tensor("result_owner_progress", torch::kUInt32, Shape::kWorldSize, 4);
    append_tensor("pull_request_scratch", torch::kUInt64, 2 * kChunkEntries, 8);
    append_tensor("dispatch_chunk_targets", torch::kInt32, kChunkEntries, 4);
    append_tensor("c56_claim_cursor", torch::kInt32, 1, 4);
    append_tensor("combine_claim_cursor", torch::kInt32, 1, 4);
    append_tensor("c56_tile_mailbox", torch::kInt32, Workspace::kMailboxEntries, 4);
    append_tensor("task_gate_packed", torch::kInt32, Workspace::kMaxTasks, 4);
    append_tensor("result_chunk_total", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("result_chunk_tally", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("result_ovf_cursor", torch::kInt32, kResultChunkEntries, 4);
    append_tensor("signal_base_scratch", torch::kUInt64, Shape::kWorldSize, 8);
    append_tensor("dispatch_chunk_signal_base_scratch", torch::kUInt64, kChunkEntries, 8);
    append_tensor("result_signal_base_scratch", torch::kUInt64, 2 * Shape::kWorldSize, 8);
    append_tensor("ack_signal_base_scratch", torch::kUInt64, Shape::kWorldSize + 1, 8);
    append_tensor("final_output", torch::kBFloat16,
                  static_cast<std::int64_t>(active_rows) * Shape::kHidden, 16);
    append_tensor("pool_fp8_u32", torch::kInt32,
                  static_cast<std::int64_t>(Workspace::kPoolRows) * Shape::kHidden / 4, 16);
    append_tensor("pool_sf_u32", torch::kInt32,
                  static_cast<std::int64_t>(Shape::kHidden / 128) * Workspace::kPoolRows, 16);
    append_tensor("routing_weight_pool", torch::kFloat32, Workspace::kPoolRows, 4);
    for (const char* name: {"meta_source_rank", "meta_token", "meta_slot", "meta_result_index"})
        append_tensor(name, torch::kInt32, Workspace::kPoolRows, 4);
    append_tensor("expert_counts", torch::kInt32, Shape::kExpertsPerRank, 4);
    append_tensor("owner_expert_route_counts", torch::kInt32, kPeerExperts, 4);
    append_tensor("source_route_sum", torch::kInt32, Shape::kWorldSize, 4);
    for (const char* name: {
             "source_expert_counts", "expert_source_base", "expert_source_offsets",
             "source_expert_prefix"})
        append_tensor(name, torch::kInt32, kPeerExperts, 4);
    append_tensor("task_max_source", torch::kInt32, Workspace::kMaxTasks, 4);
    for (const char* name: {"source_record_counts", "source_route_counts", "source_active_rows"})
        append_tensor(name, torch::kInt32, Shape::kWorldSize, 4);
    append_tensor("expert_row_offsets", torch::kInt32, Shape::kExpertsPerRank, 4);
    append_tensor("expert_task_base", torch::kInt32, Shape::kExpertsPerRank, 4);
    append_tensor("expert_block_task", torch::kInt32,
                  static_cast<std::int64_t>(Shape::kExpertsPerRank) * Workspace::kMaxTasks, 4);
    append_tensor("task_source_slot_base", torch::kInt32,
                  static_cast<std::int64_t>(Workspace::kMaxTasks) * Shape::kWorldSize, 4);
    append_tensor("expert_scatter_offsets", torch::kInt32, Shape::kExpertsPerRank, 4);
    for (const char* name: {
             "task_expert", "task_source_rank", "task_owner_rank", "task_local_expert",
             "task_pool_row", "task_m_local", "task_valid_m", "task_rows_landed"})
        append_tensor(name, torch::kInt32, Workspace::kMaxTasks, 4);
    for (const char* name: {
             "total_valid_routes", "total_padded_rows", "total_m_tasks",
             "histogram_done", "prefix_done"})
        append_tensor(name, torch::kInt32, 1, 4);
    append_tensor("w1_warp_done", torch::kInt32,
                  static_cast<std::int64_t>(Workspace::kMaxTasks) * kTilesPerTask, 4);
    append_tensor("w1_tiles_completed", torch::kInt32, 1, 4);

    packed.emplace_back(rank);
    packed.emplace_back(Shape::kWorldSize);
    packed.emplace_back(active_rows);
    packed.emplace_back(epoch);
    append_handle("gin_device_communicator");

    for (const char* name: {
             "dispatch_header_out", "dispatch_header_out_window",
             "dispatch_payload_out", "dispatch_payload_out_window",
             "dispatch_header_inbox", "dispatch_header_inbox_window",
             "dispatch_payload_inbox", "dispatch_payload_inbox_window",
             "result_out", "result_out_window", "result_inbox", "result_inbox_window",
             "ack_out", "ack_out_window", "ack_inbox", "ack_inbox_window"}) {
        const std::string argument_name(name);
        if (argument_name.size() >= 7 and
            argument_name.compare(argument_name.size() - 7, 7, "_window") == 0) {
            append_handle(name);
        } else {
            const auto bytes = argument_name.find("header") != std::string::npos ?
                Communication::kHeaderWindowBytes :
                (argument_name.find("payload") != std::string::npos ?
                     Communication::kPayloadWindowBytes :
                     (argument_name.find("ack") != std::string::npos ?
                          Communication::kAckWindowBytes :
                          Communication::kResultWindowBytes));
            append_tensor(name, torch::kUInt8, bytes, 16, true);
        }
    }

    const auto runtime = prepare_sm120_fp8_fp4_routed_moe(WorldSize, active_rows);
    SM120FP8FP4RoutedMoERuntime::Args launch_args{
        std::move(packed),
        LaunchArgs(grid_ctas, Shape::kThreads, Shape::kDynamicSharedMemoryBytes, 1, false, true),
        WorldSize,
        WorldSize == 4 and active_rows == 2048};
    SM120FP8FP4RoutedMoERuntime::launch(runtime, launch_args);
}

} // namespace deep_gemm
