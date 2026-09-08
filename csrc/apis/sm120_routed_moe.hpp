#pragma once

#include <torch/python.h>

#include "../jit_kernels/heuristics/sm120_routed_moe.hpp"

namespace deep_gemm::mega {

static pybind11::dict get_sm120_routed_moe_layout() {
    using Communication = sm120_routed_moe::CommunicationLayout;
    using Codec = sm120_routed_moe::ResultCodecLayout;
    using Shape = sm120_routed_moe::Shape;
    using Trace = sm120_routed_moe::TraceLayout;
    using Workspace = sm120_routed_moe::WorkspaceLayout;

    pybind11::dict result;
    result["world_size"] = Shape::kWorldSize;
    result["num_experts"] = Shape::kExperts;
    result["experts_per_rank"] = Shape::kExpertsPerRank;
    result["num_topk"] = Shape::kTopK;
    result["hidden"] = Shape::kHidden;
    result["intermediate_hidden"] = Shape::kIntermediate;
    result["max_rows"] = Shape::kMaxRows;
    result["task_rows"] = Shape::kTaskRows;
    result["max_grid_ctas"] = Shape::kMaxGridCTAs;
    result["activation_clamp"] = Shape::kActivationClamp;
    result["fast_math"] = Shape::kFastMath;
    result["pool_rows"] = Workspace::kPoolRows;
    result["max_tasks"] = Workspace::kMaxTasks;
    result["mailbox_entries"] = Workspace::kMailboxEntries;
    result["phase_timestamp_count"] = Trace::kTimestampCount;
    result["codec_blocks_per_row"] = Codec::kBlocksPerRow;
    result["codec_encoded_row_bytes"] = Codec::kEncodedRowBytes;
    result["codec_raw_blocks_per_row"] = Codec::kRawBlocksPerRow;
    result["codec_raw_values_in_body"] = Codec::kRawValuesInBody;
    result["codec_raw_tail_bytes"] = Codec::kRawTailBytes;
    result["codec_max_chunks_per_peer"] = Codec::kMaxChunksPerPeer;
    result["result_route_pitch_bytes"] = Codec::kRoutePitchBytes;
    result["result_slot_bytes"] = Codec::kSlotBytes;
    result["dispatch_header_window_bytes"] = Communication::kHeaderWindowBytes;
    result["dispatch_header_slot_bytes"] = Communication::kHeaderSlotBytes;
    result["dispatch_record_bytes"] = Communication::kDispatchRecordBytes;
    result["dispatch_payload_window_bytes"] = Communication::kPayloadWindowBytes;
    result["result_window_bytes"] = Communication::kResultWindowBytes;
    result["ack_window_bytes"] = Communication::kAckWindowBytes;
    result["gin_context_count"] = Communication::kGinContextCount;
    result["dispatch_chunks"] = Communication::kDispatchChunks;
    result["gin_signal_count"] = Communication::kGinSignalCount;
    result["world_barrier_count"] = Communication::kWorldBarrierCount;
    return result;
}

} // namespace deep_gemm::mega

#ifdef DG_WITH_NCCL_GIN

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>
#include <c10/cuda/CUDAGuard.h>

#include "../jit_kernels/impls/sm120_fp8_fp4_routed_moe.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace deep_gemm::mega {

namespace detail {

[[noreturn]] static void throw_cuda(cudaError_t status, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status)
            << " (" << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw_cuda(status, operation);
}

[[noreturn]] static void throw_nccl(ncclResult_t status, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << ncclGetErrorString(status);
    throw std::runtime_error(message.str());
}

static void check_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess)
        throw_nccl(status, operation);
}

struct GinWindow {
    std::string name;
    std::size_t bytes = 0;
    void* pointer = nullptr;
    ncclWindow_t handle = nullptr;
    torch::Tensor tensor;
};

} // namespace detail

class SM120RoutedMoESession final {
public:
    SM120RoutedMoESession(
        std::uintptr_t communicator,
        const pybind11::object& communicator_owner,
        int expected_rank,
        int expected_world_size,
        int expected_device):
        communicator_(reinterpret_cast<ncclComm_t>(communicator)),
        rank_(expected_rank),
        world_size_(expected_world_size),
        device_(expected_device) {
        if (communicator == 0)
            throw std::invalid_argument("NCCL communicator cannot be null");
        if (communicator_owner.is_none())
            throw std::invalid_argument("NCCL communicator owner cannot be None");
        if (rank_ < 0 or world_size_ <= 0 or rank_ >= world_size_)
            throw std::invalid_argument("invalid rank or world size");
        if (world_size_ != sm120_routed_moe::Shape::kWorldSize)
            throw std::invalid_argument("SM120 routed MoE requires an eight-rank EP group");

        const c10::cuda::CUDAGuard device_guard(device_);
        int runtime_version = 0;
        detail::check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
        if (runtime_version != kRequiredNCCLVersion) {
            std::ostringstream message;
            message << "SM120 routed MoE requires NCCL " << kRequiredNCCLVersion
                    << ", observed " << runtime_version;
            throw std::runtime_error(message.str());
        }

        int current_device = -1;
        detail::check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
        if (current_device != device_)
            throw std::runtime_error("current CUDA device does not match the NCCL communicator");

        ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
        detail::check_nccl(
            ncclCommQueryProperties(communicator_, &properties),
            "ncclCommQueryProperties");
        if (properties.rank != rank_ or properties.nRanks != world_size_ or
            properties.cudaDev != device_)
            throw std::runtime_error("NCCL communicator identity mismatch");
        if (not properties.deviceApiSupport or properties.ginType == NCCL_GIN_TYPE_NONE)
            throw std::runtime_error("NCCL communicator does not support GIN Device API");
        gin_type_ = static_cast<int>(properties.ginType);

        try {
            using Communication = sm120_routed_moe::CommunicationLayout;
            const std::array<std::pair<const char*, std::int64_t>, 8> window_specs{{
                {"dispatch_header_out", Communication::kHeaderWindowBytes},
                {"dispatch_payload_out", Communication::kPayloadWindowBytes},
                {"dispatch_header_inbox", Communication::kHeaderWindowBytes},
                {"dispatch_payload_inbox", Communication::kPayloadWindowBytes},
                {"result_out", Communication::kResultWindowBytes},
                {"result_inbox", Communication::kResultWindowBytes},
                {"ack_out", Communication::kAckWindowBytes},
                {"ack_inbox", Communication::kAckWindowBytes},
            }};
            windows_.reserve(window_specs.size());
            for (const auto& [name, signed_bytes]: window_specs)
                allocate_window(name, signed_bytes);

            ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            requirements.ginContextCount = Communication::kGinContextCount;
            requirements.ginSignalCount = Communication::kGinSignalCount;
            requirements.worldGinBarrierCount = Communication::kWorldBarrierCount;
            requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
            requirements.ginStrongSignalsRequired = true;
            requirements.ginVaSignalsRequired = false;
            detail::check_nccl(
                ncclDevCommCreate(communicator_, &requirements, &host_device_communicator_),
                "ncclDevCommCreate");
            device_communicator_created_ = true;
            if (host_device_communicator_.ginContextCount == 0)
                throw std::runtime_error("NCCL did not grant a GIN context");

            detail::check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&device_communicator_),
                           sizeof(host_device_communicator_)),
                "cudaMalloc(ncclDevComm)");
            detail::check_cuda(
                cudaMemcpy(device_communicator_, &host_device_communicator_,
                           sizeof(host_device_communicator_), cudaMemcpyHostToDevice),
                "cudaMemcpy(ncclDevComm)");
        } catch (...) {
            release_noexcept();
            throw;
        }
    }

    SM120RoutedMoESession(const SM120RoutedMoESession&) = delete;
    SM120RoutedMoESession& operator=(const SM120RoutedMoESession&) = delete;

    ~SM120RoutedMoESession() {
        release_noexcept();
    }

    torch::Tensor window_tensor(const std::string& name) const {
        require_open();
        return window(name).tensor;
    }

    std::uintptr_t window_handle(const std::string& name) const {
        require_open();
        return reinterpret_cast<std::uintptr_t>(window(name).handle);
    }

    std::uintptr_t device_communicator() const {
        require_open();
        return reinterpret_cast<std::uintptr_t>(device_communicator_);
    }

    pybind11::dict properties() const {
        require_open();
        pybind11::dict result;
        result["rank"] = rank_;
        result["world_size"] = world_size_;
        result["device"] = device_;
        result["nccl_version"] = kRequiredNCCLVersion;
        result["gin_type"] = gin_type_;
        result["gin_connection_count"] = host_device_communicator_.ginConnectionCount;
        result["gin_context_count"] = host_device_communicator_.ginContextCount;
        result["gin_signal_count"] = host_device_communicator_.ginSignalCount;
        result["gin_connections_railed"] = host_device_communicator_.ginConnectionsRailed;
        result["gin_contexts_railed"] = host_device_communicator_.ginContextsRailed;
        pybind11::list device_types;
        for (int connection = 0;
             connection < host_device_communicator_.ginConnectionCount;
             ++connection)
            device_types.append(host_device_communicator_.ginNetDeviceTypes[connection]);
        result["gin_net_device_types"] = std::move(device_types);
        result["window_count"] = windows_.size();
        return result;
    }

    bool closed() const {
        return closed_;
    }

    void launch(
        const pybind11::dict& arguments,
        int active_rows,
        std::uint32_t epoch,
        int grid_ctas,
        float activation_clamp,
        bool fast_math,
        bool enable_phase_trace,
        bool drain_only = false) const {
        require_open();

        int current_device = -1;
        detail::check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
        if (current_device != device_)
            throw std::runtime_error("current CUDA device does not match the SM120 routed MoE session");

        constexpr std::array<const char*, 18> kSessionOwnedArguments{{
            "rank",
            "world_size",
            "device",
            "gin_device_communicator",
            "dispatch_header_out",
            "dispatch_header_out_window",
            "dispatch_payload_out",
            "dispatch_payload_out_window",
            "dispatch_header_inbox",
            "dispatch_header_inbox_window",
            "dispatch_payload_inbox",
            "dispatch_payload_inbox_window",
            "result_out",
            "result_out_window",
            "result_inbox",
            "result_inbox_window",
            "ack_out_window",
            "ack_inbox_window",
        }};
        for (const char* name: kSessionOwnedArguments) {
            if (arguments.contains(name))
                throw std::invalid_argument(
                    std::string("SM120 routed MoE argument is owned by the session: ") + name);
        }

        pybind11::dict launch_arguments;
        for (const auto& item: arguments)
            launch_arguments[item.first] = item.second;
        launch_arguments["gin_device_communicator"] =
            reinterpret_cast<std::uintptr_t>(device_communicator_);
        for (const char* name: {
                 "dispatch_header_out",
                 "dispatch_payload_out",
                 "dispatch_header_inbox",
                 "dispatch_payload_inbox",
                 "result_out",
                 "result_inbox"}) {
            const auto& allocation = window(name);
            launch_arguments[pybind11::str(name)] = allocation.tensor;
            launch_arguments[pybind11::str(std::string(name) + "_window")] =
                reinterpret_cast<std::uintptr_t>(allocation.handle);
        }
        launch_arguments["ack_out_window"] =
            reinterpret_cast<std::uintptr_t>(window("ack_out").handle);
        launch_arguments["ack_inbox_window"] =
            reinterpret_cast<std::uintptr_t>(window("ack_inbox").handle);

        deep_gemm::sm120_fp8_fp4_routed_moe(
            launch_arguments,
            rank_,
            world_size_,
            active_rows,
            epoch,
            grid_ctas,
            activation_clamp,
            fast_math,
            enable_phase_trace,
            drain_only);
    }

    void quiesce(
        const pybind11::dict& arguments,
        int active_rows,
        int grid_ctas) const {
        launch(
            arguments,
            active_rows,
            0,
            grid_ctas,
            sm120_routed_moe::Shape::kActivationClamp,
            sm120_routed_moe::Shape::kFastMath,
            false,
            true);
    }

    void close() {
        if (closed_)
            return;
        const c10::cuda::CUDAGuard device_guard(device_);
        std::vector<std::string> failures;
        auto record_cuda = [&failures](cudaError_t status, const std::string& operation) {
            if (status != cudaSuccess) {
                std::ostringstream message;
                message << operation << " failed: " << cudaGetErrorName(status)
                        << " (" << cudaGetErrorString(status) << ')';
                failures.push_back(message.str());
            }
            return status == cudaSuccess;
        };
        auto record_nccl = [&failures](ncclResult_t status, const std::string& operation) {
            if (status != ncclSuccess) {
                std::ostringstream message;
                message << operation << " failed: " << ncclGetErrorString(status);
                failures.push_back(message.str());
            }
            return status == ncclSuccess;
        };

        record_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (device_communicator_ != nullptr) {
            if (record_cuda(cudaFree(device_communicator_), "cudaFree(ncclDevComm)"))
                device_communicator_ = nullptr;
        }
        if (device_communicator_created_) {
            if (record_nccl(
                    ncclDevCommDestroy(communicator_, &host_device_communicator_),
                    "ncclDevCommDestroy"))
                device_communicator_created_ = false;
        }
        for (auto iterator = windows_.rbegin(); iterator != windows_.rend(); ++iterator) {
            if (iterator->handle != nullptr and record_nccl(
                    ncclCommWindowDeregister(communicator_, iterator->handle),
                    "ncclCommWindowDeregister(" + iterator->name + ")")) {
                iterator->handle = nullptr;
                iterator->pointer = nullptr;
                iterator->tensor = torch::Tensor();
            }
        }
        closed_ = device_communicator_ == nullptr and
                  not device_communicator_created_ and
                  std::all_of(
                      windows_.begin(), windows_.end(),
                      [](const detail::GinWindow& window) {
                          return window.handle == nullptr;
                      });
        if (not failures.empty()) {
            std::ostringstream message;
            message << "SM120 routed MoE session cleanup failed";
            for (const auto& failure: failures)
                message << "; " << failure;
            throw std::runtime_error(message.str());
        }
    }

private:
    static constexpr int kRequiredNCCLVersion = 23007;

    void allocate_window(const std::string& name, std::int64_t signed_bytes) {
        if (name.empty() or signed_bytes <= 0)
            throw std::invalid_argument("GIN window name and byte count must be valid");
        if (window_indices_.find(name) != window_indices_.end())
            throw std::invalid_argument("duplicate GIN window name: " + name);

        detail::GinWindow allocation;
        allocation.name = name;
        allocation.bytes = static_cast<std::size_t>(signed_bytes);
        detail::check_nccl(
            ncclMemAlloc(&allocation.pointer, allocation.bytes),
            "ncclMemAlloc");
        try {
            detail::check_nccl(
                ncclCommWindowRegister(
                    communicator_, allocation.pointer, allocation.bytes,
                    &allocation.handle, NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister");
        } catch (...) {
            ncclMemFree(allocation.pointer);
            throw;
        }
        try {
            auto* pointer = allocation.pointer;
            allocation.tensor = torch::from_blob(
                allocation.pointer,
                {static_cast<std::int64_t>(allocation.bytes)},
                [pointer](void*) { ncclMemFree(pointer); },
                torch::TensorOptions().dtype(torch::kUInt8).device(
                    torch::Device(torch::kCUDA, device_)));
            window_indices_.emplace(name, windows_.size());
            windows_.push_back(std::move(allocation));
        } catch (...) {
            ncclCommWindowDeregister(communicator_, allocation.handle);
            if (allocation.tensor.defined())
                allocation.tensor = torch::Tensor();
            else
                ncclMemFree(allocation.pointer);
            throw;
        }
    }

    void release_noexcept() noexcept {
        if (closed_)
            return;
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_);
        cudaDeviceSynchronize();
        if (device_communicator_ != nullptr) {
            cudaFree(device_communicator_);
            device_communicator_ = nullptr;
        }
        if (device_communicator_created_) {
            ncclDevCommDestroy(communicator_, &host_device_communicator_);
            device_communicator_created_ = false;
        }
        for (auto iterator = windows_.rbegin(); iterator != windows_.rend(); ++iterator) {
            if (iterator->handle != nullptr)
                ncclCommWindowDeregister(communicator_, iterator->handle);
            iterator->handle = nullptr;
            iterator->pointer = nullptr;
            iterator->tensor = torch::Tensor();
        }
        if (previous_device >= 0)
            cudaSetDevice(previous_device);
        closed_ = true;
    }

    const detail::GinWindow& window(const std::string& name) const {
        const auto iterator = window_indices_.find(name);
        if (iterator == window_indices_.end())
            throw std::out_of_range("unknown GIN window: " + name);
        return windows_[iterator->second];
    }

    void require_open() const {
        if (closed_)
            throw std::runtime_error("GIN session is closed");
    }

    ncclComm_t communicator_ = nullptr;
    int rank_ = -1;
    int world_size_ = 0;
    int device_ = -1;
    int gin_type_ = 0;
    ncclDevComm host_device_communicator_{};
    ncclDevComm* device_communicator_ = nullptr;
    bool device_communicator_created_ = false;
    bool closed_ = false;
    std::vector<detail::GinWindow> windows_;
    std::unordered_map<std::string, std::size_t> window_indices_;
};

static void launch_sm120_fp8_fp4_routed_moe(
    SM120RoutedMoESession& session,
    const pybind11::dict& arguments,
    int active_rows,
    std::uint32_t epoch,
    int grid_ctas,
    float activation_clamp,
    bool fast_math,
    bool enable_phase_trace) {
    session.launch(
        arguments,
        active_rows,
        epoch,
        grid_ctas,
        activation_clamp,
        fast_math,
        enable_phase_trace);
}

static CUtensorMapDataType parse_sm120_tma_data_type(const std::string& name) {
    if (name == "uint8")
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (name == "fp4")
        return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
    if (name == "int32")
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    if (name == "bfloat16")
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    throw std::invalid_argument("unsupported SM120 tensor-map data type: " + name);
}

static CUtensorMapSwizzle parse_sm120_tma_swizzle(int bytes) {
    switch (bytes) {
        case 0: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: throw std::invalid_argument("unsupported SM120 tensor-map swizzle");
    }
}

static torch::Tensor make_sm120_tma_2d(
    const torch::Tensor& tensor,
    const std::string& data_type,
    std::uint64_t inner,
    std::uint64_t outer,
    std::uint64_t outer_stride_bytes,
    std::uint32_t box_inner,
    std::uint32_t box_outer,
    int swizzle_bytes) {
    const auto tensor_map_data_type = parse_sm120_tma_data_type(data_type);
    const auto tensor_map_swizzle = parse_sm120_tma_swizzle(swizzle_bytes);
    const bool is_fp4 = tensor_map_data_type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;

    bool dtype_matches = false;
    std::uint64_t storage_bytes_per_element_numerator = 0;
    std::uint64_t storage_bytes_per_element_denominator = 1;
    switch (tensor_map_data_type) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:
            dtype_matches = tensor.scalar_type() == torch::kUInt8 or
                            tensor.scalar_type() == torch::kFloat8_e4m3fn;
            storage_bytes_per_element_numerator = 1;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B:
            dtype_matches = tensor.scalar_type() == torch::kInt8;
            storage_bytes_per_element_numerator = 1;
            storage_bytes_per_element_denominator = 2;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_INT32:
            dtype_matches = tensor.scalar_type() == torch::kInt32;
            storage_bytes_per_element_numerator = 4;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
            dtype_matches = tensor.scalar_type() == torch::kBFloat16;
            storage_bytes_per_element_numerator = 2;
            break;
        default:
            throw std::invalid_argument("unsupported SM120 tensor-map data type");
    }
    if (not dtype_matches)
        throw std::invalid_argument("SM120 tensor-map data type does not match tensor dtype");

    constexpr std::uint64_t kMaxDimension = std::uint64_t{1} << 32;
    constexpr std::uint64_t kMaxStrideBytes = std::uint64_t{1} << 40;
    constexpr std::uint64_t kMaxUint64 = ~std::uint64_t{0};
    if (inner == 0 or outer == 0 or inner > kMaxDimension or outer > kMaxDimension)
        throw std::invalid_argument("SM120 tensor-map dimensions must be in [1, 2^32]");
    if (box_inner == 0 or box_outer == 0 or box_inner > 256 or box_outer > 256)
        throw std::invalid_argument("SM120 tensor-map box dimensions must be in [1, 256]");
    if (box_inner > inner or box_outer > outer)
        throw std::invalid_argument("SM120 tensor-map box dimensions exceed tensor dimensions");
    if (outer_stride_bytes == 0 or outer_stride_bytes >= kMaxStrideBytes or
        outer_stride_bytes % 16 != 0)
        throw std::invalid_argument(
            "SM120 tensor-map outer stride must be a positive multiple of 16 below 2^40");

    if (is_fp4) {
        if (inner % 128 != 0 or box_inner != 128 or outer_stride_bytes % 32 != 0)
            throw std::invalid_argument(
                "SM120 FP4 tensor maps require inner multiple of 128, box_inner 128, and stride multiple of 32");
        if (swizzle_bytes != 0 and swizzle_bytes != 128)
            throw std::invalid_argument("SM120 FP4 tensor maps support only none or 128-byte swizzle");
    }

    const auto storage_bytes = [&](std::uint64_t elements) {
        if (elements > kMaxUint64 / storage_bytes_per_element_numerator)
            throw std::invalid_argument("SM120 tensor-map byte extent overflows uint64");
        const auto scaled = elements * storage_bytes_per_element_numerator;
        return (scaled + storage_bytes_per_element_denominator - 1) /
               storage_bytes_per_element_denominator;
    };
    const auto row_bytes = storage_bytes(inner);
    const auto box_row_bytes = storage_bytes(box_inner);
    if (row_bytes > outer_stride_bytes)
        throw std::invalid_argument("SM120 tensor-map outer stride is smaller than one row");
    if (not is_fp4 and box_row_bytes % 16 != 0)
        throw std::invalid_argument("SM120 tensor-map box inner byte size must be a multiple of 16");
    if (swizzle_bytes != 0 and box_row_bytes > static_cast<std::uint64_t>(swizzle_bytes))
        throw std::invalid_argument("SM120 tensor-map box inner byte size exceeds swizzle width");
    if (outer - 1 > (kMaxUint64 - row_bytes) / outer_stride_bytes)
        throw std::invalid_argument("SM120 tensor-map byte extent overflows uint64");
    // A 2-D map can reach the final row start plus one contiguous logical row.
    // Bounding that extent by this contiguous view also bounds it by its storage.
    const auto required_bytes = (outer - 1) * outer_stride_bytes + row_bytes;

    const auto tensor_numel = static_cast<std::uint64_t>(tensor.numel());
    const auto tensor_element_size = static_cast<std::uint64_t>(tensor.element_size());
    if (tensor_numel > kMaxUint64 / tensor_element_size)
        throw std::invalid_argument("SM120 tensor byte size overflows uint64");
    const auto tensor_bytes = tensor_numel * tensor_element_size;
    if (required_bytes > tensor_bytes)
        throw std::invalid_argument("SM120 tensor-map extent exceeds tensor storage");
    if (not tensor.is_contiguous())
        throw std::invalid_argument("SM120 tensor-map tensor must be contiguous");

    const std::uintptr_t base_address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    const std::uintptr_t required_alignment = is_fp4 ? 32 : 16;
    if (base_address == 0 or base_address % required_alignment != 0)
        throw std::invalid_argument("SM120 tensor-map base pointer is insufficiently aligned");
    if (not tensor.is_cuda())
        throw std::invalid_argument("SM120 tensor-map tensor must be CUDA");

    const c10::cuda::CUDAGuard device_guard(tensor.device());
    const cuuint64_t dimensions[2] = {inner, outer};
    const cuuint64_t strides[1] = {outer_stride_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t element_strides[2] = {1, 1};
    CUtensorMap map{};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &map,
        tensor_map_data_type,
        2,
        tensor.data_ptr(),
        dimensions,
        strides,
        box,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        tensor_map_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    auto carrier = torch::empty(
        {static_cast<std::int64_t>(sizeof(CUtensorMap))},
        torch::TensorOptions().dtype(torch::kUInt8).device(tensor.device()));
    detail::check_cuda(
        cudaMemcpy(carrier.data_ptr(), &map, sizeof(map), cudaMemcpyHostToDevice),
        "cudaMemcpy(CUtensorMap)");
    return carrier;
}

static void register_sm120_routed_moe_apis(pybind11::module& module) {
    module.def("has_sm120_routed_moe", []() { return true; });
    module.def("get_sm120_routed_moe_layout", &get_sm120_routed_moe_layout);
    module.def(
        "prepare_sm120_fp8_fp4_routed_moe",
        [](bool enable_phase_trace) {
            (void)deep_gemm::prepare_sm120_fp8_fp4_routed_moe(enable_phase_trace);
        },
        pybind11::arg("enable_phase_trace") = false);
    pybind11::class_<SM120RoutedMoESession>(module, "SM120RoutedMoESession")
        .def(pybind11::init<
             std::uintptr_t, const pybind11::object&, int, int, int>(),
             pybind11::keep_alive<1, 3>())
        .def("window_tensor", &SM120RoutedMoESession::window_tensor)
        .def("window_handle", &SM120RoutedMoESession::window_handle)
        .def("device_communicator", &SM120RoutedMoESession::device_communicator)
        .def("properties", &SM120RoutedMoESession::properties)
        .def(
            "quiesce",
            &SM120RoutedMoESession::quiesce,
            pybind11::arg("arguments"),
            pybind11::arg("active_rows"),
            pybind11::arg("grid_ctas"))
        .def("close", &SM120RoutedMoESession::close)
        .def_property_readonly("closed", &SM120RoutedMoESession::closed);
    module.def(
        "can_use_sm120_routed_moe_fast_path",
        &can_use_sm120_routed_moe_fast_path,
        pybind11::arg("num_ranks"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_topk"),
        pybind11::arg("hidden"),
        pybind11::arg("intermediate_hidden"),
        pybind11::arg("num_external_shared_experts"),
        pybind11::arg("num_tokens"),
        pybind11::arg("activation_clamp"),
        pybind11::arg("fast_math"));
    module.def(
        "sm120_fp8_fp4_routed_moe",
        &launch_sm120_fp8_fp4_routed_moe,
        pybind11::arg("session"),
        pybind11::arg("arguments"),
        pybind11::arg("active_rows"),
        pybind11::arg("epoch"),
        pybind11::arg("grid_ctas"),
        pybind11::arg("activation_clamp"),
        pybind11::arg("fast_math"),
        pybind11::arg("enable_phase_trace") = false);
    module.def(
        "make_sm120_tma_2d",
        &make_sm120_tma_2d,
        pybind11::arg("tensor"),
        pybind11::arg("data_type"),
        pybind11::arg("inner"),
        pybind11::arg("outer"),
        pybind11::arg("outer_stride_bytes"),
        pybind11::arg("box_inner"),
        pybind11::arg("box_outer"),
        pybind11::arg("swizzle_bytes"));
}

} // namespace deep_gemm::mega

#else

namespace deep_gemm::mega {

static void register_sm120_routed_moe_apis(pybind11::module& module) {
    module.def("has_sm120_routed_moe", []() { return false; });
    module.def("get_sm120_routed_moe_layout", &get_sm120_routed_moe_layout);
    module.def(
        "can_use_sm120_routed_moe_fast_path",
        [](int, int, int, int, int, int, int, float, bool) { return false; },
        pybind11::arg("num_ranks"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_topk"),
        pybind11::arg("hidden"),
        pybind11::arg("intermediate_hidden"),
        pybind11::arg("num_external_shared_experts"),
        pybind11::arg("num_tokens"),
        pybind11::arg("activation_clamp"),
        pybind11::arg("fast_math"));
}

} // namespace deep_gemm::mega

#endif
