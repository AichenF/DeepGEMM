/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

// Shared fixtures, NCCL GIN setup, buffers, and independent references used by
// the SM120 MegaMoE correctness and performance hosts in this directory.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define LOOM_USE_EXTERNAL_CUDA_TYPES 1
#ifndef CAKE_GENERATED_SM120_PRODUCTION_SOURCE
#define CAKE_GENERATED_SM120_PRODUCTION_SOURCE \
  "cake_sm120_megamoe_production.cu"
#endif
#include CAKE_GENERATED_SM120_PRODUCTION_SOURCE

namespace {

constexpr int kPhysicalRanks = cake_moe::kPhysicalRanks;
constexpr int kRingSlots = cake_moe::kRingSlots;
constexpr int kEpochs = 3;
constexpr int kGinContexts = cake_moe::kGinContexts;
constexpr int kGinSignals = 2 * kPhysicalRanks;
constexpr int kHeaderWords = cake_moe::kHeaderWords;
constexpr int kTopK = cake_moe::kTopK;
constexpr int kLocalExperts = cake_moe::kLocalExperts;
constexpr int kMaxRows = cake_moe::kMaxRows;
constexpr int kHidden = cake_moe::kHidden;
constexpr int kIntermediate = cake_moe::kIntermediate;
constexpr int kOutput = cake_moe::kOutput;
constexpr int kTaskM = 16;
constexpr int kBlockK = cake_moe::kBlockK;
constexpr int kK32Steps = 4;
constexpr int kThreads = 128;

// Identity string the reference manifest must agree with, derived from the
// configured shape so a rebuild cannot silently consume the wrong references.
std::string shape_identity() {
  return "H" + std::to_string(kHidden) + "-I" + std::to_string(kIntermediate) +
         "-O" + std::to_string(kOutput) + "-K" + std::to_string(kTopK) + "-E" +
         std::to_string(cake_moe::kGlobalExperts);
}
constexpr int kW1KBlocks = kHidden / kBlockK;
constexpr int kW2KBlocks = kIntermediate / kBlockK;
constexpr int kW1LogicalNTiles = kIntermediate / 64;
constexpr int kW2NTiles = kOutput / 64;
constexpr int kW1PhysicalN = 2 * kIntermediate;
constexpr int kRecordBytes = cake_moe::kRecordBytes;
constexpr int kMaxRecordsPerPeer = kMaxRows;
constexpr int kMaxRoutesPerPeer = kMaxRows * kTopK;
constexpr std::size_t kDispatchPeerSlotBytes =
    static_cast<std::size_t>(kMaxRecordsPerPeer) * kRecordBytes;
constexpr std::size_t kDispatchWindowBytes =
    kPhysicalRanks * kRingSlots * kDispatchPeerSlotBytes;
constexpr std::size_t kResultElementsPerPeer =
    static_cast<std::size_t>(kMaxRoutesPerPeer) * kOutput;
constexpr std::size_t kResultWindowElements =
    kPhysicalRanks * kRingSlots * kResultElementsPerPeer;
constexpr int kNumWaves = 6;
constexpr int kWaveCandidates = 16384;
constexpr int kMaxWavePaddedRows = 17104;
constexpr int kMaxWaveTasks = 1069;
constexpr std::size_t kW1OutputFloatsPerTask =
    2ULL * kTaskM * kIntermediate;
constexpr std::size_t kIntermediateBytesPerTask =
    static_cast<std::size_t>(kTaskM) * kIntermediate;
constexpr std::size_t kIntermediateSfBytesPerTask =
    static_cast<std::size_t>(kTaskM) * (kIntermediate / 32);
constexpr std::size_t kW1WeightBytes =
    static_cast<std::size_t>(kLocalExperts) * kW1PhysicalN * kHidden / 2;
constexpr std::size_t kW1WeightSfWords =
    static_cast<std::size_t>(kLocalExperts) * kW1KBlocks * kW1PhysicalN;
constexpr std::size_t kW2WeightBytes =
    static_cast<std::size_t>(kLocalExperts) * kOutput * kIntermediate / 2;
constexpr std::size_t kW2WeightSfWords =
    static_cast<std::size_t>(kLocalExperts) * kW2KBlocks * kOutput;
constexpr std::size_t kFinalElements =
    static_cast<std::size_t>(kMaxRows) * kOutput;
constexpr std::size_t kOutputGuardElements = 256;
constexpr std::uint16_t kOutputGuardBits = 0xa5a5;
constexpr std::uint32_t kObservableDistinctK32Scales = 0x817f7e7du;
constexpr int kDispatchSmem = 128;
constexpr int kTaskBuildSmem = 256;
constexpr int kW1Smem = 21760;
constexpr int kRequantSmem = 128;
constexpr int kW2Smem = 12544;
constexpr int kResultSmem = 0;
constexpr unsigned long long kReferenceSeed = 20260819ULL;

enum class OraclePattern : int {
  kZero = 0,
  kAnalytic = 1,
  kDistinctK32 = 2,
  kDenseExternal = 3,
};

std::atomic<int> g_failures{0};

[[noreturn]] void fail_cuda(cudaError_t status, const char* expression,
                            const char* file, int line) {
  std::fprintf(stderr, "CUDA failure at %s:%d: %s -> %s (%s)\n", file,
               line, expression, cudaGetErrorName(status),
               cudaGetErrorString(status));
  std::abort();
}

[[noreturn]] void fail_driver(CUresult status, const char* expression,
                              const char* file, int line) {
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  std::fprintf(stderr, "CUDA driver failure at %s:%d: %s -> %s (%s)\n",
               file, line, expression, name == nullptr ? "unknown" : name,
               message == nullptr ? "unknown" : message);
  std::abort();
}

[[noreturn]] void fail_nccl(ncclResult_t status, const char* expression,
                            const char* file, int line) {
  std::fprintf(stderr, "NCCL failure at %s:%d: %s -> %s\n", file, line,
               expression, ncclGetErrorString(status));
  std::abort();
}

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t status_ = (expr);                                            \
    if (status_ != cudaSuccess)                                              \
      fail_cuda(status_, #expr, __FILE__, __LINE__);                         \
  } while (0)

#define DRIVER_CHECK(expr)                                                   \
  do {                                                                       \
    CUresult status_ = (expr);                                               \
    if (status_ != CUDA_SUCCESS)                                             \
      fail_driver(status_, #expr, __FILE__, __LINE__);                       \
  } while (0)

#define NCCL_CHECK(expr)                                                     \
  do {                                                                       \
    ncclResult_t status_ = (expr);                                           \
    if (status_ != ncclSuccess)                                              \
      fail_nccl(status_, #expr, __FILE__, __LINE__);                         \
  } while (0)

long parse_long_env(const char* name, long fallback, long minimum,
                    long maximum) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') return fallback;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < minimum || value > maximum) {
    std::fprintf(stderr, "%s must be in [%ld,%ld], observed %s\n", name,
                 minimum, maximum, text);
    std::exit(EXIT_FAILURE);
  }
  return value;
}

std::string parse_choice_env(const char* name, const char* fallback,
                             std::initializer_list<const char*> choices) {
  const char* text = std::getenv(name);
  std::string value = text == nullptr ? fallback : text;
  for (const char* choice : choices) {
    if (value == choice) return value;
  }
  std::fprintf(stderr, "%s has unsupported value %s\n", name,
               value.c_str());
  std::exit(EXIT_FAILURE);
}

bool is_supported_world_size(int world_size) {
  return world_size == 1 || world_size == 2 || world_size == 4 ||
         world_size == 8;
}

OraclePattern oracle_pattern_from_string(const std::string& value) {
  if (value == "zero") return OraclePattern::kZero;
  if (value == "analytic") return OraclePattern::kAnalytic;
  if (value == "distinct_k32") return OraclePattern::kDistinctK32;
  if (value == "dense_external") return OraclePattern::kDenseExternal;
  std::fprintf(stderr, "unreachable oracle pattern %s\n", value.c_str());
  std::abort();
}

int route_expert(int token, int slot, int source_rank, int world_size,
                 const std::string& route_mode) {
  const int active_experts = world_size * kLocalExperts;
  if (route_mode == "balanced") {
    return (token * 17 + slot * 53 + source_rank * 97) % active_experts;
  }
  if (route_mode == "skewed") {
    return (token + slot * 3 + source_rank) % std::min(8, active_experts);
  }
  // Six distinct experts preserve production top-k uniqueness while leaving
  // all remaining experts empty.
  return slot;
}

float route_weight(int slot) {
  constexpr std::array<float, kTopK> kWeights = {
      0.5f, 0.25f, 0.125f, 0.5f, 0.25f, 0.125f};
  return kWeights[slot];
}

template <typename T>
struct SymmetricWindow {
  T* pointer = nullptr;
  ncclWindow_t window{};
  std::size_t bytes = 0;
};

template <typename T>
SymmetricWindow<T> allocate_window(ncclComm_t comm, std::size_t count) {
  SymmetricWindow<T> result;
  result.bytes = count * sizeof(T);
  NCCL_CHECK(
      ncclMemAlloc(reinterpret_cast<void**>(&result.pointer), result.bytes));
  NCCL_CHECK(ncclCommWindowRegister(comm, result.pointer, result.bytes,
                                    &result.window,
                                    NCCL_WIN_COLL_SYMMETRIC));
  return result;
}

template <typename T>
void destroy_window(ncclComm_t comm, SymmetricWindow<T>& allocation) {
  NCCL_CHECK(ncclCommWindowDeregister(comm, allocation.window));
  NCCL_CHECK(ncclMemFree(allocation.pointer));
  allocation.pointer = nullptr;
}

template <typename T>
void device_alloc(T** pointer, std::size_t count) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(pointer),
                        std::max<std::size_t>(count, 1) * sizeof(T)));
}

void communicator_barrier(ncclComm_t comm, cudaStream_t stream,
                          int world_size) {
  int* value = nullptr;
  int one = 1;
  int observed = 0;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&value), sizeof(*value)));
  CUDA_CHECK(cudaMemcpyAsync(value, &one, sizeof(one),
                             cudaMemcpyHostToDevice, stream));
  NCCL_CHECK(
      ncclAllReduce(value, value, 1, ncclInt32, ncclSum, comm, stream));
  CUDA_CHECK(cudaMemcpyAsync(&observed, value, sizeof(observed),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(value));
  if (observed != world_size) {
    std::fprintf(stderr, "rank barrier observed %d, expected %d\n", observed,
                 world_size);
    std::abort();
  }
}

ncclUniqueId load_or_create_unique_id(const char* path, int rank) {
  ncclUniqueId unique_id{};
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    const std::string temporary = std::string(path) + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr ||
        std::fwrite(&unique_id, sizeof(unique_id), 1, file) != 1 ||
        std::fclose(file) != 0 || std::rename(temporary.c_str(), path) != 0) {
      std::perror("publishing NCCL unique id");
      std::abort();
    }
    return unique_id;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(180);
  while (std::chrono::steady_clock::now() < deadline) {
    FILE* file = std::fopen(path, "rb");
    if (file != nullptr) {
      const bool complete =
          std::fread(&unique_id, sizeof(unique_id), 1, file) == 1;
      std::fclose(file);
      if (complete) return unique_id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::fprintf(stderr, "rank %d timed out waiting for NCCL id %s\n", rank,
               path);
  std::abort();
}

__device__ __forceinline__ int sparse_w1_k(int expert, int physical_n) {
  const int k32 = (physical_n * 37 + expert * 13) % (kHidden / 32);
  const int lane = (physical_n * 11 + expert * 7) & 31;
  return k32 * 32 + lane;
}

__device__ __forceinline__ int sparse_w2_k(int expert, int output_column) {
  const int k32 =
      (output_column * 29 + expert * 7) % (kIntermediate / 32);
  const int lane = (output_column * 13 + expert * 5) & 31;
  return k32 * 32 + lane;
}

__device__ __forceinline__ std::uint8_t pseudo_fp8_code(
    unsigned long long index, int rank) {
  constexpr std::uint8_t codes[8] = {
      0x28, 0x30, 0x38, 0x40, 0xa8, 0xb0, 0xb8, 0xc0};
  const unsigned long long mixed =
      (index + kReferenceSeed) * 0x9e3779b97f4a7c15ULL +
      static_cast<unsigned long long>(rank + 1) * 0xbf58476d1ce4e5b9ULL;
  return codes[(mixed ^ (mixed >> 29)) & 7];
}

__device__ __forceinline__ std::uint8_t pseudo_fp4_code(
    int expert, int n, int k) {
  constexpr std::uint8_t codes[8] = {1, 2, 3, 4, 5, 6, 9, 10};
  const unsigned int mixed = static_cast<unsigned int>(
      expert * 1315423911u + n * 2654435761u + k * 2246822519u +
      static_cast<unsigned int>(kReferenceSeed));
  return codes[(mixed ^ (mixed >> 13)) & 7];
}

__device__ __forceinline__ std::uint8_t weight_code(
    OraclePattern pattern, bool is_w1, int expert, int n, int k) {
  if (pattern == OraclePattern::kZero) return 0;
  if (pattern == OraclePattern::kDenseExternal) {
    return pseudo_fp4_code(expert, n, k);
  }
  const int selected =
      is_w1 ? sparse_w1_k(expert, n) : sparse_w2_k(expert, n);
  if (k != selected) return 0;
  return static_cast<std::uint8_t>(2 | (((expert + n) & 1) ? 8 : 0));
}

__device__ __forceinline__ std::uint8_t weight_scale(
    int expert, int n, int k32_global) {
  return static_cast<std::uint8_t>(126 +
                                   (expert + n * 3 + k32_global) % 4);
}

__global__ void initialize_x(std::uint8_t* x, std::uint32_t* scales,
                             int active_rows, int rank,
                             OraclePattern pattern) {
  const unsigned long long x_count =
      static_cast<unsigned long long>(kMaxRows) * kHidden;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < x_count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int token = static_cast<int>(index / kHidden);
    std::uint8_t value = 0;
    if (token < active_rows && pattern != OraclePattern::kZero) {
      value = pattern == OraclePattern::kDistinctK32
                  ? static_cast<std::uint8_t>(0x38)
                  : pseudo_fp8_code(index, rank);
    }
    x[index] = value;
  }
  const unsigned long long sf_count =
      static_cast<unsigned long long>(kMaxRows) * kW1KBlocks;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < sf_count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int token = static_cast<int>(index / kW1KBlocks);
    if (pattern == OraclePattern::kDistinctK32 && token < active_rows) {
      scales[index] = kObservableDistinctK32Scales;
    } else {
      const int base = 126 + static_cast<int>((index + rank) & 1);
      scales[index] = static_cast<std::uint32_t>(base) |
                      (static_cast<std::uint32_t>(base + 1) << 8) |
                      (static_cast<std::uint32_t>(base + 2) << 16) |
                      (static_cast<std::uint32_t>(base + 3) << 24);
    }
  }
}

__global__ void initialize_fp4_weights(std::uint8_t* packed,
                                       unsigned long long count, int rank,
                                       OraclePattern pattern, bool is_w1) {
  const int n_extent = is_w1 ? kW1PhysicalN : kOutput;
  const int k_extent = is_w1 ? kHidden : kIntermediate;
  const unsigned long long bytes_per_n = k_extent / 2;
  const unsigned long long bytes_per_expert =
      static_cast<unsigned long long>(n_extent) * bytes_per_n;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int local_expert = static_cast<int>(index / bytes_per_expert);
    const unsigned long long expert_offset = index % bytes_per_expert;
    const int n = static_cast<int>(expert_offset / bytes_per_n);
    const int k_pair = static_cast<int>(expert_offset % bytes_per_n);
    const int expert = rank * kLocalExperts + local_expert;
    const std::uint8_t lo =
        weight_code(pattern, is_w1, expert, n, 2 * k_pair);
    const std::uint8_t hi =
        weight_code(pattern, is_w1, expert, n, 2 * k_pair + 1);
    packed[index] = static_cast<std::uint8_t>(lo | (hi << 4));
  }
}

__global__ void initialize_weight_scales(std::uint32_t* scales,
                                         unsigned long long count, int rank,
                                         bool is_w1) {
  const int n_extent = is_w1 ? kW1PhysicalN : kOutput;
  const int k_blocks = is_w1 ? kW1KBlocks : kW2KBlocks;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(index % n_extent);
    const unsigned long long outer = index / n_extent;
    const int k_block = static_cast<int>(outer % k_blocks);
    const int local_expert = static_cast<int>(outer / k_blocks);
    const int expert = rank * kLocalExperts + local_expert;
    std::uint32_t packed = 0;
    for (int k32 = 0; k32 < kK32Steps; ++k32) {
      packed |= static_cast<std::uint32_t>(
                    weight_scale(expert, n, k_block * kK32Steps + k32))
                << (k32 * 8);
    }
    scales[index] = packed;
  }
}

__device__ __forceinline__ float decode_fp8(std::uint8_t code) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
  const __half value(raw);
  return __half2float(value);
}

__device__ __forceinline__ float decode_fp4(std::uint8_t code) {
  constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                  2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitude[code & 7];
  return (code & 8) == 0 ? value : -value;
}

__device__ __forceinline__ float decode_ue8m0(std::uint8_t exponent) {
  return __uint_as_float(static_cast<unsigned int>(exponent) << 23);
}

__device__ __forceinline__ int physical_w1_n(int branch, int logical_n) {
  return (logical_n / 8) * 16 + branch * 8 + (logical_n & 7);
}

// Independent composed GPU reference for the analytic full-dimension pattern.
// A warp computes one route/gran32 group.  It does not read any CAKE output.
__global__ void reference_w1_requant(
    const std::uint8_t* x, const std::uint32_t* x_sf,
    const int* topk_idx_i32, const float* topk_weights,
    std::uint8_t* intermediate, std::uint8_t* intermediate_sf,
    int active_rows, OraclePattern pattern) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * (blockDim.x / 32) + warp;
  const int warp_stride = gridDim.x * (blockDim.x / 32);
  const int route_groups = active_rows * kTopK * (kIntermediate / 32);
  for (int route_group = global_warp; route_group < route_groups;
       route_group += warp_stride) {
    const int route = route_group / (kIntermediate / 32);
    const int group = route_group % (kIntermediate / 32);
    const int token = route / kTopK;
    const int expert = topk_idx_i32[route * 2];
    const bool valid = expert >= 0;
    const int logical_n = group * 32 + lane;
    float routed = 0.0f;
    if (valid) {
      float branch_value[2];
#pragma unroll
      for (int branch = 0; branch < 2; ++branch) {
        const int physical_n = physical_w1_n(branch, logical_n);
        const int k = sparse_w1_k(expert, physical_n);
        const int k_block = k / kBlockK;
        const int k32 = (k % kBlockK) / 32;
        const std::uint8_t a_code = x[token * kHidden + k];
        const std::uint8_t a_sf = static_cast<std::uint8_t>(
            x_sf[token * kW1KBlocks + k_block] >> (k32 * 8));
        const std::uint8_t b_code =
            weight_code(pattern, true, expert, physical_n, k);
        const std::uint8_t b_sf =
            weight_scale(expert, physical_n, k / 32);
        const float product =
            decode_fp8(a_code) * decode_fp4(b_code) * decode_ue8m0(a_sf) *
            decode_ue8m0(b_sf);
        branch_value[branch] =
            __bfloat162float(__float2bfloat16_rn(product));
      }
      const float gate = fminf(branch_value[0], 10.0f);
      const float up = fminf(fmaxf(branch_value[1], -10.0f), 10.0f);
      const float exp_value = __expf(-gate);
      float reciprocal;
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal)
          : "f"(1.0f + exp_value));
      const float silu = gate * reciprocal;
      const float swiglu = silu * up;
      routed = swiglu * topk_weights[route];
    }
    float amax = fmaxf(routed, -routed);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    const float sf = amax * 0.0022321428571428572f;
    const unsigned int bits = __float_as_uint(sf);
    unsigned int sf_exp = ((bits >> 23) & 255) +
                          (((bits & 0x7fffff) + 0x7fffff) >> 23);
    sf_exp = min(sf_exp, 254u);
    const float sf_inv = __uint_as_float((254u - sf_exp) << 23);
    intermediate[static_cast<std::size_t>(route) * kIntermediate +
                 logical_n] =
        __nv_cvt_float_to_fp8(routed * sf_inv, __NV_SATFINITE, __NV_E4M3);
    if (lane == 0) {
      intermediate_sf[static_cast<std::size_t>(route) *
                          (kIntermediate / 32) +
                      group] = static_cast<std::uint8_t>(sf_exp);
    }
  }
}

__global__ void reference_w2(
    const int* topk_idx_i32, const std::uint8_t* intermediate,
    const std::uint8_t* intermediate_sf, __nv_bfloat16* route_partials,
    int active_rows, OraclePattern pattern) {
  const unsigned long long count =
      static_cast<unsigned long long>(active_rows) * kTopK * kOutput;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int output_column = static_cast<int>(index % kOutput);
    const int route = static_cast<int>(index / kOutput);
    const int expert = topk_idx_i32[route * 2];
    float value = 0.0f;
    if (expert >= 0) {
      const int k = sparse_w2_k(expert, output_column);
      const std::uint8_t a_code =
          intermediate[static_cast<std::size_t>(route) * kIntermediate + k];
      const std::uint8_t a_sf = intermediate_sf[
          static_cast<std::size_t>(route) * (kIntermediate / 32) + k / 32];
      const std::uint8_t b_code =
          weight_code(pattern, false, expert, output_column, k);
      const std::uint8_t b_sf =
          weight_scale(expert, output_column, k / 32);
      value = decode_fp8(a_code) * decode_fp4(b_code) *
              decode_ue8m0(a_sf) * decode_ue8m0(b_sf);
    }
    route_partials[index] = __float2bfloat16_rn(value);
  }
}

__global__ void reference_combine(const int* topk_idx_i32,
                                  const __nv_bfloat16* route_partials,
                                  __nv_bfloat16* output, int active_rows) {
  const unsigned long long count =
      static_cast<unsigned long long>(active_rows) * kOutput;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int token = static_cast<int>(index / kOutput);
    const int column = static_cast<int>(index % kOutput);
    float combined = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      const int route = token * kTopK + slot;
      if (topk_idx_i32[route * 2] >= 0) {
        combined += __bfloat162float(
            route_partials[static_cast<std::size_t>(route) * kOutput +
                           column]);
      }
    }
    output[index] = __float2bfloat16_rn(combined);
  }
}

LoomTensorMap* encode_weight_tma(std::uint8_t* weight_fp4, bool is_w1) {
  CUtensorMap tensor_map{};
  const cuuint64_t global_dim[4] = {
      static_cast<cuuint64_t>(kBlockK),
      static_cast<cuuint64_t>(is_w1 ? kW1PhysicalN : kOutput),
      static_cast<cuuint64_t>(is_w1 ? kW1KBlocks : kW2KBlocks),
      static_cast<cuuint64_t>(kLocalExperts),
  };
  const cuuint64_t global_strides[3] = {
      static_cast<cuuint64_t>((is_w1 ? kHidden : kIntermediate) / 2),
      static_cast<cuuint64_t>(kBlockK / 2),
      static_cast<cuuint64_t>(is_w1 ? kW1PhysicalN : kOutput) *
          static_cast<cuuint64_t>((is_w1 ? kHidden : kIntermediate) / 2),
  };
  const cuuint32_t box_dim[4] = {
      static_cast<cuuint32_t>(kBlockK),
      static_cast<cuuint32_t>(is_w1 ? 128 : 64), 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  DRIVER_CHECK(cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 4, weight_fp4,
      global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  LoomTensorMap* device_map = nullptr;
  static_assert(sizeof(CUtensorMap) == sizeof(LoomTensorMap));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_map),
                        sizeof(LoomTensorMap)));
  CUDA_CHECK(cudaMemcpy(device_map, &tensor_map, sizeof(tensor_map),
                        cudaMemcpyHostToDevice));
  return device_map;
}

struct DeviceBuffers {
  int* topk_idx = nullptr;
  float* topk_weights = nullptr;
  std::uint8_t* x = nullptr;
  std::uint32_t* x_sf = nullptr;
  std::uint32_t* owner_record_counts = nullptr;
  std::uint32_t* owner_route_counts = nullptr;
  int* route_result_index = nullptr;
  std::uint32_t* protocol_error = nullptr;
  unsigned long long* dispatch_signal_base_scratch = nullptr;
  unsigned long long* result_signal_base_scratch = nullptr;

  std::uint8_t* pool_fp8 = nullptr;
  std::uint32_t* pool_sf = nullptr;
  float* routing_weight_pool = nullptr;
  int* meta_source_rank = nullptr;
  int* meta_token = nullptr;
  int* meta_slot = nullptr;
  int* meta_result_index = nullptr;
  int* expert_counts = nullptr;
  int* source_record_counts = nullptr;
  int* source_route_counts = nullptr;
  int* source_active_rows = nullptr;
  int* expert_row_offsets = nullptr;
  int* expert_scatter_offsets = nullptr;
  int* task_expert = nullptr;
  int* task_source_rank = nullptr;
  int* task_owner_rank = nullptr;
  int* task_local_expert = nullptr;
  int* task_pool_row = nullptr;
  int* task_m_local = nullptr;
  int* task_valid_m = nullptr;
  int* selected_task_indices = nullptr;
  int* total_valid_routes = nullptr;
  int* total_padded_rows = nullptr;
  int* total_m_tasks = nullptr;
  std::uint32_t* histogram_done = nullptr;
  std::uint32_t* prefix_done = nullptr;

  std::uint8_t* w1_weight = nullptr;
  std::uint32_t* w1_weight_sf = nullptr;
  LoomTensorMap* w1_tma = nullptr;
  float* w1_output = nullptr;
  std::uint32_t* w1_task_completion = nullptr;
  std::uint32_t* w1_completed = nullptr;
  std::uint8_t* intermediate = nullptr;
  std::uint8_t* intermediate_sf = nullptr;
  std::uint32_t* requant_completed = nullptr;
  std::uint8_t* w2_weight = nullptr;
  std::uint32_t* w2_weight_sf = nullptr;
  LoomTensorMap* w2_tma = nullptr;
  std::uint32_t* w2_task_completion = nullptr;
  std::uint32_t* w2_completed = nullptr;

  __nv_bfloat16* final_allocation = nullptr;
  __nv_bfloat16* reference_output = nullptr;
  std::uint8_t* reference_intermediate = nullptr;
  std::uint8_t* reference_intermediate_sf = nullptr;
  __nv_bfloat16* reference_partials = nullptr;
  ncclDevComm* device_comm = nullptr;
};

std::uint32_t rotate_right(std::uint32_t value, int amount) {
  return (value >> amount) | (value << (32 - amount));
}

std::string sha256_hex(const std::uint8_t* data, std::size_t size) {
  constexpr std::array<std::uint32_t, 64> kRound = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  std::array<std::uint32_t, 8> state = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::vector<std::uint8_t> padded(data, data + size);
  padded.push_back(0x80);
  while ((padded.size() % 64) != 56) padded.push_back(0);
  const std::uint64_t bit_count = static_cast<std::uint64_t>(size) * 8;
  for (int shift = 56; shift >= 0; shift -= 8) {
    padded.push_back(static_cast<std::uint8_t>(bit_count >> shift));
  }
  for (std::size_t block = 0; block < padded.size(); block += 64) {
    std::array<std::uint32_t, 64> words{};
    for (int i = 0; i < 16; ++i) {
      words[i] = static_cast<std::uint32_t>(padded[block + i * 4]) << 24 |
                 static_cast<std::uint32_t>(padded[block + i * 4 + 1])
                     << 16 |
                 static_cast<std::uint32_t>(padded[block + i * 4 + 2]) << 8 |
                 static_cast<std::uint32_t>(padded[block + i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                 rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + sum1 + choose + kRound[i] + words[i];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                 rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  char output[65]{};
  for (int i = 0; i < 8; ++i) {
    std::snprintf(output + i * 8, 9, "%08x", state[i]);
  }
  return output;
}

std::map<std::string, std::string> load_manifest(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open reference manifest %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::map<std::string, std::string> values;
  char line[1024];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    std::string text(line);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
      text.pop_back();
    }
    if (text.empty() || text[0] == '#') continue;
    const std::size_t separator = text.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == text.size()) {
      std::fprintf(stderr, "malformed reference manifest line: %s\n",
                   text.c_str());
      std::exit(EXIT_FAILURE);
    }
    const std::string key = text.substr(0, separator);
    if (!values.emplace(key, text.substr(separator + 1)).second) {
      std::fprintf(stderr, "duplicate reference manifest key %s\n",
                   key.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  std::fclose(file);
  return values;
}

void require_manifest_value(const std::map<std::string, std::string>& manifest,
                            const std::string& key,
                            const std::string& expected) {
  const auto it = manifest.find(key);
  if (it == manifest.end() || it->second != expected) {
    std::fprintf(stderr,
                 "reference manifest mismatch key=%s observed=%s expected=%s\n",
                 key.c_str(), it == manifest.end() ? "<missing>"
                                                   : it->second.c_str(),
                 expected.c_str());
    std::exit(EXIT_FAILURE);
  }
}

struct ExternalReference {
  std::vector<std::uint16_t> values;
  std::string sha256;
};

ExternalReference load_external_reference(
    const std::string& prefix, int rank, int world_size, int active_rows,
    int mask_period, const std::string& route_mode) {
  const std::string path =
      prefix + ".rank" + std::to_string(rank) + ".bf16";
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open independent reference %s\n",
                 path.c_str());
    std::exit(EXIT_FAILURE);
  }
  const std::size_t count =
      static_cast<std::size_t>(active_rows) * kOutput;
  ExternalReference result;
  result.values.resize(count);
  const bool complete =
      std::fread(result.values.data(), sizeof(result.values[0]), count, file) ==
      count;
  const int trailing = std::fgetc(file);
  std::fclose(file);
  if (!complete || trailing != EOF) {
    std::fprintf(stderr,
                 "reference %s must contain exactly %zu little-endian BF16 "
                 "values\n",
                 path.c_str(), count);
    std::exit(EXIT_FAILURE);
  }
  const std::size_t bytes = count * sizeof(result.values[0]);
  result.sha256 = sha256_hex(
      reinterpret_cast<const std::uint8_t*>(result.values.data()), bytes);
  const auto manifest = load_manifest(
      prefix + ".rank" + std::to_string(rank) + ".manifest");
  require_manifest_value(manifest, "schema",
                         "cake-sm120-dense-reference-v1");
  require_manifest_value(manifest, "reference_impl",
                         "official-deepgemm-split-a32-b32");
  require_manifest_value(manifest, "reference_repo",
                         "deepseek-ai/DeepGEMM");
  require_manifest_value(
      manifest, "reference_git_commit",
      "559d79fb6994a58b8a15b4b93bf13ccc16edf247");
  require_manifest_value(manifest, "rank", std::to_string(rank));
  require_manifest_value(manifest, "world_size", std::to_string(world_size));
  require_manifest_value(manifest, "active_rows", std::to_string(active_rows));
  require_manifest_value(manifest, "route_mode", route_mode);
  require_manifest_value(manifest, "mask_period",
                         std::to_string(mask_period));
  require_manifest_value(manifest, "seed",
                         std::to_string(kReferenceSeed));
  require_manifest_value(manifest, "fast_math", "true");
  require_manifest_value(manifest, "activation_clamp", "10");
  require_manifest_value(manifest, "shape", shape_identity());
  require_manifest_value(manifest, "activation_gran_k", "32");
  require_manifest_value(manifest, "weight_gran_k", "32");
  require_manifest_value(manifest, "dtype", "bf16-le");
  require_manifest_value(manifest, "bf16_bytes", std::to_string(bytes));
  require_manifest_value(manifest, "bf16_sha256", result.sha256);
  return result;
}

int cooperative_capacity(const void* function, int sm_count, int smem) {
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, function, kThreads, smem));
  return blocks_per_sm * sm_count;
}

void free_buffers(DeviceBuffers& b) {
  cudaFree(b.device_comm);
  cudaFree(b.reference_partials);
  cudaFree(b.reference_intermediate_sf);
  cudaFree(b.reference_intermediate);
  cudaFree(b.reference_output);
  cudaFree(b.final_allocation);
  cudaFree(b.w2_completed);
  cudaFree(b.w2_task_completion);
  cudaFree(b.w2_tma);
  cudaFree(b.w2_weight_sf);
  cudaFree(b.w2_weight);
  cudaFree(b.requant_completed);
  cudaFree(b.intermediate_sf);
  cudaFree(b.intermediate);
  cudaFree(b.w1_completed);
  cudaFree(b.w1_task_completion);
  cudaFree(b.w1_output);
  cudaFree(b.w1_tma);
  cudaFree(b.w1_weight_sf);
  cudaFree(b.w1_weight);
  cudaFree(b.prefix_done);
  cudaFree(b.histogram_done);
  cudaFree(b.total_m_tasks);
  cudaFree(b.total_padded_rows);
  cudaFree(b.total_valid_routes);
  cudaFree(b.selected_task_indices);
  cudaFree(b.task_valid_m);
  cudaFree(b.task_m_local);
  cudaFree(b.task_pool_row);
  cudaFree(b.task_local_expert);
  cudaFree(b.task_owner_rank);
  cudaFree(b.task_source_rank);
  cudaFree(b.task_expert);
  cudaFree(b.expert_scatter_offsets);
  cudaFree(b.expert_row_offsets);
  cudaFree(b.source_active_rows);
  cudaFree(b.source_route_counts);
  cudaFree(b.source_record_counts);
  cudaFree(b.expert_counts);
  cudaFree(b.meta_result_index);
  cudaFree(b.meta_slot);
  cudaFree(b.meta_token);
  cudaFree(b.meta_source_rank);
  cudaFree(b.routing_weight_pool);
  cudaFree(b.pool_sf);
  cudaFree(b.pool_fp8);
  cudaFree(b.result_signal_base_scratch);
  cudaFree(b.dispatch_signal_base_scratch);
  cudaFree(b.protocol_error);
  cudaFree(b.route_result_index);
  cudaFree(b.owner_route_counts);
  cudaFree(b.owner_record_counts);
  cudaFree(b.x_sf);
  cudaFree(b.x);
  cudaFree(b.topk_weights);
  cudaFree(b.topk_idx);
}

}  // namespace
