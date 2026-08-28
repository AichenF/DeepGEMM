/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

// Fail-closed correctness host for the single-entry SM120 MegaMoE G8 kernel.
// It uses deterministic full-dimension fixtures and independent stage oracles.

#ifndef CAKE_GENERATED_SM120_PRODUCTION_CANONICAL_FUSED_READY_CHUNK8_SOURCE
#define CAKE_GENERATED_SM120_PRODUCTION_CANONICAL_FUSED_READY_CHUNK8_SOURCE \
  "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
#endif

#define CAKE_GENERATED_SM120_PRODUCTION_SOURCE \
  CAKE_GENERATED_SM120_PRODUCTION_CANONICAL_FUSED_READY_CHUNK8_SOURCE
#include "deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu"
#undef CAKE_GENERATED_SM120_PRODUCTION_SOURCE

#include <algorithm>

namespace {

constexpr int kCanonicalTaskM = cake_moe::kTaskM;
constexpr int kCanonicalThreads = 384;
constexpr int kCanonicalMaxPaddedRows = cake_moe::kMaxPaddedRows;
constexpr int kCanonicalMaxTasks = cake_moe::kMaxTasks;
constexpr int kCanonicalReadyCtas = cake_moe::kCombineCtas;
constexpr int kCanonicalReadySmem = 94208;
constexpr int kCanonicalDescriptorCount = 10;

struct CanonicalBuffers {
  int* grouped_layout = nullptr;
  cutlass::bfloat16_t* w1_bf16 = nullptr;
  cutlass::bfloat16_t* w2_bf16 = nullptr;
  cute::TmaDescriptor* tensor_map_buffer = nullptr;
  cute::TmaDescriptor* descriptor_storage = nullptr;
  CakeSm120CanonicalFusedReadyParams* ready_params = nullptr;
  std::uint32_t* stage_mismatches = nullptr;
  std::uint32_t* w2_signed_zero_differences = nullptr;
  unsigned int* w1_warp_done = nullptr;
  unsigned int* w1_task_ready = nullptr;
  unsigned int* w1_next_tile = nullptr;
  unsigned int* w1_tiles_completed = nullptr;
  unsigned int* epilogue_claimed = nullptr;
  unsigned int* epilogue_completed = nullptr;
  unsigned int* w2_task_ready = nullptr;
  unsigned int* w2_task_claimed = nullptr;
  unsigned int* w2_tile_warp_done = nullptr;
  unsigned int* w2_tiles_completed = nullptr;
  unsigned int* source_w2_done = nullptr;
  unsigned int* combine_ready = nullptr;
  unsigned int* combine_ctas_done = nullptr;
  unsigned int* epoch_done = nullptr;
  unsigned int* ready_audit_counts = nullptr;
  int* worker_task = nullptr;
  int* worker_n = nullptr;
  unsigned long long* combine_ack_signal_base_scratch = nullptr;
#if CAKE_MOE_PHASE_TRACE
  unsigned long long* phase_ns = nullptr;
  unsigned int* phase_count = nullptr;
#endif
};

struct CanonicalMaps {
  CUtensorMap w1_a{};
  CUtensorMap w1_b{};
  CUtensorMap w1_sfa{};
  CUtensorMap w1_sfb{};
  CUtensorMap w1_d{};
  CUtensorMap w2_a{};
  CUtensorMap w2_b{};
  CUtensorMap w2_sfa{};
  CUtensorMap w2_sfb{};
  CUtensorMap w2_d{};
};

int canonical_cooperative_capacity(const void* function, int sm_count,
                                   int threads_per_block, int smem) {
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, function, threads_per_block, smem));
  return blocks_per_sm * sm_count;
}

CUtensorMap encode_canonical_2d(CUtensorMapDataType dtype, void* pointer,
                                cuuint64_t inner, cuuint64_t outer,
                                cuuint64_t outer_stride_bytes,
                                cuuint32_t box_inner,
                                cuuint32_t box_outer,
                                CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {outer_stride_bytes};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t element_strides[2] = {1, 1};
  DRIVER_CHECK(cuTensorMapEncodeTiled(
      &map, dtype, 2, pointer, dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

CanonicalMaps encode_canonical_maps(DeviceBuffers& b,
                                    CanonicalBuffers& canonical) {
  CanonicalMaps maps;
  maps.w1_a = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_UINT8, b.pool_fp8, kHidden,
      kCanonicalMaxPaddedRows, kHidden, 128, kCanonicalTaskM,
      CU_TENSOR_MAP_SWIZZLE_128B);
  maps.w1_b = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, b.w1_weight, kHidden,
      static_cast<cuuint64_t>(kW1PhysicalN) * kLocalExperts, kHidden / 2,
      128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  maps.w1_sfa = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_INT32, b.pool_sf,
      kCanonicalMaxPaddedRows, kW1KBlocks,
      static_cast<cuuint64_t>(kCanonicalMaxPaddedRows) * sizeof(std::uint32_t),
      kCanonicalTaskM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  maps.w1_sfb = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_INT32, b.w1_weight_sf, kW1PhysicalN,
      static_cast<cuuint64_t>(kW1KBlocks) * kLocalExperts,
      static_cast<cuuint64_t>(kW1PhysicalN) * sizeof(std::uint32_t), 128, 1,
      CU_TENSOR_MAP_SWIZZLE_NONE);
  maps.w1_d = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, canonical.w1_bf16, kW1PhysicalN,
      kCanonicalMaxPaddedRows,
      static_cast<cuuint64_t>(kW1PhysicalN) * sizeof(cutlass::bfloat16_t),
      128, kCanonicalTaskM, CU_TENSOR_MAP_SWIZZLE_NONE);

  maps.w2_a = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_UINT8, b.intermediate, kIntermediate,
      kCanonicalMaxPaddedRows, kIntermediate, 128, kCanonicalTaskM,
      CU_TENSOR_MAP_SWIZZLE_128B);
  maps.w2_b = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, b.w2_weight, kIntermediate,
      static_cast<cuuint64_t>(kOutput) * kLocalExperts, kIntermediate / 2,
      128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  maps.w2_sfa = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_INT32, b.intermediate_sf,
      kCanonicalMaxPaddedRows, kW2KBlocks,
      static_cast<cuuint64_t>(kCanonicalMaxPaddedRows) * sizeof(std::uint32_t),
      kCanonicalTaskM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  maps.w2_sfb = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_INT32, b.w2_weight_sf, kOutput,
      static_cast<cuuint64_t>(kW2KBlocks) * kLocalExperts,
      static_cast<cuuint64_t>(kOutput) * sizeof(std::uint32_t), 128, 1,
      CU_TENSOR_MAP_SWIZZLE_NONE);
  maps.w2_d = encode_canonical_2d(
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, canonical.w2_bf16, kOutput,
      kCanonicalMaxPaddedRows,
      static_cast<cuuint64_t>(kOutput) * sizeof(cutlass::bfloat16_t), 128,
      kCanonicalTaskM, CU_TENSOR_MAP_SWIZZLE_NONE);
  return maps;
}

void upload_canonical_maps(CanonicalBuffers& canonical,
                           const CanonicalMaps& maps) {
  static_assert(sizeof(CUtensorMap) == sizeof(cute::TmaDescriptor));
  static_assert(alignof(CakeSm120CanonicalFusedReadyParams) >= 16);
  if ((reinterpret_cast<std::uintptr_t>(canonical.descriptor_storage) & 127u) !=
      0u) {
    std::fprintf(stderr, "canonical descriptor storage is not 128B aligned\n");
    std::abort();
  }
  const std::array<CUtensorMap, kCanonicalDescriptorCount> descriptor_images = {
      maps.w1_a, maps.w1_b, maps.w1_sfa, maps.w1_sfb, maps.w1_d,
      maps.w2_a, maps.w2_b, maps.w2_sfa, maps.w2_sfb, maps.w2_d};
  CUDA_CHECK(cudaMemcpy(canonical.descriptor_storage,
                        descriptor_images.data(), sizeof(descriptor_images),
                        cudaMemcpyHostToDevice));
}

CakeSm120CanonicalFusedReadyParams make_ready_params(
    DeviceBuffers& b, CanonicalBuffers& canonical, int rank, int world_size,
    int active_rows, unsigned int epoch,
    SymmetricWindow<int>& dispatch_header_out,
    SymmetricWindow<std::uint8_t>& dispatch_payload_out,
    SymmetricWindow<int>& dispatch_header_inbox,
    SymmetricWindow<std::uint8_t>& dispatch_payload_inbox,
    SymmetricWindow<__nv_bfloat16>& result_out,
    SymmetricWindow<__nv_bfloat16>& result_inbox,
    SymmetricWindow<std::uint8_t>& ack_out,
    SymmetricWindow<std::uint8_t>& ack_inbox,
    __nv_bfloat16* final_output) {
  CakeSm120CanonicalFusedReadyParams p{};
  p.topk_idx_i32 = b.topk_idx;
  p.topk_weights = b.topk_weights;
  p.x_fp8_i32 = reinterpret_cast<int*>(b.x);
  p.x_sf_i32 = reinterpret_cast<int*>(b.x_sf);
  p.owner_record_counts = b.owner_record_counts;
  p.owner_route_counts = b.owner_route_counts;
  p.route_result_index = b.route_result_index;
  p.protocol_error = b.protocol_error;
  p.dispatch_signal_base_scratch = b.dispatch_signal_base_scratch;
  p.result_signal_base_scratch = b.result_signal_base_scratch;
  p.rank = rank;
  p.world_size = world_size;
  p.active_rows = active_rows;
  p.epoch = epoch;
  p.gin_dev_comm = b.device_comm;
  p.dispatch_header_out = dispatch_header_out.pointer;
  p.dispatch_header_out_window = dispatch_header_out.window;
  p.dispatch_payload_out = dispatch_payload_out.pointer;
  p.dispatch_payload_out_window = dispatch_payload_out.window;
  p.dispatch_header_inbox = dispatch_header_inbox.pointer;
  p.dispatch_header_inbox_window = dispatch_header_inbox.window;
  p.dispatch_payload_inbox = dispatch_payload_inbox.pointer;
  p.dispatch_payload_inbox_window = dispatch_payload_inbox.window;
  p.pool_fp8_u32 = reinterpret_cast<unsigned int*>(b.pool_fp8);
  p.pool_sf_u32 = reinterpret_cast<unsigned int*>(b.pool_sf);
  p.routing_weight_pool = b.routing_weight_pool;
  p.meta_source_rank = b.meta_source_rank;
  p.meta_token = b.meta_token;
  p.meta_slot = b.meta_slot;
  p.meta_result_index = b.meta_result_index;
  p.expert_counts = b.expert_counts;
  p.source_record_counts = b.source_record_counts;
  p.source_route_counts = b.source_route_counts;
  p.source_active_rows = b.source_active_rows;
  p.expert_row_offsets = b.expert_row_offsets;
  p.expert_scatter_offsets = b.expert_scatter_offsets;
  p.task_expert = b.task_expert;
  p.task_source_rank = b.task_source_rank;
  p.task_owner_rank = b.task_owner_rank;
  p.task_local_expert = b.task_local_expert;
  p.task_pool_row = b.task_pool_row;
  p.task_m_local = b.task_m_local;
  p.task_valid_m = b.task_valid_m;
  p.grouped_layout = canonical.grouped_layout;
  p.total_valid_routes = b.total_valid_routes;
  p.total_padded_rows = b.total_padded_rows;
  p.total_m_tasks = b.total_m_tasks;
  p.histogram_done = b.histogram_done;
  p.prefix_done = b.prefix_done;
  p.w1_weight = reinterpret_cast<__nv_fp8_e4m3*>(b.w1_weight);
  p.w1_bf16 = canonical.w1_bf16;
  p.intermediate_fp8 = b.intermediate;
  p.intermediate_sfa_u8 = reinterpret_cast<std::uint8_t*>(b.intermediate_sf);
  p.requant_groups_done = b.requant_completed;
  p.w2_weight = reinterpret_cast<__nv_fp8_e4m3*>(b.w2_weight);
  p.w2_bf16 = canonical.w2_bf16;
  p.final_output = final_output;
  p.result_out = result_out.pointer;
  p.result_out_window = result_out.window;
  p.result_inbox = result_inbox.pointer;
  p.result_inbox_window = result_inbox.window;
  p.tensor_map_buffer = canonical.tensor_map_buffer;
  p.w1_tensor_map_a = canonical.descriptor_storage + 0;
  p.w1_tensor_map_b = canonical.descriptor_storage + 1;
  p.w1_tensor_map_sfa = canonical.descriptor_storage + 2;
  p.w1_tensor_map_sfb = canonical.descriptor_storage + 3;
  p.w1_tensor_map_d = canonical.descriptor_storage + 4;
  p.w2_tensor_map_a = canonical.descriptor_storage + 5;
  p.w2_tensor_map_b = canonical.descriptor_storage + 6;
  p.w2_tensor_map_sfa = canonical.descriptor_storage + 7;
  p.w2_tensor_map_sfb = canonical.descriptor_storage + 8;
  p.w2_tensor_map_d = canonical.descriptor_storage + 9;
  p.w1_warp_done = canonical.w1_warp_done;
  p.w1_task_ready = canonical.w1_task_ready;
  p.w1_next_tile = canonical.w1_next_tile;
  p.w1_tiles_completed = canonical.w1_tiles_completed;
  p.epilogue_claimed = canonical.epilogue_claimed;
  p.epilogue_completed = canonical.epilogue_completed;
  p.w2_task_ready = canonical.w2_task_ready;
  p.w2_task_claimed = canonical.w2_task_claimed;
  p.w2_tile_warp_done = canonical.w2_tile_warp_done;
  p.w2_tiles_completed = canonical.w2_tiles_completed;
  p.source_w2_done = canonical.source_w2_done;
  p.combine_ready = canonical.combine_ready;
  p.combine_ctas_done = canonical.combine_ctas_done;
  p.epoch_done = canonical.epoch_done;
  p.ready_audit_counts = canonical.ready_audit_counts;
  p.worker_task = canonical.worker_task;
  p.worker_n = canonical.worker_n;
  p.combine_ack_signal_base_scratch =
      canonical.combine_ack_signal_base_scratch;
  p.ack_out = ack_out.pointer;
  p.ack_out_window = ack_out.window;
  p.ack_inbox = ack_inbox.pointer;
  p.ack_inbox_window = ack_inbox.window;
#if CAKE_MOE_PHASE_TRACE
  p.phase_ns = canonical.phase_ns;
  p.phase_count = canonical.phase_count;
#endif
  return p;
}

void allocate_canonical_buffers(DeviceBuffers& b, CanonicalBuffers& c,
                                int active_rows, OraclePattern pattern) {
  const std::size_t topk_pairs =
      static_cast<std::size_t>(kMaxRows) * kTopK;
  device_alloc(&b.topk_idx, topk_pairs * 2);
  device_alloc(&b.topk_weights, topk_pairs);
  device_alloc(&b.x, static_cast<std::size_t>(kMaxRows) * kHidden);
  device_alloc(&b.x_sf, static_cast<std::size_t>(kMaxRows) * kW1KBlocks);
  device_alloc(&b.owner_record_counts, kPhysicalRanks);
  device_alloc(&b.owner_route_counts, kPhysicalRanks);
  device_alloc(&b.route_result_index, topk_pairs);
  device_alloc(&b.protocol_error, 1);
  device_alloc(&b.dispatch_signal_base_scratch, kPhysicalRanks);
  device_alloc(&b.result_signal_base_scratch, kPhysicalRanks);
  device_alloc(&b.pool_fp8,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) * kHidden);
  device_alloc(&b.pool_sf,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) *
                   kW1KBlocks);
  device_alloc(&b.routing_weight_pool, kCanonicalMaxPaddedRows);
  device_alloc(&b.meta_source_rank, kCanonicalMaxPaddedRows);
  device_alloc(&b.meta_token, kCanonicalMaxPaddedRows);
  device_alloc(&b.meta_slot, kCanonicalMaxPaddedRows);
  device_alloc(&b.meta_result_index, kCanonicalMaxPaddedRows);
  device_alloc(&b.expert_counts, kLocalExperts);
  device_alloc(&b.source_record_counts, kPhysicalRanks);
  device_alloc(&b.source_route_counts, kPhysicalRanks);
  device_alloc(&b.source_active_rows, kPhysicalRanks);
  device_alloc(&b.expert_row_offsets, kLocalExperts);
  device_alloc(&b.expert_scatter_offsets, kLocalExperts);
  device_alloc(&b.task_expert, kCanonicalMaxTasks);
  device_alloc(&b.task_source_rank, kCanonicalMaxTasks);
  device_alloc(&b.task_owner_rank, kCanonicalMaxTasks);
  device_alloc(&b.task_local_expert, kCanonicalMaxTasks);
  device_alloc(&b.task_pool_row, kCanonicalMaxTasks);
  device_alloc(&b.task_m_local, kCanonicalMaxTasks);
  device_alloc(&b.task_valid_m, kCanonicalMaxTasks);
  device_alloc(&b.total_valid_routes, 1);
  device_alloc(&b.total_padded_rows, 1);
  device_alloc(&b.total_m_tasks, 1);
  device_alloc(&b.histogram_done, 1);
  device_alloc(&b.prefix_done, 1);
  device_alloc(&b.w1_weight, kW1WeightBytes);
  device_alloc(&b.w1_weight_sf, kW1WeightSfWords);
  device_alloc(&b.intermediate,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) *
                   kIntermediate);
  device_alloc(&b.intermediate_sf,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) *
                   (kIntermediate / 32));
  device_alloc(&b.requant_completed, 1);
  device_alloc(&b.w2_weight, kW2WeightBytes);
  device_alloc(&b.w2_weight_sf, kW2WeightSfWords);
  device_alloc(&b.final_allocation,
               kFinalElements + 2 * kOutputGuardElements);
  const std::size_t reference_routes =
      static_cast<std::size_t>(active_rows) * kTopK;
  if (pattern != OraclePattern::kDenseExternal) {
    device_alloc(&b.reference_output,
                 static_cast<std::size_t>(active_rows) * kOutput);
    device_alloc(&b.reference_intermediate,
                 reference_routes * kIntermediate);
    device_alloc(&b.reference_intermediate_sf,
                 reference_routes * (kIntermediate / 32));
    device_alloc(&b.reference_partials, reference_routes * kOutput);
  }

  device_alloc(&c.grouped_layout, kCanonicalMaxPaddedRows);
  device_alloc(&c.w1_bf16,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) *
                   kW1PhysicalN);
  device_alloc(&c.w2_bf16,
               static_cast<std::size_t>(kCanonicalMaxPaddedRows) * kOutput);
  device_alloc(&c.descriptor_storage, kCanonicalDescriptorCount);
  device_alloc(&c.ready_params, 1);
  device_alloc(&c.stage_mismatches, 3);
  device_alloc(&c.w2_signed_zero_differences, 1);
  device_alloc(&c.w1_warp_done, kCanonicalMaxTasks);
  device_alloc(&c.w1_task_ready, kCanonicalMaxTasks);
  device_alloc(&c.w1_next_tile, 1);
  device_alloc(&c.w1_tiles_completed, 1);
  device_alloc(&c.epilogue_claimed, kCanonicalMaxTasks);
  device_alloc(&c.epilogue_completed, 1);
  device_alloc(&c.w2_task_ready, kCanonicalMaxTasks);
  device_alloc(&c.w2_task_claimed, kCanonicalMaxTasks);
  device_alloc(&c.w2_tile_warp_done, kCanonicalMaxTasks);
  device_alloc(&c.w2_tiles_completed, 1);
  device_alloc(&c.source_w2_done, kPhysicalRanks);
  device_alloc(&c.combine_ready, 1);
  device_alloc(&c.combine_ctas_done, 1);
  device_alloc(&c.epoch_done, 1);
  device_alloc(&c.ready_audit_counts, 16);
  device_alloc(&c.worker_task, kCanonicalReadyCtas - 1);
  device_alloc(&c.worker_n, kCanonicalReadyCtas - 1);
  device_alloc(&c.combine_ack_signal_base_scratch, kPhysicalRanks);
#if CAKE_MOE_PHASE_TRACE
  device_alloc(&c.phase_ns, kCanonicalReadyCtas * cake_moe::trace::kPhaseCount);
  device_alloc(&c.phase_count, kCanonicalReadyCtas * cake_moe::trace::kPhaseCount);
#endif
}

#if CAKE_MOE_PHASE_TRACE
#include <mutex>

// One JSON line per epoch: for each phase the busiest CTA, the mean over CTAs
// that entered it, and the entry count. The busiest CTA is the one that matters
// for the critical path; the mean exposes imbalance across the 110 CTAs.
void report_phase_trace(const CanonicalBuffers& canonical, int rank,
                        int epoch_index) {
  constexpr int kSlots = kCanonicalReadyCtas * cake_moe::trace::kPhaseCount;
  std::vector<unsigned long long> ns(kSlots);
  std::vector<unsigned int> counts(kSlots);
  CUDA_CHECK(cudaMemcpy(ns.data(), canonical.phase_ns,
                        sizeof(unsigned long long) * kSlots,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(counts.data(), canonical.phase_count,
                        sizeof(unsigned int) * kSlots, cudaMemcpyDeviceToHost));

  char line[256];
  std::string out;
  std::snprintf(line, sizeof(line),
                "PHASE_TRACE_JSON={\"rank\":%d,\"epoch\":%d,\"ctas\":%d,"
                "\"phases\":[", rank, epoch_index, kCanonicalReadyCtas);
  out += line;
  for (int phase = 0; phase < cake_moe::trace::kPhaseCount; ++phase) {
    unsigned long long peak = 0;
    unsigned long long total = 0;
    unsigned long long entries = 0;
    int active_ctas = 0;
    for (int cta = 0; cta < kCanonicalReadyCtas; ++cta) {
      const int slot = cta * cake_moe::trace::kPhaseCount + phase;
      if (counts[slot] == 0) continue;
      ++active_ctas;
      peak = std::max(peak, ns[slot]);
      total += ns[slot];
      entries += counts[slot];
    }
    std::snprintf(line, sizeof(line),
                  "%s{\"phase\":\"%s\",\"peak_cta_ms\":%.6f,"
                  "\"mean_cta_ms\":%.6f,\"active_ctas\":%d,\"entries\":%llu}",
                  phase == 0 ? "" : ",", cake_moe::trace::phase_name(phase),
                  peak / 1e6, active_ctas ? total / 1e6 / active_ctas : 0.0,
                  active_ctas, entries);
    out += line;
  }
  out += "]}\n";
  // Ranks are threads in one process, so the record is emitted as one write.
  static std::mutex trace_mutex;
  std::lock_guard<std::mutex> guard(trace_mutex);
  std::fwrite(out.data(), 1, out.size(), stdout);
  std::fflush(stdout);
}
#endif

// The requested GIN context count is a hint. Reading a context the
// communicator did not grant is an illegal device access rather than an error,
// so the transport plan is validated once, on the host, before any launch.
void require_granted_gin_contexts(const ncclDevComm& device_comm) {
  const int granted = static_cast<int>(device_comm.ginContextCount);
  const int required = 1 + std::max({cake_moe::kGinDispatchContext,
                                     cake_moe::kGinResultContext,
                                     cake_moe::kGinAckContext});
  if (granted < required) {
    std::fprintf(stderr,
                 "GIN transport plan needs %d contexts but the communicator "
                 "granted %d (requested %d, connections %d, signals %d, "
                 "railed contexts %d)\n",
                 required, granted, cake_moe::kGinContexts,
                 (int)device_comm.ginConnectionCount,
                 device_comm.ginSignalCount,
                 (int)device_comm.ginContextsRailed);
    std::exit(EXIT_FAILURE);
  }
}

void free_canonical_buffers(DeviceBuffers& b, CanonicalBuffers& c) {
#if CAKE_MOE_PHASE_TRACE
  cudaFree(c.phase_count);
  cudaFree(c.phase_ns);
#endif
  cudaFree(c.combine_ack_signal_base_scratch);
  cudaFree(c.worker_n);
  cudaFree(c.worker_task);
  cudaFree(c.ready_audit_counts);
  cudaFree(c.epoch_done);
  cudaFree(c.combine_ctas_done);
  cudaFree(c.combine_ready);
  cudaFree(c.source_w2_done);
  cudaFree(c.w2_tiles_completed);
  cudaFree(c.w2_tile_warp_done);
  cudaFree(c.w2_task_claimed);
  cudaFree(c.w2_task_ready);
  cudaFree(c.epilogue_completed);
  cudaFree(c.epilogue_claimed);
  cudaFree(c.w1_tiles_completed);
  cudaFree(c.w1_next_tile);
  cudaFree(c.w1_task_ready);
  cudaFree(c.w1_warp_done);
  cudaFree(c.w2_signed_zero_differences);
  cudaFree(c.stage_mismatches);
  cudaFree(c.ready_params);
  cudaFree(c.descriptor_storage);
  cudaFree(c.w2_bf16);
  cudaFree(c.w1_bf16);
  cudaFree(c.grouped_layout);
  free_buffers(b);
}

// Independent sparse full-dimension W1 oracle.  It consumes the dispatched A
// pool and fixture rule, never the donor output, and therefore also covers
// remote-source rows without reconstructing their original source tensor.
__global__ void canonical_compare_w1_sparse(
    std::uint8_t const* pool_fp8, std::uint32_t const* pool_sf,
    cutlass::bfloat16_t const* observed, int const* grouped_layout,
    int const* meta_source_rank, std::uint32_t* mismatches,
    int total_padded_rows, int rank, int world_size, OraclePattern pattern) {
  const unsigned long long count =
      static_cast<unsigned long long>(total_padded_rows) * kW1PhysicalN;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < count;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int physical_n = static_cast<int>(index % kW1PhysicalN);
    const int row = static_cast<int>(index / kW1PhysicalN);
    const int source_rank = meta_source_rank[row];
    // Padding rows have no route semantics.  Their BF16 signed-zero payload is
    // donor-internal and must not be mistaken for a numerical mismatch.
    if (source_rank == -1) continue;
    // Every value other than the canonical -1 padding sentinel denotes a
    // semantic row and must name a live source rank.  Add one fail-closed
    // mismatch per malformed row while keeping every valid remote-source row
    // in the full W1 comparison below.
    if (source_rank < -1 || source_rank >= world_size) {
      if (physical_n == 0) atomicAdd(&mismatches[0], 1u);
      continue;
    }
    const int local_expert = grouped_layout[row];
    if (local_expert < 0 || local_expert >= kLocalExperts) {
      if (physical_n == 0) atomicAdd(&mismatches[0], 1u);
      continue;
    }
    const int expert = rank * kLocalExperts + local_expert;
    const int k = sparse_w1_k(expert, physical_n);
    const int packed_word = k / kBlockK;
    const int k32 = (k % kBlockK) / 32;
    const std::uint8_t a_code =
        pool_fp8[static_cast<unsigned long long>(row) * kHidden + k];
    const std::uint32_t packed_sf =
        pool_sf[static_cast<unsigned long long>(packed_word) *
                    kCanonicalMaxPaddedRows +
                row];
    const std::uint8_t a_sf =
        static_cast<std::uint8_t>(packed_sf >> (8 * k32));
    const std::uint8_t b_code =
        weight_code(pattern, true, expert, physical_n, k);
    const std::uint8_t b_sf =
        weight_scale(expert, physical_n, k / 32);
    const float product = decode_fp8(a_code) * decode_fp4(b_code) *
                          decode_ue8m0(a_sf) * decode_ue8m0(b_sf);
    const __nv_bfloat16 expected = __float2bfloat16_rn(product);
    const std::uint16_t expected_bits =
        *reinterpret_cast<std::uint16_t const*>(&expected);
    const std::uint16_t observed_bits = *reinterpret_cast<
        std::uint16_t const*>(observed + index);
    if (expected_bits != observed_bits) atomicAdd(mismatches, 1u);
  }
}

// P1/R1 exact stage gate.  Multi-rank rows are still checked at W1 and final
// output; intermediate/W2 checks select the local-source rows for which the
// frozen independent reference buffer is resident on this rank.
__global__ void canonical_compare_local_stages(
    std::uint8_t const* intermediate,
    std::uint8_t const* intermediate_sf,
    cutlass::bfloat16_t const* w2_bf16,
    int const* meta_source_rank, int const* meta_token,
    int const* meta_slot, std::uint8_t const* reference_intermediate,
    std::uint8_t const* reference_intermediate_sf,
    __nv_bfloat16 const* reference_partials,
    std::uint32_t* mismatches, std::uint32_t* w2_signed_zero_differences,
    int total_padded_rows, int rank) {
  const unsigned long long max_elements =
      static_cast<unsigned long long>(total_padded_rows) * kOutput;
  for (unsigned long long index =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < max_elements;
       index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(index / kOutput);
    if (meta_source_rank[row] != rank) continue;
    const int token = meta_token[row];
    const int topk_slot = meta_slot[row];
    if (token < 0 || token >= kMaxRows || topk_slot < 0 ||
        topk_slot >= kTopK) {
      atomicAdd(&mismatches[1], 1u);
      continue;
    }
    const int route = token * kTopK + topk_slot;
    const int column = static_cast<int>(index -
                                        static_cast<unsigned long long>(row) *
                                            kOutput);
    const std::uint16_t observed_bits = *reinterpret_cast<
        std::uint16_t const*>(w2_bf16 + index);
    const std::uint16_t expected_bits = *reinterpret_cast<
        std::uint16_t const*>(reference_partials +
                             static_cast<unsigned long long>(route) *
                                 kOutput +
                             column);
    if (observed_bits != expected_bits) {
      const bool both_magnitude_zero =
          (observed_bits & 0x7fffu) == 0 &&
          (expected_bits & 0x7fffu) == 0;
      if (both_magnitude_zero)
        atomicAdd(w2_signed_zero_differences, 1u);
      else
        atomicAdd(&mismatches[2], 1u);
    }
    if (column < kIntermediate) {
      const unsigned long long actual =
          static_cast<unsigned long long>(row) * kIntermediate + column;
      const unsigned long long expected =
          static_cast<unsigned long long>(route) * kIntermediate + column;
      if (intermediate[actual] != reference_intermediate[expected])
        atomicAdd(&mismatches[1], 1u);
    }
    if (column < kIntermediate / 32) {
      const unsigned long long actual =
          ((static_cast<unsigned long long>(column >> 2) *
                kCanonicalMaxPaddedRows +
            row) *
               4ull) +
          static_cast<unsigned long long>(column & 3);
      const unsigned long long expected =
          static_cast<unsigned long long>(route) * (kIntermediate / 32) +
          column;
      if (intermediate_sf[actual] != reference_intermediate_sf[expected])
        atomicAdd(&mismatches[1], 1u);
    }
  }
}

void run_canonical_rank(int rank, int world_size, int local_device,
                        ncclUniqueId unique_id, int active_rows,
                        int mask_period, const std::string& route_mode,
                        const std::string& oracle_name) {
  CUDA_CHECK(cudaSetDevice(local_device));
  DRIVER_CHECK(cuInit(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, local_device));
  if (properties.major != 12 || properties.minor != 0) {
    std::fprintf(stderr, "rank %d requires SM120, observed %d.%d\n", rank,
                 properties.major, properties.minor);
    std::abort();
  }
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    local_device));
  if (sm_count != kCanonicalReadyCtas) {
    std::fprintf(stderr,
                 "canonical donor is fail-closed to 110 SMs, observed %d\n",
                 sm_count);
    std::abort();
  }

  ncclComm_t comm{};
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, unique_id, rank));
  ncclCommProperties_t comm_properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(comm, &comm_properties));
  if (!comm_properties.deviceApiSupport ||
      comm_properties.ginType == NCCL_GIN_TYPE_NONE) {
    std::fprintf(stderr, "rank %d lacks NCCL Device API/GIN support\n", rank);
    std::abort();
  }

  const std::size_t header_count =
      kPhysicalRanks * kRingSlots * kHeaderWords;
  auto dispatch_header_out = allocate_window<int>(comm, header_count);
  auto dispatch_payload_out =
      allocate_window<std::uint8_t>(comm, kDispatchWindowBytes);
  auto dispatch_header_inbox = allocate_window<int>(comm, header_count);
  auto dispatch_payload_inbox =
      allocate_window<std::uint8_t>(comm, kDispatchWindowBytes);
  auto result_out =
      allocate_window<__nv_bfloat16>(comm, kResultWindowElements);
  auto result_inbox =
      allocate_window<__nv_bfloat16>(comm, kResultWindowElements);
  auto ack_out = allocate_window<std::uint8_t>(comm, kPhysicalRanks);
  auto ack_inbox = allocate_window<std::uint8_t>(comm, kPhysicalRanks);
  CUDA_CHECK(cudaMemset(dispatch_header_out.pointer, 0,
                        dispatch_header_out.bytes));
  CUDA_CHECK(cudaMemset(dispatch_payload_out.pointer, 0,
                        dispatch_payload_out.bytes));
  CUDA_CHECK(cudaMemset(dispatch_header_inbox.pointer, 0,
                        dispatch_header_inbox.bytes));
  CUDA_CHECK(cudaMemset(dispatch_payload_inbox.pointer, 0,
                        dispatch_payload_inbox.bytes));
  CUDA_CHECK(cudaMemset(result_out.pointer, 0, result_out.bytes));
  CUDA_CHECK(cudaMemset(result_inbox.pointer, 0, result_inbox.bytes));
  CUDA_CHECK(cudaMemset(ack_out.pointer, 0, ack_out.bytes));
  CUDA_CHECK(cudaMemset(ack_inbox.pointer, 0, ack_inbox.bytes));

  const OraclePattern pattern = oracle_pattern_from_string(oracle_name);
  if (pattern == OraclePattern::kDenseExternal) {
    std::fprintf(stderr,
                 "canonical fused ready chunk8 dense_external oracle is not installed; "
                 "fail closed\n");
    std::abort();
  }
  const std::size_t topk_pairs =
      static_cast<std::size_t>(kMaxRows) * kTopK;
  std::vector<int> host_topk(topk_pairs * 2, -1);
  std::vector<float> host_weights(topk_pairs, 0.0f);
  std::array<unsigned int, kPhysicalRanks> expected_owner_records{};
  std::array<unsigned int, kPhysicalRanks> expected_owner_routes{};
  for (int token = 0; token < active_rows; ++token) {
    std::array<bool, kPhysicalRanks> token_has_owner{};
    for (int slot = 0; slot < kTopK; ++slot) {
      const int route = token * kTopK + slot;
      if (mask_period > 0 && route % mask_period == 0) continue;
      const int expert =
          route_expert(token, slot, rank, world_size, route_mode);
      host_topk[route * 2] = expert;
      host_topk[route * 2 + 1] = 0;
      host_weights[route] = route_weight(slot);
      const int owner = expert / kLocalExperts;
      ++expected_owner_routes[owner];
      token_has_owner[owner] = true;
    }
    for (int owner = 0; owner < kPhysicalRanks; ++owner)
      expected_owner_records[owner] += token_has_owner[owner] ? 1u : 0u;
  }

  int expected_received_routes = 0;
  for (int source = 0; source < world_size; ++source) {
    for (int token = 0; token < active_rows; ++token) {
      for (int slot = 0; slot < kTopK; ++slot) {
        const int route = token * kTopK + slot;
        if (mask_period > 0 && route % mask_period == 0) continue;
        const int expert =
            route_expert(token, slot, source, world_size, route_mode);
        expected_received_routes += expert / kLocalExperts == rank;
      }
    }
  }

  DeviceBuffers b;
  CanonicalBuffers canonical;
  allocate_canonical_buffers(b, canonical, active_rows, pattern);
  CUDA_CHECK(cudaMemcpy(b.topk_idx, host_topk.data(),
                        host_topk.size() * sizeof(host_topk[0]),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.topk_weights, host_weights.data(),
                        host_weights.size() * sizeof(host_weights[0]),
                        cudaMemcpyHostToDevice));
  constexpr int kInitBlocks = 4096;
  constexpr int kInitThreads = 256;
  initialize_x<<<kInitBlocks, kInitThreads>>>(b.x, b.x_sf, active_rows,
                                              rank, pattern);
  initialize_fp4_weights<<<kInitBlocks, kInitThreads>>>(
      b.w1_weight, kW1WeightBytes, rank, pattern, true);
  initialize_weight_scales<<<kInitBlocks, kInitThreads>>>(
      b.w1_weight_sf, kW1WeightSfWords, rank, true);
  initialize_fp4_weights<<<kInitBlocks, kInitThreads>>>(
      b.w2_weight, kW2WeightBytes, rank, pattern, false);
  initialize_weight_scales<<<kInitBlocks, kInitThreads>>>(
      b.w2_weight_sf, kW2WeightSfWords, rank, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const CanonicalMaps maps = encode_canonical_maps(b, canonical);
  upload_canonical_maps(canonical, maps);

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = kGinContexts;
  // Contexts only get their own connection when they are exclusive; sharing
  // one connection leaves every context above zero without a queue pair.
  requirements.ginExclusiveContexts = kGinContexts > 1;
  requirements.ginSignalCount = 24;
  requirements.worldGinBarrierCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  requirements.ginStrongSignalsRequired = true;
  ncclDevComm device_comm{};
  NCCL_CHECK(ncclDevCommCreate(comm, &requirements, &device_comm));
  // ginContextCount in the requirements is a hint, so the transport plan is
  // only legal once the communicator confirms it granted that many contexts.
  require_granted_gin_contexts(device_comm);
  device_alloc(&b.device_comm, 1);
  CUDA_CHECK(cudaMemcpy(b.device_comm, &device_comm, sizeof(device_comm),
                        cudaMemcpyHostToDevice));

  void* ready_kernel = reinterpret_cast<void*>(
      kernel_cake_sm120_production_canonical_fused_ready_chunk8);
  // The fused entry unions the qualified transport SMEM with the canonical
  // donor's 50,720-byte mainloop buffer.  Opt in before occupancy is queried.
  CUDA_CHECK(cudaFuncSetAttribute(
      ready_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kCanonicalReadySmem));
  const int ready_capacity = canonical_cooperative_capacity(
      ready_kernel, sm_count, kCanonicalThreads, kCanonicalReadySmem);
  if (ready_capacity < kCanonicalReadyCtas) {
    std::fprintf(stderr,
                 "rank %d ready cooperative capacity %d is below fixed "
                 "donor grid %d\n",
                 rank, ready_capacity, kCanonicalReadyCtas);
    std::abort();
  }

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  communicator_barrier(comm, stream, world_size);

  const int reference_routes = active_rows * kTopK;
  const int ref_groups = reference_routes * (kIntermediate / 32);
  const int ref_blocks =
      std::max(1, std::min(kInitBlocks, (ref_groups + 3) / 4));
  reference_w1_requant<<<ref_blocks, kThreads, 0, stream>>>(
      b.x, b.x_sf, b.topk_idx, b.topk_weights, b.reference_intermediate,
      b.reference_intermediate_sf, active_rows, pattern);
  reference_w2<<<kInitBlocks, kInitThreads, 0, stream>>>(
      b.topk_idx, b.reference_intermediate, b.reference_intermediate_sf,
      b.reference_partials, active_rows, pattern);
  reference_combine<<<kInitBlocks, kInitThreads, 0, stream>>>(
      b.topk_idx, b.reference_partials, b.reference_output, active_rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<std::uint16_t> host_reference(
      static_cast<std::size_t>(active_rows) * kOutput);
  CUDA_CHECK(cudaMemcpy(host_reference.data(), b.reference_output,
                        host_reference.size() * sizeof(host_reference[0]),
                        cudaMemcpyDeviceToHost));

  std::array<unsigned long long, kPhysicalRanks> initial_dispatch{};
  std::array<unsigned long long, kPhysicalRanks> initial_result{};
  std::array<unsigned long long, kPhysicalRanks> initial_ack{};
  int owner_mismatches = 0;
  int counter_mismatches = 0;
  int signal_mismatches = 0;
  int ack_signal_mismatches = 0;
  int ready_audit_mismatches = 0;
  int output_mismatches = 0;
  int guard_mismatches = 0;
  double max_abs_error = 0.0;
  std::array<int, kEpochs> epoch_slots{};
  std::array<int, kEpochs> epoch_routes{};
  std::array<std::uint32_t, 3> stage_mismatches{};
  std::array<std::array<std::uint32_t, 3>, kEpochs>
      stage_mismatches_per_epoch{};
  std::array<std::uint32_t, kEpochs>
      w2_signed_zero_differences_per_epoch{};
  std::array<std::array<unsigned int, 16>, kEpochs>
      ready_audit_per_epoch{};
  unsigned long long w2_signed_zero_differences = 0;
  int actual_production_launches = 0;
  int diagnostic_oracle_launches = 0;
  int zero_route_compute_skips = 0;
  CUDA_CHECK(cudaMemsetAsync(b.protocol_error, 0, sizeof(std::uint32_t),
                             stream));
  CUDA_CHECK(cudaMemsetAsync(result_out.pointer, 0, result_out.bytes,
                             stream));
  CUDA_CHECK(cudaMemsetAsync(result_inbox.pointer, 0, result_inbox.bytes,
                             stream));

  for (int epoch_index = 0; epoch_index < kEpochs; ++epoch_index) {
    const unsigned int epoch = static_cast<unsigned int>(epoch_index);
    epoch_slots[epoch_index] = epoch_index & 1;
    CUDA_CHECK(cudaMemsetAsync(
        b.final_allocation, 0xa5,
        (kFinalElements + 2 * kOutputGuardElements) *
            sizeof(__nv_bfloat16),
        stream));
    __nv_bfloat16* final_output =
        b.final_allocation + kOutputGuardElements;
    const CakeSm120CanonicalFusedReadyParams host_params =
        make_ready_params(
            b, canonical, rank, world_size, active_rows, epoch,
            dispatch_header_out, dispatch_payload_out,
            dispatch_header_inbox, dispatch_payload_inbox, result_out,
            result_inbox, ack_out, ack_inbox, final_output);
    CUDA_CHECK(cudaMemcpyAsync(canonical.ready_params, &host_params,
                               sizeof(host_params), cudaMemcpyHostToDevice,
                               stream));
#if CAKE_MOE_PHASE_TRACE
    CUDA_CHECK(cudaMemsetAsync(
        canonical.phase_ns, 0,
        sizeof(unsigned long long) * kCanonicalReadyCtas *
            cake_moe::trace::kPhaseCount, stream));
    CUDA_CHECK(cudaMemsetAsync(
        canonical.phase_count, 0,
        sizeof(unsigned int) * kCanonicalReadyCtas *
            cake_moe::trace::kPhaseCount, stream));
#endif
    void* ready_args[] = {&canonical.ready_params};
    static_assert(sizeof(ready_args) / sizeof(ready_args[0]) == 1);
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        ready_kernel, dim3(kCanonicalReadyCtas),
        dim3(kCanonicalThreads), ready_args, kCanonicalReadySmem,
        stream));
    ++actual_production_launches;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if CAKE_MOE_PHASE_TRACE
    report_phase_trace(canonical, rank, epoch_index);
#endif
    int total_routes = 0;
    int total_padded_rows = 0;
    int total_tasks = 0;
    CUDA_CHECK(cudaMemcpy(&total_routes, b.total_valid_routes, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_padded_rows, b.total_padded_rows, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_tasks, b.total_m_tasks, sizeof(int),
                          cudaMemcpyDeviceToHost));
    epoch_routes[epoch_index] = total_routes;
    counter_mismatches += total_routes != expected_received_routes;
    counter_mismatches += total_padded_rows != total_tasks * kCanonicalTaskM;

    std::array<unsigned int, kPhysicalRanks> observed_records{};
    std::array<unsigned int, kPhysicalRanks> observed_routes{};
    std::array<unsigned long long, kPhysicalRanks> observed_dispatch{};
    CUDA_CHECK(cudaMemcpy(observed_records.data(), b.owner_record_counts,
                          sizeof(observed_records), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_routes.data(), b.owner_route_counts,
                          sizeof(observed_routes), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_dispatch.data(),
                          b.dispatch_signal_base_scratch,
                          sizeof(observed_dispatch), cudaMemcpyDeviceToHost));
    for (int owner = 0; owner < kPhysicalRanks; ++owner) {
      owner_mismatches +=
          observed_records[owner] != expected_owner_records[owner];
      owner_mismatches +=
          observed_routes[owner] != expected_owner_routes[owner];
    }
    if (epoch_index == 0) initial_dispatch = observed_dispatch;
    for (int source = 0; source < kPhysicalRanks; ++source) {
      const unsigned long long delta = source < world_size
                                           ? 2ull * epoch_index
                                           : 0ull;
      signal_mismatches +=
          observed_dispatch[source] - initial_dispatch[source] != delta;
    }

    zero_route_compute_skips += total_padded_rows == 0;

    std::array<unsigned int, 16> observed_ready_audit{};
    CUDA_CHECK(cudaMemcpy(observed_ready_audit.data(),
                          canonical.ready_audit_counts,
                          sizeof(observed_ready_audit),
                          cudaMemcpyDeviceToHost));
    ready_audit_per_epoch[epoch_index] = observed_ready_audit;
    const unsigned int expected_w1_tiles =
        static_cast<unsigned int>(total_tasks) *
        static_cast<unsigned int>(cake_moe::kW1TilesPerTask);
    const unsigned int expected_w2_tiles =
        static_cast<unsigned int>(total_tasks) *
        static_cast<unsigned int>(cake_moe::kW2TilesPerTask);
    const unsigned int expected_route_tiles =
        static_cast<unsigned int>(total_routes) *
        static_cast<unsigned int>(cake_moe::kW2TilesPerTask);
    ready_audit_mismatches += observed_ready_audit[0] !=
                              static_cast<unsigned int>(total_tasks);
    ready_audit_mismatches += observed_ready_audit[1] != expected_w1_tiles;
    ready_audit_mismatches += observed_ready_audit[2] != expected_w1_tiles;
    ready_audit_mismatches += observed_ready_audit[3] !=
                              static_cast<unsigned int>(total_tasks);
    ready_audit_mismatches += observed_ready_audit[4] != expected_w2_tiles;
    ready_audit_mismatches += observed_ready_audit[5] != expected_w2_tiles;
    ready_audit_mismatches += observed_ready_audit[6] != expected_route_tiles;
    ready_audit_mismatches += observed_ready_audit[7] != expected_route_tiles;
    ready_audit_mismatches += observed_ready_audit[8] < expected_w1_tiles;
    ready_audit_mismatches +=
        observed_ready_audit[8] > expected_w1_tiles + 108u;
    ready_audit_mismatches += observed_ready_audit[9] != 109u;
    ready_audit_mismatches += observed_ready_audit[10] != 110u;

    std::array<unsigned long long, kPhysicalRanks> observed_ack{};
    CUDA_CHECK(cudaMemcpy(observed_ack.data(),
                          canonical.combine_ack_signal_base_scratch,
                          sizeof(observed_ack), cudaMemcpyDeviceToHost));
    if (epoch_index == 0) initial_ack = observed_ack;
    for (int source = 0; source < kPhysicalRanks; ++source) {
      const unsigned long long delta =
          source < world_size
              ? static_cast<unsigned long long>(epoch_index)
              : 0ull;
      ack_signal_mismatches +=
          observed_ack[source] - initial_ack[source] != delta;
    }

    // Diagnostics are deliberately outside the single production launch.
    // They cannot mutate any production buffer or weaken the
    // dispatch/result slot-credit boundary.
    CUDA_CHECK(cudaMemsetAsync(canonical.stage_mismatches, 0,
                               3 * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(canonical.w2_signed_zero_differences, 0,
                               sizeof(std::uint32_t), stream));
    if (total_padded_rows > 0) {
      canonical_compare_w1_sparse<<<4096, 256, 0, stream>>>(
          b.pool_fp8, b.pool_sf, canonical.w1_bf16,
          canonical.grouped_layout, b.meta_source_rank,
          canonical.stage_mismatches, total_padded_rows, rank, world_size,
          pattern);
      canonical_compare_local_stages<<<4096, 256, 0, stream>>>(
          b.intermediate, b.intermediate_sf, canonical.w2_bf16,
          b.meta_source_rank, b.meta_token, b.meta_slot,
          b.reference_intermediate, b.reference_intermediate_sf,
          b.reference_partials, canonical.stage_mismatches,
          canonical.w2_signed_zero_differences, total_padded_rows, rank);
      diagnostic_oracle_launches += 2;
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    unsigned int requant_completed = 0;
    CUDA_CHECK(cudaMemcpy(&requant_completed, b.requant_completed,
                          sizeof(requant_completed), cudaMemcpyDeviceToHost));
    counter_mismatches +=
        requant_completed !=
        static_cast<unsigned int>(total_padded_rows * (kIntermediate / 32));
    std::array<std::uint32_t, 3> observed_stage_mismatches{};
    CUDA_CHECK(cudaMemcpy(observed_stage_mismatches.data(),
                          canonical.stage_mismatches,
                          sizeof(observed_stage_mismatches),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &w2_signed_zero_differences_per_epoch[epoch_index],
        canonical.w2_signed_zero_differences, sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost));
    w2_signed_zero_differences +=
        w2_signed_zero_differences_per_epoch[epoch_index];
    stage_mismatches_per_epoch[epoch_index] = observed_stage_mismatches;
    for (int stage = 0; stage < 3; ++stage)
      stage_mismatches[stage] += observed_stage_mismatches[stage];

    std::array<unsigned long long, kPhysicalRanks> observed_result{};
    CUDA_CHECK(cudaMemcpy(observed_result.data(),
                          b.result_signal_base_scratch,
                          sizeof(observed_result), cudaMemcpyDeviceToHost));
    if (epoch_index == 0) initial_result = observed_result;
    for (int owner = 0; owner < kPhysicalRanks; ++owner) {
      const unsigned long long delta = owner < world_size
                                           ? static_cast<unsigned long long>(
                                                 epoch_index)
                                           : 0ull;
      signal_mismatches +=
          observed_result[owner] - initial_result[owner] != delta;
    }

    const std::size_t output_elements =
        static_cast<std::size_t>(active_rows) * kOutput;
    std::vector<std::uint16_t> observed(output_elements);
    CUDA_CHECK(cudaMemcpy(observed.data(), final_output,
                          observed.size() * sizeof(observed[0]),
                          cudaMemcpyDeviceToHost));
    std::array<std::uint16_t, kOutputGuardElements> guard{};
    CUDA_CHECK(cudaMemcpy(guard.data(), b.final_allocation, sizeof(guard),
                          cudaMemcpyDeviceToHost));
    for (std::uint16_t bits : guard)
      guard_mismatches += bits != kOutputGuardBits;
    CUDA_CHECK(cudaMemcpy(
        guard.data(),
        b.final_allocation + kOutputGuardElements + kFinalElements,
        sizeof(guard), cudaMemcpyDeviceToHost));
    for (std::uint16_t bits : guard)
      guard_mismatches += bits != kOutputGuardBits;
    for (std::size_t i = 0; i < observed.size(); ++i) {
      output_mismatches += observed[i] != host_reference[i];
      __nv_bfloat16 actual{};
      __nv_bfloat16 expected{};
      std::memcpy(&actual, &observed[i], sizeof(actual));
      std::memcpy(&expected, &host_reference[i], sizeof(expected));
      max_abs_error = std::max(
          max_abs_error,
          std::abs(static_cast<double>(__bfloat162float(actual)) -
                   static_cast<double>(__bfloat162float(expected))));
    }
  }

  unsigned int protocol_error = 0;
  CUDA_CHECK(cudaMemcpy(&protocol_error, b.protocol_error,
                        sizeof(protocol_error), cudaMemcpyDeviceToHost));
  const int ring_mismatches =
      epoch_slots != std::array<int, kEpochs>{0, 1, 0};
  const int launch_mismatches = actual_production_launches != kEpochs;
  const int failures = static_cast<int>(protocol_error != 0) +
                       owner_mismatches + counter_mismatches +
                       signal_mismatches + ack_signal_mismatches +
                       ready_audit_mismatches +
                       static_cast<int>(stage_mismatches[0] != 0) +
                       static_cast<int>(stage_mismatches[1] != 0) +
                       static_cast<int>(stage_mismatches[2] != 0) +
                       output_mismatches + guard_mismatches +
                       ring_mismatches + launch_mismatches;
  std::printf(
      "RANK_RESULT_JSON={\"kind\":\"cake_sm120_production_canonical_fused_ready_chunk8\","
      "\"rank\":%d,\"world_size\":%d,\"active_rows\":%d,"
      "\"launch_count_per_epoch\":1,\"kernel_count\":1,"
      "\"single_entry\":true,\"full_shape\":true,"
      "\"direct_canonical_donor\":true,\"oracle\":\"%s\","
      "\"route_mode\":\"%s\",\"mask_period\":%d,"
      "\"activation_clamp\":10,\"fast_math\":true,"
      "\"activation_scale_granularity_k\":32,"
      "\"epoch_slots\":[%d,%d,%d],"
      "\"epoch_route_totals\":[%d,%d,%d],"
      "\"expected_received_routes\":%d,\"protocol_error\":%u,"
      "\"owner_mismatches\":%d,\"counter_mismatches\":%d,"
      "\"signal_mismatches\":%d,"
      "\"ack_signal_mismatches\":%d,"
      "\"ready_audit_mismatches\":%d,"
      "\"w1_bf16_mismatches\":%u,"
      "\"requant_fp8_sf_mismatches\":%u,"
      "\"w2_bf16_partial_mismatches\":%u,"
      "\"w2_signed_zero_differences\":%llu,"
      "\"w2_signed_zero_differences_per_epoch\":[%u,%u,%u],"
      "\"stage_mismatches_per_epoch\":[[%u,%u,%u],[%u,%u,%u],"
      "[%u,%u,%u]],"
      "\"ready_audit_per_epoch\":[[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u],"
      "[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u],"
      "[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u]],"
      "\"output_mismatches\":%d,\"output_guard_mismatches\":%d,"
      "\"max_abs_error\":%.9g,\"exact_bf16_equal\":%s,"
      "\"stage_oracle_installed\":true,"
      "\"ready_cooperative_capacity\":%d,"
      "\"requested_ctas\":110,\"actual_ctas\":110,"
      "\"threads_per_cta\":384,\"dynamic_smem_bytes\":94208,"
      "\"actual_production_launches\":%d,"
      "\"launch_mismatches\":%d,"
      "\"diagnostic_oracle_launches\":%d,"
      "\"zero_route_compute_skips\":%d,"
      "\"gin_signal_count\":24,\"post_combine_ack\":true,"
      "\"service_cta\":0,\"worker_ctas\":109,"
      "\"one_pointer_params\":true,"
      "\"descriptor_storage\":\"global_aligned\","
      "\"barrier_ordered\":false,\"ready_driven\":true,\"chunked_task_claim\":true,\"chunk_physical_n128_tiles\":8,\"w1_chunks_per_task\":8,\"w2_chunks_per_task\":4,\"task_major_chunk_issuance\":true,\"w1_warmup_tasks\":27,\"early_w2_worker_limit\":27,\"forced_w1_opportunity_after_each_early_w2_chunk\":true,"
      "\"runtime_register_repartition_qualified\":false,"
      "\"resource_qualified\":false,"
      "\"production_compute_comparable\":false,"
      "\"functional_qualified\":false,\"status\":\"%s\"}\n",
      rank, world_size, active_rows, oracle_name.c_str(), route_mode.c_str(),
      mask_period, epoch_slots[0], epoch_slots[1], epoch_slots[2],
      epoch_routes[0], epoch_routes[1], epoch_routes[2],
      expected_received_routes, protocol_error, owner_mismatches,
      counter_mismatches, signal_mismatches, ack_signal_mismatches,
      ready_audit_mismatches, stage_mismatches[0],
      stage_mismatches[1], stage_mismatches[2],
      w2_signed_zero_differences,
      w2_signed_zero_differences_per_epoch[0],
      w2_signed_zero_differences_per_epoch[1],
      w2_signed_zero_differences_per_epoch[2],
      stage_mismatches_per_epoch[0][0], stage_mismatches_per_epoch[0][1],
      stage_mismatches_per_epoch[0][2], stage_mismatches_per_epoch[1][0],
      stage_mismatches_per_epoch[1][1], stage_mismatches_per_epoch[1][2],
      stage_mismatches_per_epoch[2][0], stage_mismatches_per_epoch[2][1],
      stage_mismatches_per_epoch[2][2],
      ready_audit_per_epoch[0][0], ready_audit_per_epoch[0][1],
      ready_audit_per_epoch[0][2], ready_audit_per_epoch[0][3],
      ready_audit_per_epoch[0][4], ready_audit_per_epoch[0][5],
      ready_audit_per_epoch[0][6], ready_audit_per_epoch[0][7],
      ready_audit_per_epoch[0][8], ready_audit_per_epoch[0][9],
      ready_audit_per_epoch[0][10], ready_audit_per_epoch[1][0],
      ready_audit_per_epoch[1][1], ready_audit_per_epoch[1][2],
      ready_audit_per_epoch[1][3], ready_audit_per_epoch[1][4],
      ready_audit_per_epoch[1][5], ready_audit_per_epoch[1][6],
      ready_audit_per_epoch[1][7], ready_audit_per_epoch[1][8],
      ready_audit_per_epoch[1][9], ready_audit_per_epoch[1][10],
      ready_audit_per_epoch[2][0], ready_audit_per_epoch[2][1],
      ready_audit_per_epoch[2][2], ready_audit_per_epoch[2][3],
      ready_audit_per_epoch[2][4], ready_audit_per_epoch[2][5],
      ready_audit_per_epoch[2][6], ready_audit_per_epoch[2][7],
      ready_audit_per_epoch[2][8], ready_audit_per_epoch[2][9],
      ready_audit_per_epoch[2][10], output_mismatches, guard_mismatches,
      max_abs_error,
      output_mismatches == 0 ? "true" : "false",
      ready_capacity, actual_production_launches, launch_mismatches,
      diagnostic_oracle_launches,
      zero_route_compute_skips,
      failures == 0 ? "pass" : "fail");
  std::fflush(stdout);
  if (failures != 0) g_failures.fetch_add(1);

  communicator_barrier(comm, stream, world_size);
  CUDA_CHECK(cudaStreamDestroy(stream));
  free_canonical_buffers(b, canonical);
  NCCL_CHECK(ncclDevCommDestroy(comm, &device_comm));
  destroy_window(comm, ack_inbox);
  destroy_window(comm, ack_out);
  destroy_window(comm, result_inbox);
  destroy_window(comm, result_out);
  destroy_window(comm, dispatch_payload_inbox);
  destroy_window(comm, dispatch_header_inbox);
  destroy_window(comm, dispatch_payload_out);
  destroy_window(comm, dispatch_header_out);
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

}  // namespace

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  const int active_rows = static_cast<int>(
      parse_long_env("CAKE_ACTIVE_ROWS", 1, 1, kMaxRows));
  const int mask_period = static_cast<int>(
      parse_long_env("CAKE_MASK_PERIOD", 0, 0, kMaxRows * kTopK));
  const std::string route_mode = parse_choice_env(
      "CAKE_ROUTE_MODE", "balanced", {"balanced", "skewed", "empty"});
  const std::string oracle_name = parse_choice_env(
      "CAKE_ORACLE", "distinct_k32", {"zero", "analytic", "distinct_k32"});

  const char* rank_text = std::getenv("RANK");
  if (rank_text == nullptr) rank_text = std::getenv("SLURM_PROCID");
  const char* world_text = std::getenv("WORLD_SIZE");
  if (world_text == nullptr) world_text = std::getenv("SLURM_NTASKS");
  if (rank_text != nullptr && world_text != nullptr) {
    const int rank = std::atoi(rank_text);
    const int world_size = std::atoi(world_text);
    const int local_device = static_cast<int>(
        parse_long_env("LOCAL_DEVICE", 0, 0, std::max(0, device_count - 1)));
    const char* id_path = std::getenv("NCCL_UNIQUE_ID_FILE");
    if (!is_supported_world_size(world_size) || rank < 0 ||
        rank >= world_size || id_path == nullptr || id_path[0] == '\0') {
      std::fprintf(stderr, "invalid canonical fused ready chunk8 process launch\n");
      return EXIT_FAILURE;
    }
    const ncclUniqueId unique_id = load_or_create_unique_id(id_path, rank);
    run_canonical_rank(rank, world_size, local_device, unique_id, active_rows,
                       mask_period, route_mode, oracle_name);
    const int failures = g_failures.load();
    std::printf(
        "RESULT_JSON={\"kind\":\"cake_sm120_production_canonical_fused_ready_chunk8\","
        "\"rank\":%d,\"world_size\":%d,\"launch_count\":1,"
        "\"stage_oracle_installed\":true,"
        "\"functional_qualified\":false,\"failures\":%d,"
        "\"status\":\"%s\",\"launch\":\"multi_process\"}\n",
        rank, world_size, failures, failures == 0 ? "pass" : "fail");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  const int world_size = static_cast<int>(
      parse_long_env("NTHREADS", 1, 1, kPhysicalRanks));
  if (!is_supported_world_size(world_size) || world_size > device_count) {
    std::fprintf(stderr,
                 "NTHREADS must be one of {1,2,4,8} and <= devices=%d\n",
                 device_count);
    return EXIT_FAILURE;
  }
  ncclUniqueId unique_id{};
  NCCL_CHECK(ncclGetUniqueId(&unique_id));
  std::vector<std::thread> threads;
  threads.reserve(world_size);
  for (int rank = 0; rank < world_size; ++rank) {
    threads.emplace_back(run_canonical_rank, rank, world_size, rank,
                         unique_id, active_rows, mask_period, route_mode,
                         oracle_name);
  }
  for (auto& thread : threads) thread.join();
  const int failures = g_failures.load();
  std::printf(
      "RESULT_JSON={\"kind\":\"cake_sm120_production_canonical_fused_ready_chunk8\","
      "\"world_size\":%d,\"launch_count\":1,"
      "\"stage_oracle_installed\":true,"
      "\"functional_qualified\":false,\"failures\":%d,"
      "\"status\":\"%s\",\"launch\":\"threaded\"}\n",
      world_size, failures, failures == 0 ? "pass" : "fail");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
