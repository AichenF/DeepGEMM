// CUPTI activity tracer for SM120 MegaMoE timing.
//
// Loaded through ``CUDA_INJECTION64_PATH`` so the timed executable needs no
// instrumentation of its own. Every concurrent-kernel activity record is
// written as one JSON line carrying the device, stream, correlation id, device
// timestamps and demangled-free kernel name. The analyzer turns those records
// into per-rank iteration envelopes, which is what makes a multi-kernel
// candidate comparable with a single fused kernel: inter-kernel gaps, launch
// latency and tail are all inside the reconstructed envelope.
//
// Build:
//   nvcc -O2 -Xcompiler=-fPIC -shared cupti_kernel_trace.cpp -o libcake_cupti_trace.so \
//        -I$CUPTI/include -L$CUPTI/lib -lcupti

#include <cupti.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {

constexpr size_t kBufferSize = 8u << 20;
constexpr size_t kBufferAlign = 8;

std::mutex g_mutex;
std::FILE* g_output = nullptr;
std::atomic<unsigned long long> g_records{0};
std::atomic<bool> g_finalized{false};

void write_escaped(std::FILE* out, const char* text) {
  for (const char* c = text; *c != '\0'; ++c) {
    if (*c == '"' || *c == '\\') {
      std::fputc('\\', out);
      std::fputc(*c, out);
    } else if (static_cast<unsigned char>(*c) < 0x20) {
      std::fprintf(out, "\\u%04x", static_cast<unsigned char>(*c));
    } else {
      std::fputc(*c, out);
    }
  }
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size,
                               size_t* max_num_records) {
  auto* raw = static_cast<uint8_t*>(std::malloc(kBufferSize + kBufferAlign));
  if (raw == nullptr) {
    *buffer = nullptr;
    *size = 0;
    *max_num_records = 0;
    return;
  }
  auto address = reinterpret_cast<uintptr_t>(raw);
  auto aligned = (address + (kBufferAlign - 1)) & ~(uintptr_t)(kBufferAlign - 1);
  *buffer = reinterpret_cast<uint8_t*>(aligned);
  *size = kBufferSize;
  *max_num_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer,
                               size_t /*size*/, size_t valid_size) {
  if (valid_size > 0) {
    std::lock_guard<std::mutex> guard(g_mutex);
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
           CUPTI_SUCCESS) {
      if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
      const auto* kernel = reinterpret_cast<CUpti_ActivityKernel12*>(record);
      if (g_output != nullptr) {
        std::fprintf(g_output,
                     "{\"device\":%u,\"stream\":%u,\"correlation\":%u,"
                     "\"start_ns\":%llu,\"end_ns\":%llu,\"name\":\"",
                     kernel->deviceId, kernel->streamId, kernel->correlationId,
                     static_cast<unsigned long long>(kernel->start),
                     static_cast<unsigned long long>(kernel->end));
        write_escaped(g_output, kernel->name == nullptr ? "" : kernel->name);
        std::fprintf(g_output, "\"}\n");
      }
      g_records.fetch_add(1, std::memory_order_relaxed);
    }
  }
  std::free(buffer);
}

void finalize() {
  if (g_finalized.exchange(true)) return;
  cuptiActivityFlushAll(1);
  std::lock_guard<std::mutex> guard(g_mutex);
  if (g_output != nullptr) {
    std::fflush(g_output);
    std::fclose(g_output);
    g_output = nullptr;
  }
}

}  // namespace

extern "C" int InitializeInjection(void) {
  static std::once_flag once;
  int status = 1;
  std::call_once(once, [&status]() {
    std::string path;
    if (const char* configured = std::getenv("CAKE_CUPTI_TRACE")) {
      path = configured;
    } else {
      path = "/tmp/cake-cupti-trace." + std::to_string(getpid()) + ".jsonl";
    }
    g_output = std::fopen(path.c_str(), "w");
    if (g_output == nullptr) {
      std::fprintf(stderr, "cake cupti trace: cannot open %s\n", path.c_str());
      status = 0;
      return;
    }
    std::setvbuf(g_output, nullptr, _IOFBF, 1u << 20);

    CUptiResult result =
        cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (result != CUPTI_SUCCESS) {
      std::fprintf(stderr, "cake cupti trace: register failed %d\n", result);
      status = 0;
      return;
    }
    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
      std::fprintf(stderr, "cake cupti trace: enable failed %d\n", result);
      status = 0;
      return;
    }
    std::atexit(finalize);
  });
  return status;
}
