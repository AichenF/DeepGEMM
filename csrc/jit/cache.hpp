#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;
    std::mutex mutex;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        {
            const std::lock_guard<std::mutex> guard(mutex);
            if (const auto iterator = cache.find(dir_path); iterator != cache.end())
                return iterator->second;
        }

        if (not KernelRuntime::check_validity(dir_path))
            return nullptr;

        // CUBIN inspection and module loading can be slow. Do it outside the
        // map mutex so independent kernels may load concurrently. If two
        // threads race on the same path, retain the first published runtime.
        auto runtime = std::make_shared<KernelRuntime>(dir_path);
        const std::lock_guard<std::mutex> guard(mutex);
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;
        cache.emplace(dir_path, runtime);
        return runtime;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
