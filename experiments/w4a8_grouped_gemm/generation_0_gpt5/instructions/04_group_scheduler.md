# Optimal Group GEMM Strategy

## Overview

Group GEMM (Grouped General Matrix Multiplication) computes multiple independent GEMM operations with **distinct problem sizes** in a single kernel launch. This differs from batched GEMM where all problems have identical dimensions.

**Key Advantage**: A single kernel launch handles heterogeneous problem sizes, eliminating the overhead of multiple kernel launches and enabling better GPU utilization through work-stealing across problems.

### 3. Persistent Kernel Design

The kernel uses a **persistent loop** pattern where each threadblock processes multiple tiles:

```cpp
__global__ void group_gemm_kernel(KernelParams params) {
    // Each block starts with its block index as the initial tile
    int tile_idx = blockIdx.x;

    // Persistent loop: keep processing tiles until no more work
    while (tile_idx < params.total_tiles) {
        // Find which problem this tile belongs to
        int problem_idx;
        int problem_tile_start;
        find_problem_for_tile(params, tile_idx, &problem_idx, &problem_tile_start);

        ProblemSize problem = params.problem_sizes[problem_idx];
        int local_tile_idx = tile_idx - problem_tile_start;

        // Compute tile position within problem
        int2 grid = grid_shape(problem);
        int tile_m = local_tile_idx / grid.y;
        int tile_n = local_tile_idx % grid.y;

        // Execute GEMM for this tile
        execute_gemm_tile(A, B, C, D, ldm_A, ldm_B,
                          tile_m, tile_n, problem,
                          params.alpha, params.beta);

        // Advance to next tile (strided by grid size for load balancing)
        tile_idx += gridDim.x;
    }
}
```

## 7. Variable-Length M Dimension (Varlen Scheduling)

Handle problems where different batches have different sequence lengths:

```cpp
// Cumulative sequence lengths define batch boundaries
// cu_seqlens = [0, 128, 256, 512] means:
//   batch 0: rows 0-127
//   batch 1: rows 128-255
//   batch 2: rows 256-511

struct VarlenScheduler {
    int* cu_seqlens;     // Cumulative sequence lengths
    int num_batches;
    int tile_m;

    __device__ void find_batch_for_tile(
        int tile_m_idx,
        int* out_batch_idx,
        int* out_m_start,
        int* out_m_end
    ) {
        // Binary search to find which batch contains this M tile
        int m_global = tile_m_idx * tile_m;

        int lo = 0, hi = num_batches;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cu_seqlens[mid + 1] <= m_global) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        *out_batch_idx = lo;
        *out_m_start = cu_seqlens[lo];
        *out_m_end = cu_seqlens[lo + 1];
    }
};
```
