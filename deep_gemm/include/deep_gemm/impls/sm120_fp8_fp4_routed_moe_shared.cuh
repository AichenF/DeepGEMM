#pragma once

#if DG_SM120_ROUTED_MOE_WORLD_SIZE == 4
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe_shared_ep4.cuh>
#elif DG_SM120_ROUTED_MOE_WORLD_SIZE == 8
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe_shared_ep8.cuh>
#else
#error "SM120 fused shared-expert MoE supports EP4 and EP8"
#endif
