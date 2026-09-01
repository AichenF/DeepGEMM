"""Step A of the WGMMA+TMA rewrite: a CORRECT WGMMA fp8 GEMM.
C[M=64,N=64] = A[64,K=128] @ B[64,K=128]^T, fp8 e4m3, via TMA-load into 128B-swizzled smem
+ DeepGEMM FP8MMA::wgmma. Validate vs torch. Foundation for swap-AB + dequant + pipeline."""
import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_wg')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'   # sm_90a ONLY (WGMMA is Hopper; sm_100 rejects wgmma.*)
INC='/home/xutingz/fac/DeepGEMM/deep_gemm/include'
M,N,K = 64,64,128
CUDA=r'''
#include <cuda.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

using namespace deep_gemm;
static constexpr int MM=64, NN=64, KK=128;

// one warpgroup (128 threads). smem A[64,128], B[64,128] fp8, 128B-swizzled (TMA writes swizzle).
extern "C" __global__ void __launch_bounds__(128)
gemm(const __grid_constant__ CUtensorMap tma_a, const __grid_constant__ CUtensorMap tma_b, float* C){
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smem_a = smem;               // [64,128]
  uint8_t* smem_b = smem + MM*KK;       // [64,128] swizzled (manual)
  uint8_t* smem_braw = smem + MM*KK + NN*KK;  // [64,128] contiguous (TMA none)
  __shared__ __align__(8) uint64_t bar;
  const uint32_t tid = threadIdx.x;
  const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem_a));
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(smem_braw));
  if (tid==0){
    asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(bar_addr));
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();
  if (tid==0){
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(bar_addr),"n"(MM*KK+NN*KK));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(sa),"l"(&tma_a),"r"(0),"r"(0),"r"(bar_addr):"memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(sb),"l"(&tma_b),"r"(0),"r"(0),"r"(bar_addr):"memory");
  }
  // wait phase 0
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra L;}"::"r"(bar_addr):"memory");
  // manual 128B-swizzle copy: smem_b[r*128 + (c ^ ((r&7)<<4))] = smem_braw[r*128 + c], as uint2 (8B)
  for(int u=tid; u<MM*KK/8; u+=blockDim.x){ int r=u/16, cg=u%16, cb=cg*8; int cs=cb ^ ((r&7)<<4);
    *reinterpret_cast<uint2*>(smem_b + r*128 + cs) = *reinterpret_cast<uint2*>(smem_braw + r*128 + cb); }
  __syncthreads();

  using WGMMA = mma::sm90::FP8MMASelector<NN>::type;   // MMA_64xNNx32
  constexpr int kAcc = WGMMA::kNumAccum;               // 64*64/128 = 32
  float accum[kAcc];
  #pragma unroll
  for (int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
  ptx::warpgroup_arrive();
  #pragma unroll
  for (int k=0;k<KK/32;k++){
    auto da = mma::sm90::make_smem_desc(smem_a + k*32, 1);
    auto db = mma::sm90::make_smem_desc(smem_b + k*32, 1);
    WGMMA::wgmma(da, db, accum, k);
  }
  ptx::warpgroup_commit_batch();
  #pragma unroll
  for (int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
  ptx::warpgroup_wait<0>();
  // WGMMA accum layout (m64nN): thread t (warp w=t/32, lane l=t%32): rows r = w*16 + l/4, cols vary.
  // Standard: for i in kAcc/2: element (row = (t/32)*16 + (t%32)/4, col = i*8 + (t%32%4)*2 + {0,1}) but with 2-row.
  // Use the canonical m64 layout: gid=(t%32)/4 in [0,8), tid=(t%32)%4, warp=t/32 in [0,4).
  const int warp=tid/32, lane=tid%32;
  const int r0=warp*16 + lane/4, r1=r0+8, cbase=(lane%4)*2;
  #pragma unroll
  for (int i=0;i<kAcc/4;i++){
    C[r0*NN + cbase + i*8 + 0] = accum[i*4+0];
    C[r0*NN + cbase + i*8 + 1] = accum[i*4+1];
    C[r1*NN + cbase + i*8 + 0] = accum[i*4+2];
    C[r1*NN + cbase + i*8 + 1] = accum[i*4+3];
  }
}

CUtensorMap make_desc(void* ptr, int gmem_inner, int gmem_outer, int stride, int swz){
  CUtensorMap tm;
  cuuint64_t gd[2]={(cuuint64_t)gmem_inner,(cuuint64_t)gmem_outer};
  cuuint64_t gs[1]={(cuuint64_t)stride};
  cuuint32_t sd[2]={swz?128u:(cuuint32_t)gmem_inner,(cuuint32_t)gmem_outer};
  cuuint32_t es[2]={1,1};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr, gd, gs, sd, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swz?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); printf("TMA desc err: %s\n", s); }
  return tm;
}
torch::Tensor run(torch::Tensor a, torch::Tensor b){
  auto C=torch::zeros({MM,NN},torch::device(a.device()).dtype(torch::kFloat32));
  auto tma_a=make_desc(a.data_ptr(), KK, MM, KK, 1);
  auto tma_b=make_desc(b.data_ptr(), KK, NN, KK, 0);  // B non-swizzled -> manual swizzle
  int sh = MM*KK + NN*KK + NN*KK;
  cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, sh);
  gemm<<<1,128,sh>>>(tma_a, tma_b, C.data_ptr<float>());
  return C;
}
'''
e=load_inline(name='step_c_swz', cpp_sources="torch::Tensor run(torch::Tensor a, torch::Tensor b);", cuda_sources=CUDA, functions=['run'],
  extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode','arch=compute_90a,code=sm_90a','-std=c++17',f'-I{INC}'],
  extra_ldflags=['-lcuda'], verbose=True)
torch.manual_seed(0)
a=(torch.randn(M,K,device='cuda')*0.3).to(torch.float8_e4m3fn)
b=(torch.randn(N,K,device='cuda')*0.3).to(torch.float8_e4m3fn)
C=e.run(a.view(torch.uint8).contiguous(), b.view(torch.uint8).contiguous())
ref=(a.float() @ b.float().t())
cos=torch.nn.functional.cosine_similarity(C.flatten(),ref.flatten(),dim=0).item()
print(f"WGMMA_SWZ cos={cos:.5f}  C[0,:4]={C[0,:4].tolist()} ref[0,:4]={ref[0,:4].tolist()}")
print("WGMMA_SWZ_OK" if cos>0.99 else "WGMMA_SWZ_WRONG")
