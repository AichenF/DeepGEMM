import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_wg')
INC='/home/xutingz/fac/DeepGEMM/deep_gemm/include'
CUDA=r'''
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
#include <deep_gemm/quantization/mxfp4_dequant.cuh>
extern "C" __global__ void smoke(float* o){
  __shared__ __align__(1024) uint8_t sm[2048];
  auto desc = deep_gemm::mma::sm90::make_smem_desc(sm, 0);
  o[threadIdx.x]=(float)(threadIdx.x) + (float)(desc.desc_ & 0x1u);
}
void run(torch::Tensor o){ smoke<<<1,32>>>(o.data_ptr<float>()); }
'''
e=load_inline(name='smoke_wgmma', cpp_sources="void run(torch::Tensor o);", cuda_sources=CUDA, functions=['run'],
  extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda',
                     '-gencode','arch=compute_90a,code=sm_90a','-std=c++17',f'-I{INC}'], verbose=True)
o=torch.zeros(32,device='cuda'); e.run(o); torch.cuda.synchronize(); print("SMOKE_OK", o[5].item())
