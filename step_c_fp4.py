"""Step C: MXFP4 WGMMA GEMM. C[M=64,N=64]=A_fp8[64,128] @ dequant(W_fp4[64,128])^T.
W is fp4 (E2M1) + per-32 E8M0; dequant to fp8 via DeepGEMM dequant_mxfp4_to_fp8_pair,
interleave to K-order, WRITE swizzled (col^((row&7)<<4)) into smem_b, WGMMA. Validate vs torch."""
import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_wg'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
INC='/home/xutingz/fac/DeepGEMM/deep_gemm/include'
M,N,K=64,64,128
FP4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device='cuda')
def quant_mxfp4_contig(W):
  Nn,H=W.shape; G=H//32
  w=W.float().view(Nn,G,32); amax=w.abs().amax(-1,keepdim=True).clamp_min(1e-30)
  e8=torch.ceil(torch.log2(amax/6.0)).clamp(-127,127); scale=torch.exp2(e8)
  e8b=(e8+127).to(torch.uint8).view(Nn,G); wn=(w/scale).clamp(-6,6)
  bnd=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.0],device=W.device)
  mag=torch.bucketize(wn.abs(),bnd); sign=(wn<0).to(torch.uint8)
  nib=((sign<<3)|mag.to(torch.uint8)).view(Nn,H)
  packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8).contiguous()
  return packed,e8b.contiguous()
def dequant_contig(packed,e8b):
  Nn,Hh=packed.shape; H=Hh*2; G=e8b.shape[1]
  lo=packed&0xF; hi=(packed>>4)&0xF; nib=torch.stack([lo,hi],-1).view(Nn,H)
  mag=FP4.to(packed.device)[(nib&7).long()]; val=torch.where((nib>>3&1).bool(),-mag,mag)
  scale=torch.exp2((e8b.int()-127).float()).view(Nn,G,1).expand(Nn,G,32).reshape(Nn,H)
  return val*scale
CUDA=r'''
#include <cuda.h>
#include <cutlass/arch/barrier.h>
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
using namespace deep_gemm;
static constexpr int MM=64,NN=64,KK=128;
extern "C" __global__ void __launch_bounds__(128)
gemm(const __grid_constant__ CUtensorMap tma_a, const uint8_t* Wp, const uint8_t* We, float* C){
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smem_a = smem;           // [64,128] fp8 activations (swizzled TMA)
  uint8_t* smem_b = smem + MM*KK;   // [64,128] fp8 weights (swizzled, dequant)
  __shared__ __align__(8) uint64_t bar;
  const uint32_t tid=threadIdx.x;
  const uint32_t bar_addr=(uint32_t)__cvta_generic_to_shared(&bar);
  const uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem_a);
  if(tid==0){ asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(bar_addr)); asm volatile("fence.proxy.async.shared::cta;"); }
  __syncthreads();
  if(tid==0){ asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(bar_addr),"n"(MM*KK));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"::"r"(sa),"l"(&tma_a),"r"(0),"r"(0),"r"(bar_addr):"memory"); }
  // dequant W fp4 -> fp8 swizzled smem_b, in parallel with the A TMA
  // 1024 uint2 positions: u -> n=u/16, kb=(u%16)*8. read uint32 (8 nibbles) from Wp[n*64 + kb/2].
  for(int u=tid; u<MM*KK/8; u+=blockDim.x){
    int n=u/16, kg=u%16, kb=kg*8;
    unsigned uq=*reinterpret_cast<const unsigned*>(Wp + n*(KK/2) + (kb>>1));
    unsigned e8=(unsigned)We[n*(KK/32) + (kb>>5)];
    uint2 lut=mxfp4::load_e2m1_e8m0_lut(e8);
    uint2 hl=mxfp4::dequant_mxfp4_to_fp8_pair_with_lut(uq, lut);   // hl.x=hi(odd K), hl.y=lo(even K)
    unsigned w0=__byte_perm(hl.y, hl.x, 0x5140u);   // [K0,K1,K2,K3]
    unsigned w1=__byte_perm(hl.y, hl.x, 0x7362u);   // [K4,K5,K6,K7]
    int cs = kb ^ ((n&7)<<4);
    *reinterpret_cast<uint2*>(smem_b + n*128 + cs) = make_uint2(w0,w1);
  }
  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra L;}"::"r"(bar_addr):"memory");
  __syncthreads();
  using WGMMA=mma::sm90::FP8MMASelector<NN>::type; constexpr int kAcc=WGMMA::kNumAccum;
  float accum[kAcc];
  #pragma unroll
  for(int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
  ptx::warpgroup_arrive();
  #pragma unroll
  for(int k=0;k<KK/32;k++){ auto da=mma::sm90::make_smem_desc(smem_a+k*32,1); auto db=mma::sm90::make_smem_desc(smem_b+k*32,1); WGMMA::wgmma(da,db,accum,k); }
  ptx::warpgroup_commit_batch();
  #pragma unroll
  for(int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
  ptx::warpgroup_wait<0>();
  const int warp=tid/32,lane=tid%32,r0=warp*16+lane/4,r1=r0+8,cbase=(lane%4)*2;
  #pragma unroll
  for(int i=0;i<kAcc/4;i++){ C[r0*NN+cbase+i*8+0]=accum[i*4+0]; C[r0*NN+cbase+i*8+1]=accum[i*4+1]; C[r1*NN+cbase+i*8+0]=accum[i*4+2]; C[r1*NN+cbase+i*8+1]=accum[i*4+3]; }
}
CUtensorMap make_desc(void* ptr,int gi,int go,int stride){ CUtensorMap tm;
  cuuint64_t gd[2]={(cuuint64_t)gi,(cuuint64_t)go}; cuuint64_t gs[1]={(cuuint64_t)stride};
  cuuint32_t sd[2]={128u,(cuuint32_t)go}; cuuint32_t es[2]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,ptr,gd,gs,sd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS){const char* s;cuGetErrorString(r,&s);printf("TMA err %s\n",s);} return tm; }
torch::Tensor run(torch::Tensor a, torch::Tensor Wp, torch::Tensor We){
  auto C=torch::zeros({MM,NN},torch::device(a.device()).dtype(torch::kFloat32));
  auto tma_a=make_desc(a.data_ptr(), KK, MM, KK);
  int sh=MM*KK+MM*KK; cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,sh);
  gemm<<<1,128,sh>>>(tma_a, Wp.data_ptr<uint8_t>(), We.data_ptr<uint8_t>(), C.data_ptr<float>());
  return C; }
'''
e=load_inline(name='step_c_fp4',cpp_sources="torch::Tensor run(torch::Tensor a, torch::Tensor Wp, torch::Tensor We);",cuda_sources=CUDA,functions=['run'],
  extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode','arch=compute_90a,code=sm_90a','-std=c++17',f'-I{INC}'],extra_ldflags=['-lcuda'],verbose=True)
torch.manual_seed(0)
a=(torch.randn(M,K,device='cuda')*0.3).to(torch.float8_e4m3fn)
W=(torch.randn(N,K,device='cuda')*0.3)
Wp,We=quant_mxfp4_contig(W)
C=e.run(a.view(torch.uint8).contiguous(), Wp, We)
ref=a.float() @ dequant_contig(Wp,We).t()
cos=torch.nn.functional.cosine_similarity(C.flatten(),ref.flatten(),dim=0).item()
print(f"MXFP4_WGMMA cos={cos:.5f}  C[0,:4]={C[0,:4].tolist()} ref[0,:4]={ref[0,:4].tolist()}")
print("MXFP4_WGMMA_OK" if cos>0.99 else "MXFP4_WGMMA_WRONG")
