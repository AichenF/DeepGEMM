"""Step D: pipelined swap-AB MXFP4 WGMMA GEMM (the memory-bound perf test).
C'[Wout=64, tok=8] = W_fp4[64,K=6144] @ X_fp8[8,K]^T. swap-AB: weights=A(m64), tokens=B(N=8).
Double-buffered TMA-load of fp4 weight + fp8 act overlaps dequant+WGMMA. Validate cos + time -> eff BW."""
import os, torch, time
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_wg'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
INC='/home/xutingz/fac/DeepGEMM/deep_gemm/include'
WOUT,TOK,K=64,8,7168
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
static constexpr int WOUT=64,TOK=8,K=7168,BK=128,NKT=K/BK,STG=3;
static constexpr int WPB=WOUT*(BK/2);   // packed weight bytes/tile = 64*64=4096
static constexpr int XB=TOK*BK;          // act bytes/tile = 8*128=1024

__device__ __forceinline__ void mbar_init(uint32_t a){ asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(a)); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph){ asm volatile("{.reg .pred p; L:mbarrier.try_wait.parity.shared.b64 p,[%0],%1; @!p bra L;}"::"r"(a),"r"(ph):"memory"); }

extern "C" __global__ void __launch_bounds__(128)
gemm(const __grid_constant__ CUtensorMap tma_w, const __grid_constant__ CUtensorMap tma_x, const uint8_t* We, float* C){
  const int blk=blockIdx.x; const long wrow=(long)blk*WOUT;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sw = smem;                     // packed weight [STG][64,64]
  uint8_t* sx = sw + STG*WPB;             // act fp8 [STG][8,128] swizzled
  uint8_t* swf = sx + STG*XB;             // dequant weight fp8 [64,128] swizzled (single)
  __shared__ __align__(8) uint64_t full[STG], empty[STG];
  __shared__ uint2 smem_lut[256];         // precompute E8M0 -> fp8-mag LUT ONCE (else exp2f+8cvt per elem)
  __shared__ uint8_t sWe[WOUT*(K/32)];    // preload E8M0 scales [64,192] once (else 384 global reads/thread)
  const uint32_t tid=threadIdx.x;
  for(int i=tid;i<256;i+=blockDim.x) smem_lut[i]=mxfp4::load_e2m1_e8m0_lut(i);
  for(int i=tid;i<WOUT*(K/32);i+=blockDim.x) sWe[i]=We[wrow*(K/32)+i];
  uint32_t fa[STG], ea[STG];
  #pragma unroll
  for(int s=0;s<STG;s++){ fa[s]=(uint32_t)__cvta_generic_to_shared(&full[s]); ea[s]=(uint32_t)__cvta_generic_to_shared(&empty[s]); }
  if(tid==0){
    #pragma unroll
    for(int s=0;s<STG;s++){ mbar_init(fa[s]); mbar_init(ea[s]); }
    asm volatile("fence.proxy.async.shared::cta;"); }
  __syncthreads();
  auto load_stage=[&](int kt,int s){
    if(tid==0){
      uint32_t swp=(uint32_t)__cvta_generic_to_shared(sw+s*WPB), sxp=(uint32_t)__cvta_generic_to_shared(sx+s*XB);
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;"::"r"(fa[s]),"n"(WPB+XB));
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(swp),"l"(&tma_w),"r"(kt*(BK/2)),"r"((int)wrow),"r"(fa[s]):"memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(sxp),"l"(&tma_x),"r"(kt*BK),"r"(0),"r"(fa[s]):"memory");
    }
  };
  #pragma unroll
  for(int s=0;s<STG;s++) load_stage(s,s);   // prologue: fill all STG buffers (tiles 0..STG-1)
  using WGMMA=mma::sm90::FP8MMASelector<TOK>::type; constexpr int kAcc=WGMMA::kNumAccum; // 64*8/128=4
  float accum[kAcc];
  #pragma unroll
  for(int i=0;i<kAcc;i++) accum[i]=0.f;
  const int warp=tid/32,lane=tid%32;
  for(int kt=0;kt<NKT;kt++){
    int s=kt%STG; uint32_t ph=(kt/STG)&1u;   // buffer s holds tiles s,s+STG,...; phase flips each reuse
    mbar_wait(fa[s], ph);
    // dequant sw[s] (fp4 [64,64]) -> swf [64,128] swizzled fp8
    for(int u=tid;u<WOUT*BK/8;u+=blockDim.x){ int n=u/16,kg=u%16,kb=kg*8;
      unsigned uq=*reinterpret_cast<const unsigned*>(sw+s*WPB + n*(BK/2) + (kb>>1));
      unsigned e8=(unsigned)sWe[n*(K/32) + ((kt*BK+kb)>>5)];
      uint2 hl=mxfp4::dequant_mxfp4_to_fp8_pair_with_lut(uq, smem_lut[e8]);
      unsigned w0=__byte_perm(hl.y,hl.x,0x5140u), w1=__byte_perm(hl.y,hl.x,0x7362u);
      int cs=kb ^ ((n&7)<<4);
      *reinterpret_cast<uint2*>(swf + n*128 + cs)=make_uint2(w0,w1);
    }
    asm volatile("bar.sync 0;");
    #pragma unroll
    for(int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_arrive();
    #pragma unroll
    for(int k=0;k<BK/32;k++){ auto da=mma::sm90::make_smem_desc(swf+k*32,1); auto db=mma::sm90::make_smem_desc(sx+s*XB+k*32,1); WGMMA::wgmma(da,db,accum,1); }
    ptx::warpgroup_commit_batch();
    #pragma unroll
    for(int i=0;i<kAcc;i++) ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_wait<0>();   // warpgroup_wait already syncs the 128-thread WG -> swf/sx free
    if(tid==0 && kt+STG<NKT) load_stage(kt+STG, s);   // reload tile kt+STG into just-freed buffer s
  }
  // epilogue: C'[64 wout, 8 tok]. m64 fragment.
  const int r0=warp*16+lane/4,r1=r0+8,cbase=(lane%4)*2;
  #pragma unroll
  for(int i=0;i<kAcc/4;i++){ C[(wrow+r0)*TOK+cbase+i*8+0]=accum[i*4+0]; C[(wrow+r0)*TOK+cbase+i*8+1]=accum[i*4+1]; C[(wrow+r1)*TOK+cbase+i*8+0]=accum[i*4+2]; C[(wrow+r1)*TOK+cbase+i*8+1]=accum[i*4+3]; }
}
CUtensorMap make_desc(void* ptr,int gi,int go,int so,int stride,int swz){ CUtensorMap tm;
  cuuint64_t gd[2]={(cuuint64_t)gi,(cuuint64_t)go}; cuuint64_t gs[1]={(cuuint64_t)stride};
  cuuint32_t sd[2]={swz?128u:64u,(cuuint32_t)so}; cuuint32_t es[2]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,ptr,gd,gs,sd,es,CU_TENSOR_MAP_INTERLEAVE_NONE,swz?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS){const char* s;cuGetErrorString(r,&s);printf("TMA err %s\n",s);} return tm; }
torch::Tensor run(torch::Tensor Wp, torch::Tensor We, torch::Tensor X, int G){
  static CUtensorMap tw, tx; static void* lastW=nullptr; static bool sh_set=false;
  if(lastW != Wp.data_ptr()){   // create TMA descriptors ONCE per tensor (host driver call is ~100us)
    tw=make_desc(Wp.data_ptr(), K/2, Wp.size(0), WOUT, K/2, 0);
    tx=make_desc(X.data_ptr(), K, TOK, TOK, K, 1);
    lastW=Wp.data_ptr();
  }
  auto C=torch::zeros({(long)G*WOUT,TOK},torch::device(Wp.device()).dtype(torch::kFloat32));
  int sh=STG*WPB + STG*XB + WOUT*128;
  if(!sh_set){ cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,sh); sh_set=true; }
  gemm<<<G,128,sh>>>(tw, tx, We.data_ptr<uint8_t>(), C.data_ptr<float>());
  return C; }
'''
e=load_inline(name='step_e_bench',cpp_sources="torch::Tensor run(torch::Tensor Wp, torch::Tensor We, torch::Tensor X, int G);",cuda_sources=CUDA,functions=['run'],
  extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode','arch=compute_90a,code=sm_90a','-std=c++17',f'-I{INC}'],extra_ldflags=['-lcuda'],verbose=True)
torch.manual_seed(0)
G=int(os.environ.get('G','2960'))   # FC1 M=32: ~185 experts x 16 out-tiles
# block 0 = real quantized (correctness); rest = random packed bytes (throughput only, avoids 4.6GB float)
W0=(torch.randn(WOUT,K,device='cuda')*0.2); Wp0,We0=quant_mxfp4_contig(W0)
Wp=torch.randint(0,256,(G*WOUT,K//2),dtype=torch.uint8,device='cuda'); Wp[:WOUT]=Wp0
We=torch.randint(120,132,(G*WOUT,K//32),dtype=torch.uint8,device='cuda'); We[:WOUT]=We0
X=(torch.randn(TOK,K,device='cuda')*0.3).to(torch.float8_e4m3fn)
C=e.run(Wp,We,X.view(torch.uint8).contiguous(),G)
ref0=(dequant_contig(Wp0,We0).float() @ X.float().t())
cos=torch.nn.functional.cosine_similarity(C[:WOUT].flatten(),ref0.flatten(),dim=0).item()
print(f"STEPE cos(blk0)={cos:.5f}  G={G}")
for _ in range(5): e.run(Wp,We,X.view(torch.uint8).contiguous(),G)
torch.cuda.synchronize(); t0=time.time()
for _ in range(30): e.run(Wp,We,X.view(torch.uint8).contiguous(),G)
torch.cuda.synchronize(); ms=(time.time()-t0)/30*1e3
wbytes=G*WOUT*K/2
print(f"STEPE grid={G} time={ms:.3f} ms  weight={wbytes/1e6:.0f}MB  BW={wbytes/(ms*1e-3)/1e9:.0f} GB/s")
print("STEPE_OK" if cos>0.99 else "STEPE_WRONG")
