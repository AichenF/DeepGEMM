"""Milestone FC1: gate/up[T,2Is] = x[T,H] @ W1[2Is,H]^T via fp8-mma with in-smem
MXFP4->fp8 fold + per-token x scale. Contiguous MXFP4 packing (nibble k = K elem k).
Verify vs torch mxfp4-dequant reference."""
import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

FP4 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.], device='cuda')

def quant_mxfp4_contig(W):  # W [N,H] -> packed [N,H/2] uint8 (nibble k = elem k), e8m0 [N,H/32]
    N,H = W.shape; G=H//32
    w=W.float().view(N,G,32); amax=w.abs().amax(-1,keepdim=True).clamp_min(1e-30)
    e8=torch.ceil(torch.log2(amax/6.0)).clamp(-127,127); scale=torch.exp2(e8)
    e8b=(e8+127).to(torch.uint8).view(N,G)
    wn=(w/scale).clamp(-6,6)
    bnd=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.0],device='cuda')
    mag=torch.bucketize(wn.abs(),bnd); sign=(wn<0).to(torch.uint8)
    nib=(sign<<3)|mag.to(torch.uint8)  # [N,G,32]
    nib=nib.view(N,H)
    packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8).contiguous()
    return packed, e8b.contiguous()

def dequant_contig(packed,e8b):
    N,Hh=packed.shape; H=Hh*2; G=e8b.shape[1]
    lo=packed&0xF; hi=(packed>>4)&0xF
    nib=torch.stack([lo,hi],-1).view(N,H)
    mag=FP4[(nib&7).long()]; val=torch.where((nib>>3&1).bool(),-mag,mag)
    scale=torch.exp2((e8b.int()-127).float()).view(N,G,1).expand(N,G,32).reshape(N,H)
    return val*scale

CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void fold_tbl(unsigned e8,unsigned&lo,unsigned&hi){
  const unsigned bLo=0x0c080000u,bHi=0x1c181410u; const unsigned off=(e8>=121u)?((e8-121u)<<3):0u,offp=off*0x01010101u;
  lo=(__vminu4(__vaddus4(bLo,offp),0x7e7e7e7eu)&0x7f7f7f7fu)&0xffffff00u; hi=__vminu4(__vaddus4(bHi,offp),0x7e7e7e7eu)&0x7f7f7f7fu; }
__device__ __forceinline__ void fold32(const uint8_t*p16,unsigned e8,uint8_t*out){
  unsigned lo,hi; fold_tbl(e8,lo,hi);
  #pragma unroll
  for(int q=0;q<4;q++){ unsigned w=reinterpret_cast<const unsigned*>(p16)[q];
    #pragma unroll
    for(int j=0;j<8;j++){ unsigned nib=(w>>(4*j))&0xF,m=nib&7u,t=(m<4)?lo:hi,byte=(t>>(8*(m&3)))&0xff; if(nib&8)byte|=0x80u; out[q*8+j]=(uint8_t)byte; } } }
// mma A[16,K]xf8 @ B[N,K]xf8 -> C[16,N] f32. weights folded per k-tile from smem Bs.
// FUSED: FC1(W1 [2Is,H]) -> gate/up[16,2Is] in smem -> SwiGLU -> act fp8[16,Is] -> FC2(W2 [H,Is]) -> y[16,H].
extern "C" __global__ void ffn(const uint8_t* xf,const float* sx,
    const uint8_t* W1p,const uint8_t* W1e,const uint8_t* W2p,const uint8_t* W2e,
    float* y,int T,int H,int Is){
  const int t=threadIdx.x&31,gid=t>>2,tid=t&3,warp=threadIdx.x>>5,NW=blockDim.x>>5;
  const int twoIs=2*Is;
  extern __shared__ uint8_t sm[];
  uint8_t* xs=sm;                    // [16,H] fp8
  float* gu=reinterpret_cast<float*>(sm+16*H);     // [16,2Is] gate/up
  uint8_t* af=reinterpret_cast<uint8_t*>(gu+16*twoIs); // [16,Is] act fp8
  float* asc=reinterpret_cast<float*>(af+16*Is);   // [16] act scale
  __shared__ uint8_t Bs[8][256];
  for(int i=threadIdx.x;i<16*H;i+=blockDim.x) xs[i]=(i/H<T)?xf[i]:0;
  __syncthreads();
  const int Hh=H>>1,Ge=H>>5;
  // ---- FC1 -> gu ----
  for(int n0=warp*8;n0<twoIs;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    for(int k0=0;k0<H;k0+=32){
      if(t<8) fold32(W1p+(long)(n0+t)*Hh+(k0>>1),W1e[(long)(n0+t)*Ge+(k0>>5)],&Bs[warp][t*32]);
      __syncwarp();
      unsigned a0=*(const unsigned*)(xs+gid*H+k0+tid*4),a1=*(const unsigned*)(xs+(gid+8)*H+k0+tid*4);
      unsigned a2=*(const unsigned*)(xs+gid*H+k0+tid*4+16),a3=*(const unsigned*)(xs+(gid+8)*H+k0+tid*4+16);
      unsigned b0=*(const unsigned*)(&Bs[warp][gid*32+tid*4]),b1=*(const unsigned*)(&Bs[warp][gid*32+tid*4+16]);
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }
    gu[gid*twoIs+n0+tid*2]=c0*sx[gid]; gu[gid*twoIs+n0+tid*2+1]=c1*sx[gid];
    gu[(gid+8)*twoIs+n0+tid*2]=c2*sx[gid+8]; gu[(gid+8)*twoIs+n0+tid*2+1]=c3*sx[gid+8]; }
  __syncthreads();
  // ---- SwiGLU + requant act -> af, asc ---- (gate=gu[:, :Is], up=gu[:, Is:])
  for(int r=warp; r<T; r+=NW){ float amax=1e-30f;
    for(int i=(t&31); i<Is; i+=32){ float g=gu[r*twoIs+i],u=gu[r*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u; amax=fmaxf(amax,fabsf(a)); }
    #pragma unroll
    for(int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(-1u,amax,o));
    amax=__shfl_sync(-1u,amax,0); float s=amax/448.f; if(t==0||(t&31)==0) asc[r]=s;
    for(int i=(t&31); i<Is; i+=32){ float g=gu[r*twoIs+i],u=gu[r*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u;
      af[r*Is+i]=(uint8_t)__nv_cvt_float_to_fp8(a/s,__NV_SATFINITE,__NV_E4M3); } }
  __syncthreads();
  // ---- FC2: y[16,H] = act[16,Is] @ W2[H,Is]^T  (N=H, K=Is) ----
  const int Ih=Is>>1,Ie=Is>>5;
  for(int n0=warp*8;n0<H;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    for(int k0=0;k0<Is;k0+=32){
      if(t<8) fold32(W2p+(long)(n0+t)*Ih+(k0>>1),W2e[(long)(n0+t)*Ie+(k0>>5)],&Bs[warp][t*32]);
      __syncwarp();
      unsigned a0=*(const unsigned*)(af+gid*Is+k0+tid*4),a1=*(const unsigned*)(af+(gid+8)*Is+k0+tid*4);
      unsigned a2=*(const unsigned*)(af+gid*Is+k0+tid*4+16),a3=*(const unsigned*)(af+(gid+8)*Is+k0+tid*4+16);
      unsigned b0=*(const unsigned*)(&Bs[warp][gid*32+tid*4]),b1=*(const unsigned*)(&Bs[warp][gid*32+tid*4+16]);
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }
    if(gid<T){ y[gid*H+n0+tid*2]=c0*asc[gid]; y[gid*H+n0+tid*2+1]=c1*asc[gid]; }
    if(gid+8<T){ y[(gid+8)*H+n0+tid*2]=c2*asc[gid+8]; y[(gid+8)*H+n0+tid*2+1]=c3*asc[gid+8]; } }
}
torch::Tensor run(torch::Tensor xf,torch::Tensor sx,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,int T,int Is){
  int H=xf.size(1); auto y=torch::zeros({T,H},torch::device(xf.device()).dtype(torch::kFloat32));
  long sh=(long)16*H + 16*2*Is*4 + 16*Is + 16*4; int thr=256;
  cudaFuncSetAttribute(ffn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
  ffn<<<1,thr,sh>>>(xf.data_ptr<uint8_t>(),sx.data_ptr<float>(),W1p.data_ptr<uint8_t>(),W1e.data_ptr<uint8_t>(),W2p.data_ptr<uint8_t>(),W2e.data_ptr<uint8_t>(),y.data_ptr<float>(),T,H,Is);
  return y; }
"""

e=load_inline(name='ffn_fp8mma',cpp_sources="torch::Tensor run(torch::Tensor xf,torch::Tensor sx,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,int T,int Is);",cuda_sources=CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
torch.manual_seed(0)
T,H,Is=16,6144,512
x=torch.randn(T,H,device='cuda'); W1=(torch.randn(2*Is,H,device='cuda')*0.05); W2=(torch.randn(H,Is,device='cuda')*0.05)
W1p,W1e=quant_mxfp4_contig(W1); W2p,W2e=quant_mxfp4_contig(W2)
sx=(x.abs().amax(1)/448.0).clamp_min(1e-30); xf=(x/sx.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
y=e.run(xf,sx.float().contiguous(),W1p,W1e,W2p,W2e,T,Is)
W1d=dequant_contig(W1p,W1e); W2d=dequant_contig(W2p,W2e)
h1=x@W1d.t(); g=h1[:,:Is]; u=h1[:,Is:]; a=(g*torch.sigmoid(g))*u; ref=a@W2d.t()
cos=torch.nn.functional.cosine_similarity(y.flatten(),ref.flatten(),dim=0).item()
print(f"FFN fp8-mma cos={cos:.5f} y_absmax={y.abs().max():.3f} ref={ref.abs().max():.3f}")
print("FFN_OK" if cos>0.99 else "FFN_WRONG")
