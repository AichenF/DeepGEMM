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


def tile_major(packed,e8b,Nblk=8,Kblk=32):  # packed[E,N,H/2],e8[E,N,H/32] -> contiguous [E, N/8, H/32, 8, 16] and [E,N/8,H/32,8]
  E,N,Hh=packed.shape; H=Hh*2; nb=N//Nblk; kb=H//Kblk
  p=packed.view(E,nb,Nblk,kb,Kblk//2).permute(0,1,3,2,4).contiguous().view(E,nb*kb*Nblk*(Kblk//2))
  e=e8b.view(E,nb,Nblk,kb,1).permute(0,1,3,2,4).contiguous().view(E,nb*kb*Nblk)
  return p,e

CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void fold_tbl(unsigned e8,unsigned&lo,unsigned&hi){
  const unsigned bLo=0x0c080000u,bHi=0x1c181410u; const unsigned off=(e8>=121u)?((e8-121u)<<3):0u,offp=off*0x01010101u;
  lo=(__vminu4(__vaddus4(bLo,offp),0x7e7e7e7eu)&0x7f7f7f7fu)&0xffffff00u; hi=__vminu4(__vaddus4(bHi,offp),0x7e7e7e7eu)&0x7f7f7f7fu; }
__device__ __forceinline__ unsigned fold4(unsigned h,unsigned lo,unsigned hi){ // 4 nibbles(uint16)->4 fp8
  unsigned o,sel=h&0x7777u; asm("prmt.b32 %0,%1,%2,%3;":"=r"(o):"r"(lo),"r"(hi),"r"(sel));
  return o|((h&8u)<<4)|((h&0x80u)<<8)|((h&0x800u)<<12)|((h&0x8000u)<<16); }
__device__ __forceinline__ void fold32(const uint8_t*p16,unsigned e8,uint8_t*out){
  unsigned lo,hi; fold_tbl(e8,lo,hi);
  #pragma unroll
  for(int q=0;q<4;q++){ unsigned w=reinterpret_cast<const unsigned*>(p16)[q];
    // contiguous nibbles: mag(nj)=(w>>4j)&7. sel_lo picks n0..3 mags, sel_hi n4..7.
    const unsigned sl=w&0x7777u, sh=(w>>16)&0x7777u;
    unsigned ol,oh; asm("prmt.b32 %0,%1,%2,%3;":"=r"(ol):"r"(lo),"r"(hi),"r"(sl));
    asm("prmt.b32 %0,%1,%2,%3;":"=r"(oh):"r"(lo),"r"(hi),"r"(sh));
    const unsigned sgl=((w&8u)<<4)|((w&0x80u)<<8)|((w&0x800u)<<12)|((w&0x8000u)<<16);
    const unsigned sgh=((w&0x80000u)>>12)|((w&0x800000u)>>8)|((w&0x8000000u)>>4)|(w&0x80000000u);
    reinterpret_cast<unsigned*>(out)[q*2]=ol|sgl; reinterpret_cast<unsigned*>(out)[q*2+1]=oh|sgh; } }
// grouped MoE: block=tile. gather tokens, FC1->SwiGLU->FC2, scatter weighted to y.
extern "C" __global__ void moe(const uint8_t* xall,const float* sxall,
    const uint8_t* W1p,const uint8_t* W1e,const uint8_t* W2p,const uint8_t* W2e,
    const int* te,const int* tn,const int* ttok,const float* tw, float* y,int H,int Is,int E){
  const int b=blockIdx.x; const int e=te[b],T=tn[b];
  const int t=threadIdx.x&31,gid=t>>2,tid=t&3,warp=threadIdx.x>>5,NW=blockDim.x>>5;
  const int twoIs=2*Is;
  extern __shared__ uint8_t sm[];
  uint8_t* xs=sm; float* gu=reinterpret_cast<float*>(sm+8*H);
  uint8_t* af=reinterpret_cast<uint8_t*>(gu+8*twoIs); float* asc=reinterpret_cast<float*>(af+8*Is);
  float* lsx=asc+8;   // gathered token x-scale
  __shared__ int tok[16]; __shared__ float rw[16];
  if(threadIdx.x<8){ int r=threadIdx.x; int tk=(r<T)?ttok[b*8+r]:-1; tok[r]=tk; rw[r]=(r<T)?tw[b*8+r]:0.f; lsx[r]=(tk>=0)?sxall[tk]:0.f; }
  __syncthreads();
  for(int i=threadIdx.x;i<8*H;i+=blockDim.x){ int r=i/H,k=i%H; xs[i]=(r<T)?xall[(long)tok[r]*H+k]:0; }
  __syncthreads();
  for(int n0=warp*8;n0<twoIs;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    const long nb=((long)e*(twoIs>>3)+(n0>>3))*(H>>5);
    const uint8_t* wr0=W1p + nb*128 + gid*16;
    unsigned h0N=*(const unsigned short*)(wr0+tid*2), h1N=*(const unsigned short*)(wr0+tid*2+8), eN=(unsigned)W1e[nb*8+gid];
    for(int k0=0;k0<H;k0+=32){
      unsigned h0=h0N,h1=h1N,e8=eN;
      if(k0+32<H){ const long tbn=nb+((k0+32)>>5); const uint8_t* wrn=W1p + tbn*128 + gid*16;
        h0N=*(const unsigned short*)(wrn+tid*2); h1N=*(const unsigned short*)(wrn+tid*2+8); eN=(unsigned)W1e[tbn*8+gid]; }
      unsigned lo,hi; fold_tbl(e8,lo,hi);
      unsigned b0=fold4(h0,lo,hi), b1=fold4(h1,lo,hi);
      unsigned a0=*(const unsigned*)(xs+gid*H+k0+tid*4),a1=0u;
      unsigned a2=*(const unsigned*)(xs+gid*H+k0+tid*4+16),a3=0u;
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }
    if(gid<T){ gu[gid*twoIs+n0+tid*2]=c0*lsx[gid]; gu[gid*twoIs+n0+tid*2+1]=c1*lsx[gid]; } }
  __syncthreads();
  for(int r=warp;r<T;r+=NW){ float amax=1e-30f;
    for(int i=(t&31);i<Is;i+=32){ float g=gu[r*twoIs+i],u=gu[r*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u; amax=fmaxf(amax,fabsf(a)); }
    #pragma unroll
    for(int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(-1u,amax,o)); amax=__shfl_sync(-1u,amax,0);
    float s=amax/448.f; if((t&31)==0) asc[r]=s;
    for(int i=(t&31);i<Is;i+=32){ float g=gu[r*twoIs+i],u=gu[r*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u; af[r*Is+i]=(uint8_t)__nv_cvt_float_to_fp8(a/s,__NV_SATFINITE,__NV_E4M3);} }
  __syncthreads();
  for(int n0=warp*8;n0<H;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    const long nb=((long)e*(H>>3)+(n0>>3))*(Is>>5);
    const uint8_t* wr0=W2p + nb*128 + gid*16;
    unsigned h0N=*(const unsigned short*)(wr0+tid*2), h1N=*(const unsigned short*)(wr0+tid*2+8), eN=(unsigned)W2e[nb*8+gid];
    for(int k0=0;k0<Is;k0+=32){
      unsigned h0=h0N,h1=h1N,e8=eN;
      if(k0+32<Is){ const long tbn=nb+((k0+32)>>5); const uint8_t* wrn=W2p + tbn*128 + gid*16;
        h0N=*(const unsigned short*)(wrn+tid*2); h1N=*(const unsigned short*)(wrn+tid*2+8); eN=(unsigned)W2e[tbn*8+gid]; }
      unsigned lo,hi; fold_tbl(e8,lo,hi);
      unsigned b0=fold4(h0,lo,hi), b1=fold4(h1,lo,hi);
      unsigned a0=*(const unsigned*)(af+gid*Is+k0+tid*4),a1=0u;
      unsigned a2=*(const unsigned*)(af+gid*Is+k0+tid*4+16),a3=0u;
      asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }
    if(gid<T){ int tk=tok[gid]; float w=rw[gid]*asc[gid]; atomicAdd(&y[(long)tk*H+n0+tid*2],c0*w); atomicAdd(&y[(long)tk*H+n0+tid*2+1],c1*w); } }
}
torch::Tensor run(torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,
                  torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E){
  int H=xall.size(1),NT=te.size(0); auto y=torch::zeros({M,H},torch::device(xall.device()).dtype(torch::kFloat32));
  long sh=(long)8*H + 8*2*Is*4 + 8*Is + 8*4 + 8*4 + 8*256; int thr=256;
  cudaFuncSetAttribute(moe,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
  moe<<<NT,thr,sh>>>(xall.data_ptr<uint8_t>(),sxall.data_ptr<float>(),W1p.data_ptr<uint8_t>(),W1e.data_ptr<uint8_t>(),W2p.data_ptr<uint8_t>(),W2e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),tw.data_ptr<float>(),y.data_ptr<float>(),H,Is,E);
  return y; }
"""

e=load_inline(name='moe_fp8mma_v8',cpp_sources="torch::Tensor run(torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E);",cuda_sources=CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
import time
def build(M,H,Is,E,topk):
  torch.manual_seed(0)
  x=torch.randn(M,H,device='cuda'); W1=(torch.randn(E,2*Is,H,device='cuda')*0.05); W2=(torch.randn(E,H,Is,device='cuda')*0.05)
  sc=torch.randn(M,E,device='cuda'); tw,ti=torch.topk(sc,topk,dim=-1)
  W1p=[];W1e=[];W2p=[];W2e=[]
  for i in range(E): p,ee=quant_mxfp4_contig(W1[i]); W1p.append(p);W1e.append(ee); p,ee=quant_mxfp4_contig(W2[i]); W2p.append(p);W2e.append(ee)
  W1p=torch.stack(W1p);W1e=torch.stack(W1e);W2p=torch.stack(W2p);W2e=torch.stack(W2e)
  W1pt,W1et=tile_major(W1p,W1e); W2pt,W2et=tile_major(W2p,W2e)
  sx=(x.abs().amax(1)/448.).clamp_min(1e-30); xf=(x/sx.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
  # group pairs by expert into tiles<=16
  fe=ti.reshape(-1); ft=torch.arange(M,device='cuda').repeat_interleave(topk); fw=tw.reshape(-1)
  order=torch.argsort(fe); fe,ft,fw=fe[order],ft[order],fw[order]
  te=[];tn=[];tt=[];tww=[]
  i=0;P=fe.numel()
  while i<P:
    j=i
    while j<P and fe[j]==fe[i] and j-i<8: j+=1
    te.append(int(fe[i])); n=j-i; tn.append(n)
    tks=ft[i:j].tolist()+[0]*(8-n); ws=fw[i:j].tolist()+[0.]*(8-n)
    tt+=tks; tww+=ws; i=j
  te=torch.tensor(te,dtype=torch.int32,device='cuda'); tn=torch.tensor(tn,dtype=torch.int32,device='cuda')
  tt=torch.tensor(tt,dtype=torch.int32,device='cuda'); tww=torch.tensor(tww,dtype=torch.float32,device='cuda')
  return dict(x=x,W1=W1,W2=W2,ti=ti,tw=tw,W1p=W1p,W1e=W1e,W2p=W2p,W2e=W2e,W1pt=W1pt,W1et=W1et,W2pt=W2pt,W2et=W2et,sx=sx,xf=xf,te=te,tn=tn,tt=tt,tww=tww,NT=len(te.tolist()) if False else te.numel())
M,H,Is,E,topk=32,6144,512,384,8
d=build(M,H,Is,E,topk)
y=e.run(d['xf'],d['sx'].float().contiguous(),d['W1pt'],d['W1et'],d['W2pt'],d['W2et'],d['te'],d['tn'],d['tt'],d['tww'],M,Is,E)
# torch MoE ref (mxfp4-dequant)
ref=torch.zeros(M,H,device='cuda')
for i in range(M):
  for jj in range(topk):
    ee=int(d['ti'][i,jj]); w=float(d['tw'][i,jj])
    W1d=dequant_contig(d['W1p'][ee],d['W1e'][ee]); W2d=dequant_contig(d['W2p'][ee],d['W2e'][ee])
    h1=d['x'][i]@W1d.t(); g=h1[:Is];u=h1[Is:]; a=(g*torch.sigmoid(g))*u; ref[i]+=w*(a@W2d.t())
cos=torch.nn.functional.cosine_similarity(y.flatten(),ref.flatten(),dim=0).item()
print(f"MOE fp8-mma cos={cos:.5f} NT={d['te'].numel()} y_absmax={y.abs().max():.2f} ref={ref.abs().max():.2f}")
for _ in range(10): e.run(d['xf'],d['sx'].float().contiguous(),d['W1pt'],d['W1et'],d['W2pt'],d['W2et'],d['te'],d['tn'],d['tt'],d['tww'],M,Is,E)
torch.cuda.synchronize(); t0=time.time()
for _ in range(50): e.run(d['xf'],d['sx'].float().contiguous(),d['W1pt'],d['W1et'],d['W2pt'],d['W2et'],d['te'],d['tn'],d['tt'],d['tww'],M,Is,E)
torch.cuda.synchronize(); print(f"MOE fp8-mma runtime={((time.time()-t0)/50*1e3):.3f} ms (M={M} E={E})")
print("MOE_OK" if cos>0.99 else "MOE_WRONG")
