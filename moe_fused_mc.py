"""3-kernel split fp8-mma MoE: FC1 (x@W1->gu) / SwiGLU (gu->act) / FC2 (act@W2->y).
Each kernel carries tiny smem -> high occupancy (FC1 4 blk/SM @48KB, FC2 8+ @4KB) -> saturate DRAM.
Register-resident B (fold4). tile-major weights. Verify cos vs torch mxfp4 ref."""
import os, torch, time
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0'
FP4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device='cuda')
def quant_mxfp4_contig(W):
  N,H=W.shape; G=H//32
  w=W.float().view(N,G,32); amax=w.abs().amax(-1,keepdim=True).clamp_min(1e-30)
  e8=torch.ceil(torch.log2(amax/6.0)).clamp(-127,127); scale=torch.exp2(e8)
  e8b=(e8+127).to(torch.uint8).view(N,G); wn=(w/scale).clamp(-6,6)
  bnd=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.0],device=W.device)
  mag=torch.bucketize(wn.abs(),bnd); sign=(wn<0).to(torch.uint8)
  nib=((sign<<3)|mag.to(torch.uint8)).view(N,H)
  packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8).contiguous()
  return packed,e8b.contiguous()
def dequant_contig(packed,e8b):
  N,Hh=packed.shape; H=Hh*2; G=e8b.shape[1]
  lo=packed&0xF; hi=(packed>>4)&0xF; nib=torch.stack([lo,hi],-1).view(N,H)
  mag=FP4.to(packed.device)[(nib&7).long()]; val=torch.where((nib>>3&1).bool(),-mag,mag)
  scale=torch.exp2((e8b.int()-127).float()).view(N,G,1).expand(N,G,32).reshape(N,H)
  return val*scale
def tile_major(packed,e8b):  # [E,N,H/2],[E,N,H/32] -> contiguous [E,N/8,H/32,8,16],[E,N/8,H/32,8]
  E,N,Hh=packed.shape; H=Hh*2; nb=N//8; kb=H//32
  p=packed.view(E,nb,8,kb,16).permute(0,1,3,2,4).contiguous().view(E,nb*kb*8*16)
  ee=e8b.view(E,nb,8,kb,1).permute(0,1,3,2,4).contiguous().view(E,nb*kb*8)
  return p,ee

CUDA=r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void fold_tbl(unsigned e8,unsigned& lo,unsigned& hi){
  const unsigned bLo=0x0c080000u,bHi=0x1c181410u;
  const unsigned off=(e8>=121u)?((e8-121u)<<3):0u,offp=off*0x01010101u;
  lo=(__vminu4(__vaddus4(bLo,offp),0x7e7e7e7eu)&0x7f7f7f7fu)&0xffffff00u;
  hi=__vminu4(__vaddus4(bHi,offp),0x7e7e7e7eu)&0x7f7f7f7fu; }
__device__ __forceinline__ unsigned fold4(unsigned h,unsigned lo,unsigned hi){
  unsigned o,sel=h&0x7777u; asm("prmt.b32 %0,%1,%2,%3;":"=r"(o):"r"(lo),"r"(hi),"r"(sel));
  return o|((h&8u)<<4)|((h&0x80u)<<8)|((h&0x800u)<<12)|((h&0x8000u)<<16); }
#define MMA(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1) asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

extern "C" __global__ void fc1(const uint8_t* xall,const float* sxall,const uint8_t* W1p,const uint8_t* W1e,
    const int* te,const int* tn,const int* ttok, float* gu,int H,int Is,int NB1){
  const int bt=blockIdx.x/NB1, nblk=blockIdx.x%NB1; const int e=te[bt],T=tn[bt];
  const int t=threadIdx.x&31,gid=t>>2,tid=t&3,warp=threadIdx.x>>5,NW=blockDim.x>>5;
  const int twoIs=2*Is, BN1=twoIs/NB1;
  extern __shared__ uint8_t sm[]; uint8_t* xs=sm;
  __shared__ int tok[4]; __shared__ float lsx[4]; __shared__ unsigned fLUT[512];
  for(int i=threadIdx.x;i<256;i+=blockDim.x){ unsigned lo,hi; fold_tbl((unsigned)i,lo,hi); fLUT[i*2]=lo; fLUT[i*2+1]=hi; }
  if(threadIdx.x<4){ int r=threadIdx.x; int tk=(r<T)?ttok[bt*4+r]:-1; tok[r]=tk; lsx[r]=(tk>=0)?sxall[tk]:0.f; }
  __syncthreads();
  for(int i=threadIdx.x;i<4*H;i+=blockDim.x){ int r=i/H,k=i%H; xs[i]=(r<T)?xall[(long)tok[r]*H+k]:0; }
  __syncthreads();
  const int n1=nblk*BN1;
  for(int n0=n1+warp*8;n0<n1+BN1;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    const long nb=((long)e*(twoIs>>3)+(n0>>3))*(H>>5);
    for(int k0=0;k0<H;k0+=32){
      const long tb=nb+(k0>>5); const uint8_t* wr=W1p + tb*128 + gid*16;
      unsigned h0=*(const unsigned short*)(wr+tid*2), h1=*(const unsigned short*)(wr+tid*2+8);
      unsigned e8=(unsigned)W1e[tb*8+gid], lo=fLUT[e8*2], hi=fLUT[e8*2+1];
      unsigned b0=fold4(h0,lo,hi), b1=fold4(h1,lo,hi);
      unsigned a0=(gid<4)?*(const unsigned*)(xs+gid*H+k0+tid*4):0u,a1=0u;
      unsigned a2=(gid<4)?*(const unsigned*)(xs+gid*H+k0+tid*4+16):0u,a3=0u;
      MMA(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1); }
    if(gid<T){ gu[((long)bt*4+gid)*twoIs+n0+tid*2]=c0*lsx[gid]; gu[((long)bt*4+gid)*twoIs+n0+tid*2+1]=c1*lsx[gid]; } }
}
extern "C" __global__ void swig(const float* gu, uint8_t* act, float* asc, const int* tn, int Is){
  const int bt=blockIdx.x, T=tn[bt], twoIs=2*Is;
  const int warp=threadIdx.x>>5, NW=blockDim.x>>5, lane=threadIdx.x&31;
  for(int r=warp;r<T;r+=NW){ float amax=1e-30f;
    for(int i=lane;i<Is;i+=32){ float g=gu[((long)bt*4+r)*twoIs+i],u=gu[((long)bt*4+r)*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u; amax=fmaxf(amax,fabsf(a)); }
    #pragma unroll
    for(int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(-1u,amax,o)); amax=__shfl_sync(-1u,amax,0);
    float s=amax/448.f; if(lane==0) asc[bt*4+r]=s;
    for(int i=lane;i<Is;i+=32){ float g=gu[((long)bt*4+r)*twoIs+i],u=gu[((long)bt*4+r)*twoIs+Is+i]; float a=(g/(1.f+__expf(-g)))*u; act[((long)bt*4+r)*Is+i]=(uint8_t)__nv_cvt_float_to_fp8(a/s,__NV_SATFINITE,__NV_E4M3); } }
}
extern "C" __global__ void fc2(const uint8_t* act,const float* asc,const uint8_t* W2p,const uint8_t* W2e,
    const int* te,const int* tn,const int* ttok,const float* tw, float* y,int H,int Is,int NB2){
  const int bt=blockIdx.x/NB2, nblk=blockIdx.x%NB2; const int e=te[bt],T=tn[bt];
  const int t=threadIdx.x&31,gid=t>>2,tid=t&3,warp=threadIdx.x>>5,NW=blockDim.x>>5;
  const int BN2=H/NB2;
  extern __shared__ uint8_t sm[]; uint8_t* af=sm;
  __shared__ int tok[4]; __shared__ float rw[4], ascs[4]; __shared__ unsigned fLUT[512];
  for(int i=threadIdx.x;i<256;i+=blockDim.x){ unsigned lo,hi; fold_tbl((unsigned)i,lo,hi); fLUT[i*2]=lo; fLUT[i*2+1]=hi; }
  if(threadIdx.x<4){ int r=threadIdx.x; tok[r]=(r<T)?ttok[bt*4+r]:-1; rw[r]=(r<T)?tw[bt*4+r]:0.f; ascs[r]=(r<T)?asc[bt*4+r]:0.f; }
  __syncthreads();
  for(int i=threadIdx.x;i<4*Is;i+=blockDim.x){ int r=i/Is; af[i]=(r<T)?act[(long)bt*4*Is+i]:0; }
  __syncthreads();
  const int n1=nblk*BN2;
  for(int n0=n1+warp*8;n0<n1+BN2;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    const long nb=((long)e*(H>>3)+(n0>>3))*(Is>>5);
    for(int k0=0;k0<Is;k0+=32){
      const long tb=nb+(k0>>5); const uint8_t* wr=W2p + tb*128 + gid*16;
      unsigned h0=*(const unsigned short*)(wr+tid*2), h1=*(const unsigned short*)(wr+tid*2+8);
      unsigned e8=(unsigned)W2e[tb*8+gid], lo=fLUT[e8*2], hi=fLUT[e8*2+1];
      unsigned b0=fold4(h0,lo,hi), b1=fold4(h1,lo,hi);
      unsigned a0=(gid<4)?*(const unsigned*)(af+gid*Is+k0+tid*4):0u,a1=0u;
      unsigned a2=(gid<4)?*(const unsigned*)(af+gid*Is+k0+tid*4+16):0u,a3=0u;
      MMA(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1); }
    if(gid<T){ int tk=tok[gid]; float w=rw[gid]*ascs[gid]; atomicAdd(&y[(long)tk*H+n0+tid*2],c0*w); atomicAdd(&y[(long)tk*H+n0+tid*2+1],c1*w); } }
}

extern "C" __global__ void fc2_mc(const uint8_t* act,const float* asc,const uint8_t* W2p,const uint8_t* W2e,
    const int* te,const int* tn,const int* ttok,const float* tw, unsigned long long y_mc,int H,int Is,int NB2){
  const int bt=blockIdx.x/NB2, nblk=blockIdx.x%NB2; const int e=te[bt],T=tn[bt];
  const int t=threadIdx.x&31,gid=t>>2,tid=t&3,warp=threadIdx.x>>5,NW=blockDim.x>>5;
  const int BN2=H/NB2;
  extern __shared__ uint8_t sm[]; uint8_t* af=sm;
  __shared__ int tok[4]; __shared__ float rw[4], ascs[4]; __shared__ unsigned fLUT[512];
  for(int i=threadIdx.x;i<256;i+=blockDim.x){ unsigned lo,hi; fold_tbl((unsigned)i,lo,hi); fLUT[i*2]=lo; fLUT[i*2+1]=hi; }
  if(threadIdx.x<4){ int r=threadIdx.x; tok[r]=(r<T)?ttok[bt*4+r]:-1; rw[r]=(r<T)?tw[bt*4+r]:0.f; ascs[r]=(r<T)?asc[bt*4+r]:0.f; }
  __syncthreads();
  for(int i=threadIdx.x;i<4*Is;i+=blockDim.x){ int r=i/Is; af[i]=(r<T)?act[(long)bt*4*Is+i]:0; }
  __syncthreads();
  const int n1=nblk*BN2;
  for(int n0=n1+warp*8;n0<n1+BN2;n0+=NW*8){ float c0=0,c1=0,c2=0,c3=0;
    const long nb=((long)e*(H>>3)+(n0>>3))*(Is>>5);
    for(int k0=0;k0<Is;k0+=32){
      const long tb=nb+(k0>>5); const uint8_t* wr=W2p + tb*128 + gid*16;
      unsigned h0=*(const unsigned short*)(wr+tid*2), h1=*(const unsigned short*)(wr+tid*2+8);
      unsigned e8=(unsigned)W2e[tb*8+gid], lo=fLUT[e8*2], hi=fLUT[e8*2+1];
      unsigned b0=fold4(h0,lo,hi), b1=fold4(h1,lo,hi);
      unsigned a0=(gid<4)?*(const unsigned*)(af+gid*Is+k0+tid*4):0u,a1=0u;
      unsigned a2=(gid<4)?*(const unsigned*)(af+gid*Is+k0+tid*4+16):0u,a3=0u;
      MMA(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1); }
    if(gid<T){ int tk=tok[gid]; float w=rw[gid]*ascs[gid];
      unsigned long long a0=y_mc+(unsigned long long)((long)tk*H+n0+tid*2)*4;
      float v0=c0*w, v1=c1*w;
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"::"l"(a0),"f"(v0));
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"::"l"(a0+4ULL),"f"(v1)); } }
}
void run_y(torch::Tensor y, torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,
                  torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2){
  int H=xall.size(1),NT=te.size(0); int twoIs=2*Is;
  auto gu=torch::empty({(long)NT*4,twoIs},torch::device(xall.device()).dtype(torch::kFloat32));
  auto act=torch::empty({(long)NT*4,Is},torch::device(xall.device()).dtype(torch::kUInt8));
  auto asc=torch::empty({(long)NT*4},torch::device(xall.device()).dtype(torch::kFloat32));
  int thr=256; long sh1=(long)4*H, sh2=(long)4*Is;
  cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh1);
  cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh2);
  fc1<<<NT*NB1,thr,sh1>>>(xall.data_ptr<uint8_t>(),sxall.data_ptr<float>(),W1p.data_ptr<uint8_t>(),W1e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),gu.data_ptr<float>(),H,Is,NB1);
  swig<<<NT,thr>>>(gu.data_ptr<float>(),act.data_ptr<uint8_t>(),asc.data_ptr<float>(),tn.data_ptr<int>(),Is);
  fc2<<<NT*NB2,thr,sh2>>>(act.data_ptr<uint8_t>(),asc.data_ptr<float>(),W2p.data_ptr<uint8_t>(),W2e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),tw.data_ptr<float>(),y.data_ptr<float>(),H,Is,NB2);
}
void run_ymc(long long y_mc, torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,
                  torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2){
  int H=xall.size(1),NT=te.size(0); int twoIs=2*Is;
  auto gu=torch::empty({(long)NT*4,twoIs},torch::device(xall.device()).dtype(torch::kFloat32));
  auto act=torch::empty({(long)NT*4,Is},torch::device(xall.device()).dtype(torch::kUInt8));
  auto asc=torch::empty({(long)NT*4},torch::device(xall.device()).dtype(torch::kFloat32));
  int thr=256; long sh1=(long)4*H, sh2=(long)4*Is;
  cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh1);
  cudaFuncSetAttribute(fc2_mc,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh2);
  fc1<<<NT*NB1,thr,sh1>>>(xall.data_ptr<uint8_t>(),sxall.data_ptr<float>(),W1p.data_ptr<uint8_t>(),W1e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),gu.data_ptr<float>(),H,Is,NB1);
  swig<<<NT,thr>>>(gu.data_ptr<float>(),act.data_ptr<uint8_t>(),asc.data_ptr<float>(),tn.data_ptr<int>(),Is);
  fc2_mc<<<NT*NB2,thr,sh2>>>(act.data_ptr<uint8_t>(),asc.data_ptr<float>(),W2p.data_ptr<uint8_t>(),W2e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),tw.data_ptr<float>(),(unsigned long long)y_mc,H,Is,NB2);
}
torch::Tensor run(torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,
                  torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2){
  int H=xall.size(1),NT=te.size(0); int twoIs=2*Is;
  auto y=torch::zeros({M,H},torch::device(xall.device()).dtype(torch::kFloat32));
  auto gu=torch::empty({(long)NT*4,twoIs},torch::device(xall.device()).dtype(torch::kFloat32));
  auto act=torch::empty({(long)NT*4,Is},torch::device(xall.device()).dtype(torch::kUInt8));
  auto asc=torch::empty({(long)NT*4},torch::device(xall.device()).dtype(torch::kFloat32));
  int thr=256; long sh1=(long)4*H, sh2=(long)4*Is;
  cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh1);
  cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh2);
  fc1<<<NT*NB1,thr,sh1>>>(xall.data_ptr<uint8_t>(),sxall.data_ptr<float>(),W1p.data_ptr<uint8_t>(),W1e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),gu.data_ptr<float>(),H,Is,NB1);
  swig<<<NT,thr>>>(gu.data_ptr<float>(),act.data_ptr<uint8_t>(),asc.data_ptr<float>(),tn.data_ptr<int>(),Is);
  fc2<<<NT*NB2,thr,sh2>>>(act.data_ptr<uint8_t>(),asc.data_ptr<float>(),W2p.data_ptr<uint8_t>(),W2e.data_ptr<uint8_t>(),
    te.data_ptr<int>(),tn.data_ptr<int>(),ttok.data_ptr<int>(),tw.data_ptr<float>(),y.data_ptr<float>(),H,Is,NB2);
  return y; }
"""
e=load_inline(name='moe_fp8mma_symm',
  cpp_sources="void run_ymc(long long y_mc,torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2);\nvoid run_y(torch::Tensor y,torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2);\ntorch::Tensor run(torch::Tensor xall,torch::Tensor sxall,torch::Tensor W1p,torch::Tensor W1e,torch::Tensor W2p,torch::Tensor W2e,torch::Tensor te,torch::Tensor tn,torch::Tensor ttok,torch::Tensor tw,int M,int Is,int E,int NB1,int NB2);",
  cuda_sources=CUDA,functions=['run_ymc','run_y','run'],extra_cuda_cflags=['-O3'],verbose=False)

import torch.distributed as dist, torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symm_mem

def group_pairs(ti,tw,M,topk,dev):
  fe=ti.reshape(-1); ft=torch.arange(M,device=dev).repeat_interleave(topk); fw=tw.reshape(-1)
  order=torch.argsort(fe); fe,ft,fw=fe[order],ft[order],fw[order]
  te=[];tn=[];tt=[];tww=[]; i=0;P=fe.numel()
  while i<P:
    j=i
    while j<P and fe[j]==fe[i] and j-i<4: j+=1
    te.append(int(fe[i]));n=j-i;tn.append(n); tt+=ft[i:j].tolist()+[0]*(4-n); tww+=fw[i:j].tolist()+[0.]*(4-n); i=j
  return (torch.tensor(te,dtype=torch.int32,device=dev),torch.tensor(tn,dtype=torch.int32,device=dev),
          torch.tensor(tt,dtype=torch.int32,device=dev),torch.tensor(tww,dtype=torch.float32,device=dev))

def worker(rank, world, port, M, H, I, E, topk):
  os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(port)
  dist.init_process_group('nccl',world_size=world,rank=rank,device_id=torch.device(f'cuda:{rank}'))
  torch.cuda.set_device(rank); dev=torch.device('cuda',rank); Is=I//world
  gname=dist.group.WORLD.group_name
  try: symm_mem.enable_symm_mem_for_group(gname)
  except Exception: pass
  # global model (seed-synced across ranks), then slice this rank's I-shard
  torch.manual_seed(0)
  x=torch.randn(M,H,device=dev)
  W1g=(torch.randn(E,2*I,H,device=dev)*0.05); W2g=(torch.randn(E,H,I,device=dev)*0.05)
  sc=torch.randn(M,E,device=dev); tw,ti=torch.topk(sc,topk,dim=-1)
  gate_r=W1g[:, rank*Is:(rank+1)*Is, :]; up_r=W1g[:, I+rank*Is:I+(rank+1)*Is, :]
  W1r=torch.cat([gate_r,up_r],dim=1).contiguous(); W2r=W2g[:, :, rank*Is:(rank+1)*Is].contiguous()
  W1p=[];W1e=[];W2p=[];W2e=[]
  for i in range(E): p,ee=quant_mxfp4_contig(W1r[i]);W1p.append(p);W1e.append(ee);p,ee=quant_mxfp4_contig(W2r[i]);W2p.append(p);W2e.append(ee)
  W1p=torch.stack(W1p);W1e=torch.stack(W1e);W2p=torch.stack(W2p);W2e=torch.stack(W2e)
  W1pt,W1et=tile_major(W1p,W1e);W2pt,W2et=tile_major(W2p,W2e)
  sx=(x.abs().amax(1)/448.).clamp_min(1e-30); xf=(x/sx.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
  te,tn,tt,tww=group_pairs(ti,tw,M,topk,dev); sxf=sx.float().contiguous()
  # ---- torch partial reference (dequant this shard) ----
  refp=torch.zeros(M,H,device=dev)
  for i in range(M):
    for jj in range(topk):
      ee=int(ti[i,jj]); w=float(tw[i,jj]); W1d=dequant_contig(W1p[ee],W1e[ee]); W2d=dequant_contig(W2p[ee],W2e[ee])
      h1=x[i]@W1d.t(); g=h1[:Is];u=h1[Is:]; a=(g*torch.sigmoid(g))*u; refp[i]+=w*(a@W2d.t())
  ref=refp.clone(); dist.all_reduce(ref,op=dist.ReduceOp.SUM)
  # ---- kernel partial into SYMMETRIC buffer + one-shot NVLink all-reduce ----
  ybuf=symm_mem.empty((M,H),dtype=torch.float32,device=dev); hdl=symm_mem.rendezvous(ybuf,gname)
  mc=hdl.multicast_ptr
  NB1,NB2=8,16
  ybuf.zero_(); e.run_y(ybuf,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2)
  yout=torch.ops.symm_mem.one_shot_all_reduce(ybuf,"sum",gname); torch.cuda.synchronize()
  cos=torch.nn.functional.cosine_similarity(yout.flatten(),ref.flatten(),dim=0).item()
  # ---- FUSED: FC2 epilogue does multimem.red.add into multicast buffer = scatter + all-reduce in one kernel ----
  ybuf.zero_(); dist.barrier(); torch.cuda.synchronize()
  e.run_ymc(mc,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2)
  torch.cuda.synchronize(); dist.barrier()
  cosf=torch.nn.functional.cosine_similarity(ybuf.flatten(),ref.flatten(),dim=0).item()
  # ---- timing: 2-step (compute + symm one-shot) vs FUSED (compute+multimem) vs NCCL ----
  def full_symm():
    ybuf.zero_(); e.run_y(ybuf,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2)
    torch.ops.symm_mem.one_shot_all_reduce(ybuf,"sum",gname)
  def full_fused():
    ybuf.zero_(); dist.barrier(); e.run_ymc(mc,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2); dist.barrier()
  def full_nccl():
    yy=e.run(xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2); dist.all_reduce(yy,op=dist.ReduceOp.SUM)
  def bench(fn,it=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); dist.barrier(); t0=time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/it*1e3
  def k_fused(): e.run_ymc(mc,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2)
  def k_2step(): ybuf.zero_(); e.run_y(ybuf,xf,sxf,W1pt,W1et,W2pt,W2et,te,tn,tt,tww,M,Is,E,NB1,NB2)
  ts=bench(full_symm); tf=bench(full_fused); tn_=bench(full_nccl); tkf=bench(k_fused); tk2=bench(k_2step)
  tms=torch.tensor([ts,tf,tn_,tkf,tk2],device=dev); dist.all_reduce(tms,op=dist.ReduceOp.MAX)
  if rank==0:
    print(f"TP-MoE: cos_2step={cos:.5f} cos_FUSED={cosf:.5f}  M={M} E={E} I={I} tp={world} Is={Is}",flush=True)
    print(f"  2-step  (compute + symm one-shot allreduce): {tms[0].item():.4f} ms",flush=True)
    print(f"  FUSED   (FC2 epilogue multimem.red.add)    : {tms[1].item():.4f} ms",flush=True)
    print(f"  NCCL    (compute + dist.all_reduce)        : {tms[2].item():.4f} ms",flush=True)
    print(f"  [kernel-only] fused-multimem-FC2 {tms[3].item():.4f} ms  vs  2step-atomicAdd-FC2+zero {tms[4].item():.4f} ms",flush=True)
    print("TP_FUSED_OK" if (cos>0.99 and cosf>0.99) else "TP_FUSED_WRONG",flush=True)
  dist.barrier(); os._exit(0)

if __name__=='__main__':
  M=int(os.environ.get('MM','32')); world=int(os.environ.get('TP','4'))
  E=int(os.environ.get('EE','64')); port=int(os.environ.get('MASTER_PORT','22800'))
  mp.spawn(worker,args=(world,port,M,6144,2048,E,8),nprocs=world,join=True)
