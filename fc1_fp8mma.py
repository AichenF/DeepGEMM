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

CUDA = r'''
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
__device__ __forceinline__ void fold_tbl(unsigned e8, unsigned& lo, unsigned& hi){
    const unsigned bLo=0x0c080000u,bHi=0x1c181410u;
    const unsigned off=(e8>=121u)?((e8-121u)<<3):0u,offp=off*0x01010101u;
    lo=(__vminu4(__vaddus4(bLo,offp),0x7e7e7e7eu)&0x7f7f7f7fu)&0xffffff00u;
    hi=__vminu4(__vaddus4(bHi,offp),0x7e7e7e7eu)&0x7f7f7f7fu;
}
// contiguous packed: byte b = nib(2b)|(nib(2b+1)<<4). 32 nibbles = 16 bytes = one 32-block (1 e8m0).
// fold 32 contiguous nibbles (packed16 bytes) with scale e8 -> 32 fp8 bytes into out[32].
__device__ __forceinline__ void fold32(const uint8_t* p16, unsigned e8, uint8_t* out){
    unsigned lo,hi; fold_tbl(e8,lo,hi);
    #pragma unroll
    for(int q=0;q<4;q++){                       // 4 uint32 = 32 nibbles
        unsigned w=reinterpret_cast<const unsigned*>(p16)[q];
        // nibbles: byte b -> low=nib(2b), high=nib(2b+1). 8 nibbles per uint32 in order n0..n7
        const unsigned sel=(w&7u)|((w>>4)&0x70u)|((w>>8)&0x700u)|((w>>12)&0x7000u);      // even nibbles n0,n2,n4,n6? no
        // do it per-nibble (8) explicitly, order: n0=low(b0),n1=high(b0),n2=low(b1),...
        #pragma unroll
        for(int j=0;j<8;j++){ unsigned nib=(w>>(4*j))&0xF; unsigned m=nib&7u;
            unsigned t=(m<4)?lo:hi; unsigned byte=(t>>(8*(m&3)))&0xff; if(nib&8) byte|=0x80u;
            out[q*8+j]=(uint8_t)byte; }
    }
}
extern "C" __global__ void fc1(const uint8_t* xf,const float* sx,const uint8_t* Wp,const uint8_t* We,
                               float* guo,int T,int H,int N){ // N=2Is
    const int t=threadIdx.x&31, gid=t>>2, tid=t&3, warp=threadIdx.x>>5, NW=blockDim.x>>5;
    extern __shared__ uint8_t sm[];
    uint8_t* xs=sm;                 // [16,H] fp8
    for(int i=threadIdx.x;i<16*H;i+=blockDim.x) xs[i]=(i/H<T)?xf[i]:0;
    __syncthreads();
    __shared__ uint8_t Bs[8][32*8]; // per-warp fold buffer [8 warps max][8n*32k]... use [warp][256]
    const int Hh=H>>1, Ge=H>>5;
    for(int n0=warp*8;n0<N;n0+=NW*8){
        float c0=0,c1=0,c2=0,c3=0;
        for(int k0=0;k0<H;k0+=32){
            // fold 8 rows x 32 cols -> Bs[warp] (8x32). threads 0..7 of warp fold rows.
            if(t<8){ fold32(Wp+(long)(n0+t)*Hh + (k0>>1), We[(long)(n0+t)*Ge+(k0>>5)], &Bs[warp][t*32]); }
            __syncwarp();
            unsigned a0=*reinterpret_cast<const unsigned*>(xs+gid*H+k0+tid*4);
            unsigned a1=*reinterpret_cast<const unsigned*>(xs+(gid+8)*H+k0+tid*4);
            unsigned a2=*reinterpret_cast<const unsigned*>(xs+gid*H+k0+tid*4+16);
            unsigned a3=*reinterpret_cast<const unsigned*>(xs+(gid+8)*H+k0+tid*4+16);
            unsigned b0=*reinterpret_cast<const unsigned*>(&Bs[warp][gid*32+tid*4]);
            unsigned b1=*reinterpret_cast<const unsigned*>(&Bs[warp][gid*32+tid*4+16]);
            asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        // store row gid, gid+8 ; cols n0+tid*2, +1 ; scale by sx[row]
        int r0=gid, r1=gid+8;
        if(r0<T){ guo[r0*N+n0+tid*2]=c0*sx[r0]; guo[r0*N+n0+tid*2+1]=c1*sx[r0]; }
        if(r1<T){ guo[r1*N+n0+tid*2]=c2*sx[r1]; guo[r1*N+n0+tid*2+1]=c3*sx[r1]; }
    }
}
torch::Tensor run(torch::Tensor xf,torch::Tensor sx,torch::Tensor Wp,torch::Tensor We,int T){
    int H=xf.size(1), N=Wp.size(0);
    auto guo=torch::zeros({T,N},torch::device(xf.device()).dtype(torch::kFloat32));
    int sh=16*H+8*256; int thr=256;
    cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,sh);
    fc1<<<1,thr,sh>>>(xf.data_ptr<uint8_t>(),sx.data_ptr<float>(),Wp.data_ptr<uint8_t>(),We.data_ptr<uint8_t>(),guo.data_ptr<float>(),T,H,N);
    return guo;
}
'''
e=load_inline(name='fc1_fp8mma',cpp_sources="torch::Tensor run(torch::Tensor xf,torch::Tensor sx,torch::Tensor Wp,torch::Tensor We,int T);",
              cuda_sources=CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
torch.manual_seed(0)
T,H,twoIs=16,6144,1024
x=torch.randn(T,H,device='cuda')
W1=(torch.randn(twoIs,H,device='cuda')*0.05)
Wp,We=quant_mxfp4_contig(W1)
# x -> fp8 per-token
sx=(x.abs().amax(1)/448.0).clamp_min(1e-30)
xf=(x/sx.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
guo=e.run(xf,sx.float().contiguous(),Wp,We,T)
ref=x @ dequant_contig(Wp,We).t()
cos=torch.nn.functional.cosine_similarity(guo.flatten(),ref.flatten(),dim=0).item()
print(f"FC1 fp8-mma cos={cos:.5f}  guo_absmax={guo.abs().max():.3f} ref_absmax={ref.abs().max():.3f}")
print("FC1_OK" if cos>0.99 else "FC1_WRONG")
