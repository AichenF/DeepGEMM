import os, torch, time
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0'
CUDA=r'''
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
// E8M0-fold: each nibble -> fp8 e4m3 via magnitude byte + (e8m0-121)<<3 saturating add (Humming-style).
// out fp8. Measures whether fold is cheap. packed [N,K/2], e8m0 [N,K/32].
__global__ void fold(const uint32_t* p32, const uint8_t* sc, uint8_t* out, long nch, int cpr){
  const int spr=cpr>>2; const long K=(long)cpr*8;
  // unscaled e2m1 magnitude -> fp8 e4m3 bytes (value*2^-6): {0,0,8,12,16,20,24,28} for mag 0..7 (mag0,1 share base per Humming)
  const unsigned baseLo=0x0c080000u, baseHi=0x1c181410u;
  for(long c=(long)blockIdx.x*blockDim.x+threadIdx.x;c<nch;c+=(long)gridDim.x*blockDim.x){
    const long n=c/cpr; const int cir=(int)(c-n*cpr);
    const uint32_t w=p32[c]; const unsigned e8=sc[n*spr+(cir>>2)];
    // fold offset (e8-121)<<3, saturating add to magnitude bytes
    const unsigned off = (e8>=121u)?((e8-121u)<<3):0u; const unsigned offp=off|(off<<8)|(off<<16)|(off<<24);
    unsigned lo=__vminu4(__vaddus4(baseLo,offp),0x7e7e7e7eu)&0x7f7f7f7fu;
    unsigned hi=__vminu4(__vaddus4(baseHi,offp),0x7e7e7e7eu)&0x7f7f7f7fu; lo&=0xffffff00u;
    // select 8 fp8 bytes by nibble via byte_perm (2 prmt) + sign
    const unsigned uq=w;
    const unsigned selhi=((uq>>4)&7)|((uq>>8)&0x70)|((uq>>12)&0x700)|((uq>>16)&0x7000);
    const unsigned sello=(uq&7)|((uq>>4)&0x70)|((uq>>8)&0x700)|((uq>>12)&0x7000);
    unsigned oh,ol; asm("prmt.b32 %0,%1,%2,%3;":"=r"(oh):"r"(lo),"r"(hi),"r"(selhi));
    asm("prmt.b32 %0,%1,%2,%3;":"=r"(ol):"r"(lo),"r"(hi),"r"(sello));
    oh|=uq&0x80808080u; ol|=(uq<<4)&0x80808080u;
    *reinterpret_cast<uint2*>(out+n*K+(long)cir*8)=make_uint2(ol,oh);
  }
}
torch::Tensor f(torch::Tensor p, torch::Tensor s){ long N=p.size(0),K=p.size(1)*2; auto o=torch::empty({N,K},torch::device(p.device()).dtype(torch::kUInt8));
  int cpr=(int)(K>>3); long nch=N*cpr; int th=256; int bl=(int)std::min<long>((nch+th-1)/th,65535L);
  fold<<<bl,th>>>(reinterpret_cast<const uint32_t*>(p.data_ptr<uint8_t>()),s.data_ptr<uint8_t>(),o.data_ptr<uint8_t>(),nch,cpr); return o; }
'''
e=load_inline(name='foldp',cpp_sources="torch::Tensor f(torch::Tensor p, torch::Tensor s);",cuda_sources=CUDA,functions=['f'],extra_cuda_cflags=['-O3'],verbose=False)
N,K=384*1024, 6144  # ~ FC1 weights for many experts
p=torch.randint(0,255,(N,K//2),dtype=torch.uint8,device='cuda'); s=torch.randint(100,140,(N,K//32),dtype=torch.uint8,device='cuda')
for _ in range(5): e.f(p,s)
torch.cuda.synchronize(); t=time.time()
for _ in range(20): o=e.f(p,s)
torch.cuda.synchronize(); dt=(time.time()-t)/20*1e3
gb=(p.numel()+o.numel())/1e9
print(f"fp8-fold dequant: {dt:.3f} ms  ({N*K/1e9:.2f}G nibbles)  {gb/(dt/1e3):.0f} GB/s traffic  ({gb:.2f}GB rd+wr)")
