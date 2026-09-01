"""Variant: vectorized CUDA MXFP4->bf16 dequant (memory-speed) of touched experts,
then torch matmul. Tests whether decoupling a fast dequant from the matmul beats
the fused inline-scalar-dequant kernel at tp=4 (larger per-rank shards)."""
import importlib.util, os
import torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_mm')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__constant__ float kFP4V[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
__device__ __forceinline__ float dqv(unsigned nib){ float v=kFP4V[nib&7]; return (nib&8)?-v:v; }
__global__ void deq(const uint32_t* __restrict__ p32, const uint8_t* __restrict__ sc,
                    __nv_bfloat16* __restrict__ out, long nchunks, int cpr){
    const int spr = cpr>>2; const long K=(long)cpr*8;
    for(long c=(long)blockIdx.x*blockDim.x+threadIdx.x;c<nchunks;c+=(long)gridDim.x*blockDim.x){
        const long n=c/cpr; const int cir=(int)(c-n*cpr);
        const uint32_t w=p32[c]; const unsigned sb=((unsigned)sc[n*spr+(cir>>2)])<<23; const float s=__int_as_float(sb);
        const unsigned b0=w&0xFF,b1=(w>>8)&0xFF,b2=(w>>16)&0xFF,b3=(w>>24)&0xFF;
        __nv_bfloat16 o[8];
        o[0]=__float2bfloat16(dqv(b0>>4)*s);o[1]=__float2bfloat16(dqv(b1>>4)*s);
        o[2]=__float2bfloat16(dqv(b2>>4)*s);o[3]=__float2bfloat16(dqv(b3>>4)*s);
        o[4]=__float2bfloat16(dqv(b0&0xF)*s);o[5]=__float2bfloat16(dqv(b1&0xF)*s);
        o[6]=__float2bfloat16(dqv(b2&0xF)*s);o[7]=__float2bfloat16(dqv(b3&0xF)*s);
        *reinterpret_cast<uint4*>(out+n*K+(long)cir*8)=*reinterpret_cast<uint4*>(o);
    }
}
torch::Tensor mxfp4_dequant_bf16(torch::Tensor packed, torch::Tensor scale){
    const long N=packed.size(0), K=packed.size(1)*2;
    auto out=torch::empty({N,K},torch::device(packed.device()).dtype(torch::kBFloat16));
    const int cpr=(int)(K>>3); const long nchunks=N*cpr; const int th=256;
    const int bl=(int)std::min<long>((nchunks+th-1)/th,65535L);
    deq<<<bl,th>>>(reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>()),scale.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),nchunks,cpr);
    return out;
}
"""
_ext = load_inline(name='tp_dq_only', cpp_sources="torch::Tensor mxfp4_dequant_bf16(torch::Tensor packed, torch::Tensor scale);",
                   cuda_sources=_CUDA, functions=['mxfp4_dequant_bf16'], extra_cuda_cflags=['-O3'], verbose=False)


def _dq(p, s, N, K):
    return _ext.mxfp4_dequant_bf16(p.reshape(N, K // 2).contiguous(), s.reshape(N, K // 32).contiguous())


def tp_moe_partial(x, l1p, l1s, l2p, l2s, topk_idx, topk_w, Is):
    M, H = x.shape; topk = topk_idx.shape[1]; dt = torch.bfloat16
    fe = topk_idx.reshape(-1).long(); fw = topk_w.reshape(-1).float()
    uniq, inv = torch.unique(fe, return_inverse=True); U = uniq.numel()
    W1 = _dq(l1p[uniq], l1s[uniq], U * 2 * Is, H).view(U, 2 * Is, H)
    W2 = _dq(l2p[uniq], l2s[uniq], U * H, Is).view(U, H, Is)
    gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]
    xt = x.to(dt).unsqueeze(1).expand(M, topk, H).reshape(-1, H)
    g = torch.einsum('nh,nih->ni', xt, gate_w[inv])
    u = torch.einsum('nh,nih->ni', xt, up_w[inv])
    gf = g.float(); a = ((gf * torch.sigmoid(gf)) * u.float()).to(dt)
    yo = torch.einsum('ni,nhi->nh', a, W2[inv]).float() * fw.unsqueeze(1)
    y = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    y.index_add_(0, torch.arange(M, device=x.device).repeat_interleave(topk), yo)
    return y
