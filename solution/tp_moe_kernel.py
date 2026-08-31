"""TP MXFP4 MoE — per-rank partial FFN over the intermediate shard.

iter-2: replace the torch mxfp4 dequant (94% of runtime) with a fused CUDA
MXFP4->bf16 dequant kernel (Marlin nibble unpack + per-32 E8M0 power-of-two
scale, in one memory-bound pass). Matmul/SwiGLU stay bf16-torch for now.
"""
import importlib.util
import os

import torch
from torch.utils.cpp_extension import load_inline

os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_tp')  # home NFS quota is full

_REPO = '/home/xutingz/fac/DeepGEMM_tp'
_spec = importlib.util.spec_from_file_location(
    'quantization_mxfp4', os.path.join(_REPO, 'deep_gemm', 'quantization_mxfp4.py'))
_qm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qm)

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__constant__ float kFP4V[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__device__ __forceinline__ float nib2val(unsigned nib) {
    float v = kFP4V[nib & 7];
    return (nib & 8) ? -v : v;
}

// Vectorized: one thread per 8-nibble chunk. Coalesced uint32 read (4 packed
// bytes = 8 nibbles) + uint4 write (8 bf16). Marlin order within a chunk:
// out[0..3]=high(b0..b3), out[4..7]=low(b0..b3). One E8M0 scale per chunk
// (8 elems within one 32-block); scale = 2^(code-127).
__global__ void mxfp4_dequant_bf16_kernel(
        const uint32_t* __restrict__ packed32, const uint8_t* __restrict__ scale,
        __nv_bfloat16* __restrict__ out, long nchunks, int cpr) {  // cpr = chunks-per-row = K/8
    const int spr = cpr >> 2;  // scale bytes per row = K/32
    const long K = (long)cpr * 8;
    for (long c = (long)blockIdx.x * blockDim.x + threadIdx.x; c < nchunks;
         c += (long)gridDim.x * blockDim.x) {
        const long n = c / cpr;
        const int cir = (int)(c - n * cpr);
        const uint32_t w = packed32[c];
        const unsigned sbits = ((unsigned)scale[n * spr + (cir >> 2)]) << 23;
        const float s = __int_as_float(sbits);
        const unsigned b0 = w & 0xFF, b1 = (w >> 8) & 0xFF, b2 = (w >> 16) & 0xFF, b3 = (w >> 24) & 0xFF;
        __nv_bfloat16 o[8];
        o[0] = __float2bfloat16(nib2val(b0 >> 4) * s);
        o[1] = __float2bfloat16(nib2val(b1 >> 4) * s);
        o[2] = __float2bfloat16(nib2val(b2 >> 4) * s);
        o[3] = __float2bfloat16(nib2val(b3 >> 4) * s);
        o[4] = __float2bfloat16(nib2val(b0 & 0xF) * s);
        o[5] = __float2bfloat16(nib2val(b1 & 0xF) * s);
        o[6] = __float2bfloat16(nib2val(b2 & 0xF) * s);
        o[7] = __float2bfloat16(nib2val(b3 & 0xF) * s);
        *reinterpret_cast<uint4*>(out + n * K + (long)cir * 8) = *reinterpret_cast<uint4*>(o);
    }
}

torch::Tensor mxfp4_dequant_bf16(torch::Tensor packed, torch::Tensor scale) {
    TORCH_CHECK(packed.is_cuda() && packed.dtype() == torch::kUInt8 && packed.is_contiguous());
    TORCH_CHECK(scale.is_cuda() && scale.dtype() == torch::kUInt8 && scale.is_contiguous());
    const long N = packed.size(0), K = packed.size(1) * 2;
    auto out = torch::empty({N, K}, torch::device(packed.device()).dtype(torch::kBFloat16));
    const int cpr = (int)(K >> 3);
    const long nchunks = N * cpr;
    const int threads = 256;
    const int blocks = (int)std::min<long>((nchunks + threads - 1) / threads, 65535L);
    mxfp4_dequant_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>()),
        scale.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nchunks, cpr);
    return out;
}
"""

_ext = load_inline(
    name='tp_mxfp4_dequant_v2',
    cpp_sources="torch::Tensor mxfp4_dequant_bf16(torch::Tensor packed, torch::Tensor scale);",
    cuda_sources=_CUDA,
    functions=['mxfp4_dequant_bf16'],
    extra_cuda_cflags=['-O3'],
    verbose=False,
)


def _dq_bf16(packed, scale, N, K):
    return _ext.mxfp4_dequant_bf16(packed.reshape(N, K // 2).contiguous(),
                                   scale.reshape(N, K // 32).contiguous())


def tp_moe_partial(x, l1_packed, l1_scale, l2_packed, l2_scale, topk_idx, topk_w, Is):
    M, H = x.shape
    topk = topk_idx.shape[1]
    dt = torch.bfloat16

    flat_e = topk_idx.reshape(-1).long()
    flat_w = topk_w.reshape(-1).to(torch.float32)
    uniq, inv = torch.unique(flat_e, return_inverse=True)
    U = uniq.numel()

    W1 = _dq_bf16(l1_packed[uniq], l1_scale[uniq], U * 2 * Is, H).view(U, 2 * Is, H)
    W2 = _dq_bf16(l2_packed[uniq], l2_scale[uniq], U * H, Is).view(U, H, Is)
    gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]

    xt = x.to(dt).unsqueeze(1).expand(M, topk, H).reshape(-1, H)
    g = torch.einsum('nh,nih->ni', xt, gate_w[inv])
    u = torch.einsum('nh,nih->ni', xt, up_w[inv])
    gf = g.float()
    a = ((gf * torch.sigmoid(gf)) * u.float()).to(dt)
    yo = torch.einsum('ni,nhi->nh', a, W2[inv]).float() * flat_w.unsqueeze(1)

    y = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    row = torch.arange(M, device=x.device).repeat_interleave(topk)
    y.index_add_(0, row, yo)
    return y
