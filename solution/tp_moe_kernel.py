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

// packed [N, K/2] uint8 (Marlin: within an 8-elem chunk, out[0..3]=high of bytes
// 0..3, out[4..7]=low of bytes 0..3); scale [N, K/32] uint8 E8M0 (2^(code-127)).
__global__ void mxfp4_dequant_bf16_kernel(
        const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scale,
        __nv_bfloat16* __restrict__ out, long N, long K) {
    const long total = N * K;
    const long Kh = K >> 1, Ks = K >> 5;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const long n = idx / K;
        const long k = idx - n * K;
        const int pos = (int)(k & 7);
        const uint8_t byte = packed[n * Kh + (k >> 3) * 4 + (pos & 3)];
        const uint8_t nib = (pos < 4) ? (byte >> 4) : (byte & 0xF);
        float v = kFP4V[nib & 7];
        if (nib & 8) v = -v;
        const unsigned sbits = ((unsigned)scale[n * Ks + (k >> 5)]) << 23;  // 2^(code-127)
        out[idx] = __float2bfloat16(v * __int_as_float(sbits));
    }
}

torch::Tensor mxfp4_dequant_bf16(torch::Tensor packed, torch::Tensor scale) {
    TORCH_CHECK(packed.is_cuda() && packed.dtype() == torch::kUInt8);
    TORCH_CHECK(scale.is_cuda() && scale.dtype() == torch::kUInt8);
    const long N = packed.size(0), K = packed.size(1) * 2;
    auto out = torch::empty({N, K}, torch::device(packed.device()).dtype(torch::kBFloat16));
    const int threads = 256;
    const int blocks = (int)std::min<long>((N * K + threads - 1) / threads, 65535);
    mxfp4_dequant_bf16_kernel<<<blocks, threads>>>(
        packed.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, K);
    return out;
}
"""

_ext = load_inline(
    name='tp_mxfp4_dequant',
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
