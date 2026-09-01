"""TP MXFP4 MoE — per-rank partial FFN over the intermediate shard.

iter-4: single FUSED CUDA kernel — one block per (token, expert) pair does
FC1 (MXFP4 W1 @ x, dequant in-register) -> SwiGLU -> FC2 (MXFP4 W2 @ act) ->
atomic scatter-add (route-weighted) into y_partial. No bf16 weight
materialization: reads only the packed MXFP4 weights.
"""
import importlib.util
import os

import torch
from torch.utils.cpp_extension import load_inline

os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_tp')

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__constant__ float kFP4V[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
__device__ __forceinline__ float dqv(unsigned nib){ float v=kFP4V[nib&7]; return (nib&8)?-v:v; }

// One block per (token,expert) pair. Templated on H, Is (compile-time).
template<int H, int Is>
__global__ void fused_tp_moe_kernel(
        const __nv_bfloat16* __restrict__ x,          // [M, H]
        const uint8_t* __restrict__ l1p, const uint8_t* __restrict__ l1s,  // [E,2Is,H/2],[E,2Is,H/32]
        const uint8_t* __restrict__ l2p, const uint8_t* __restrict__ l2s,  // [E,H,Is/2],[E,H,Is/32]
        const int* __restrict__ tok, const int* __restrict__ exp,
        const float* __restrict__ wt, float* __restrict__ y, int H_) {
    const int p = blockIdx.x;
    const int t = tok[p], e = exp[p];
    const float rw = wt[p];
    const int tid = threadIdx.x, NT = blockDim.x;

    __shared__ __nv_bfloat16 xs[H];
    __shared__ float acts[Is];
    for (int k = tid; k < H; k += NT) xs[k] = x[(long)t * H + k];
    __syncthreads();

    // ---- FC1 + SwiGLU: compute act[Is]. Each thread owns a set of intermediate i. ----
    const int Hc = H >> 3;           // chunks per row (K=H)
    const long l1_row_p = (long)e * (2 * Is) * (H >> 1);
    const long l1_row_s = (long)e * (2 * Is) * (H >> 5);
    for (int i = tid; i < Is; i += NT) {
        // gate row = i, up row = Is + i
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(l1p + l1_row_p + (long)i * (H >> 1));
        const uint8_t*  gs = l1s + l1_row_s + (long)i * (H >> 5);
        const uint32_t* up = reinterpret_cast<const uint32_t*>(l1p + l1_row_p + (long)(Is + i) * (H >> 1));
        const uint8_t*  us = l1s + l1_row_s + (long)(Is + i) * (H >> 5);
        float ga = 0.f, ua = 0.f;
        #pragma unroll 4
        for (int c = 0; c < Hc; ++c) {
            const float sg = __int_as_float(((unsigned)gs[c >> 2]) << 23);
            const float su = __int_as_float(((unsigned)us[c >> 2]) << 23);
            const uint32_t wg = gp[c], wu = up[c];
            const int base = c << 3;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned bg = (j < 4) ? (wg >> (8 * j)) : (wg >> (8 * (j - 4)));
                const unsigned bu = (j < 4) ? (wu >> (8 * j)) : (wu >> (8 * (j - 4)));
                const unsigned ng = (j < 4) ? ((bg >> 4) & 0xF) : (bg & 0xF);
                const unsigned nu = (j < 4) ? ((bu >> 4) & 0xF) : (bu & 0xF);
                const float xv = __bfloat162float(xs[base + j]);
                ga += dqv(ng) * sg * xv;
                ua += dqv(nu) * su * xv;
            }
        }
        acts[i] = (ga / (1.f + __expf(-ga))) * ua;   // SwiGLU
    }
    __syncthreads();

    // ---- FC2: y[t, h] += rw * sum_i dequant(W2[e,h,i]) * act[i] ----
    const int Ic = Is >> 3;
    const long l2_row_p = (long)e * H * (Is >> 1);
    const long l2_row_s = (long)e * H * (Is >> 5);
    for (int h = tid; h < H; h += NT) {
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(l2p + l2_row_p + (long)h * (Is >> 1));
        const uint8_t*  ws = l2s + l2_row_s + (long)h * (Is >> 5);
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < Ic; ++c) {
            const float s = __int_as_float(((unsigned)ws[c >> 2]) << 23);
            const uint32_t w = wp[c];
            const int base = c << 3;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned b = (j < 4) ? (w >> (8 * j)) : (w >> (8 * (j - 4)));
                const unsigned nb = (j < 4) ? ((b >> 4) & 0xF) : (b & 0xF);
                acc += dqv(nb) * s * acts[base + j];
            }
        }
        atomicAdd(&y[(long)t * H + h], rw * acc);
    }
}

torch::Tensor fused_tp_moe(torch::Tensor x, torch::Tensor l1p, torch::Tensor l1s,
                           torch::Tensor l2p, torch::Tensor l2s, torch::Tensor tok,
                           torch::Tensor exp, torch::Tensor wt, int M, int H, int Is) {
    auto y = torch::zeros({M, H}, torch::device(x.device()).dtype(torch::kFloat32));
    const int P = tok.size(0);
    const int threads = 256;
    TORCH_CHECK(H == 6144 && Is == 256, "kernel specialized for H=6144, Is=256");
    fused_tp_moe_kernel<6144, 256><<<P, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        l1p.data_ptr<uint8_t>(), l1s.data_ptr<uint8_t>(),
        l2p.data_ptr<uint8_t>(), l2s.data_ptr<uint8_t>(),
        tok.data_ptr<int>(), exp.data_ptr<int>(), wt.data_ptr<float>(),
        y.data_ptr<float>(), H);
    return y;
}
"""

_ext = load_inline(
    name='tp_mxfp4_fused_v1',
    cpp_sources="torch::Tensor fused_tp_moe(torch::Tensor x, torch::Tensor l1p, torch::Tensor l1s, torch::Tensor l2p, torch::Tensor l2s, torch::Tensor tok, torch::Tensor exp, torch::Tensor wt, int M, int H, int Is);",
    cuda_sources=_CUDA,
    functions=['fused_tp_moe'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False,
)


def tp_moe_partial(x, l1_packed, l1_scale, l2_packed, l2_scale, topk_idx, topk_w, Is):
    M, H = x.shape
    topk = topk_idx.shape[1]
    tok = torch.arange(M, device=x.device, dtype=torch.int32).repeat_interleave(topk)
    exp = topk_idx.reshape(-1).to(torch.int32)
    wt = topk_w.reshape(-1).to(torch.float32)
    return _ext.fused_tp_moe(x.to(torch.bfloat16).contiguous(),
                             l1_packed.contiguous(), l1_scale.contiguous(),
                             l2_packed.contiguous(), l2_scale.contiguous(),
                             tok, exp, wt, M, H, Is)
