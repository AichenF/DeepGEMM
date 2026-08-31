"""TP MXFP4 MoE per-rank partial FFN.

iter-6+: two fused MXFP4 kernels with RUNTIME-configurable output tiling so small
M isn't starved (M=1 was 8 blocks -> latency-bound). Grid = P*nA (FC1) and P*nB
(FC2); (nA, nB, threads) are launch params (no recompile) -> cheap config sweep /
ako4x 8-GPU parallel search. FC1(dequant+GEMV)+SwiGLU -> act[P,Is]; FC2(dequant+
GEMV) -> route-weighted atomic scatter-add to y_partial. In-register MXFP4 dequant.
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

__constant__ float kFP4V[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
__device__ __forceinline__ float dqv(unsigned nib){ float v=kFP4V[nib&7]; return (nib&8)?-v:v; }

// FC1 + SwiGLU. grid = P*nA. Block (p,ta) computes intermediates [ta*Is/nA, ...).
template<int H, int Is>
__global__ void fc1_kernel(const __nv_bfloat16* __restrict__ x,
        const uint8_t* __restrict__ l1p, const uint8_t* __restrict__ l1s,
        const int* __restrict__ tok, const int* __restrict__ exp,
        float* __restrict__ act, int nA) {
    const int b = blockIdx.x, p = b / nA, ta = b % nA;
    const int t = tok[p], e = exp[p];
    const int tid = threadIdx.x, NT = blockDim.x, warp = tid >> 5, lane = tid & 31, NW = NT >> 5;
    const int IsA = Is / nA, i0 = ta * IsA;
    __shared__ __nv_bfloat16 xs[H];
    for (int k = tid; k < H; k += NT) xs[k] = x[(long)t * H + k];
    __syncthreads();
    const int Hc = H >> 3;
    const long rp = (long)e * (2 * Is) * (H >> 1), rs = (long)e * (2 * Is) * (H >> 5);
    for (int ii = warp; ii < IsA; ii += NW) {
        const int i = i0 + ii;
        const uint32_t* gp = reinterpret_cast<const uint32_t*>(l1p + rp + (long)i * (H >> 1));
        const uint8_t*  gs = l1s + rs + (long)i * (H >> 5);
        const uint32_t* upp = reinterpret_cast<const uint32_t*>(l1p + rp + (long)(Is + i) * (H >> 1));
        const uint8_t*  us = l1s + rs + (long)(Is + i) * (H >> 5);
        float ga = 0.f, ua = 0.f;
        for (int c = lane; c < Hc; c += 32) {
            const float sg = __int_as_float(((unsigned)gs[c >> 2]) << 23);
            const float su = __int_as_float(((unsigned)us[c >> 2]) << 23);
            const uint32_t wg = gp[c], wu = upp[c];
            const int base = c << 3;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned bg = (j < 4) ? (wg >> (8*j)) : (wg >> (8*(j-4)));
                const unsigned bu = (j < 4) ? (wu >> (8*j)) : (wu >> (8*(j-4)));
                const float xv = __bfloat162float(xs[base + j]);
                ga += dqv((j<4)?((bg>>4)&0xF):(bg&0xF)) * sg * xv;
                ua += dqv((j<4)?((bu>>4)&0xF):(bu&0xF)) * su * xv;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1){ ga += __shfl_down_sync(-1u,ga,o); ua += __shfl_down_sync(-1u,ua,o); }
        if (lane == 0) act[(long)p * Is + i] = (ga / (1.f + __expf(-ga))) * ua;
    }
}

// FC2. grid = P*nB. Block (p,tb) computes outputs [tb*H/nB, ...).
template<int H, int Is>
__global__ void fc2_kernel(const uint8_t* __restrict__ l2p, const uint8_t* __restrict__ l2s,
        const int* __restrict__ tok, const int* __restrict__ exp, const float* __restrict__ wt,
        const float* __restrict__ act, float* __restrict__ y, int nB) {
    const int b = blockIdx.x, p = b / nB, tb = b % nB;
    const int t = tok[p], e = exp[p];
    const float rw = wt[p];
    const int tid = threadIdx.x, NT = blockDim.x, warp = tid >> 5, lane = tid & 31, NW = NT >> 5;
    const int Ht = H / nB, h0 = tb * Ht;
    __shared__ float as[Is];
    for (int i = tid; i < Is; i += NT) as[i] = act[(long)p * Is + i];
    __syncthreads();
    const int Ic = Is >> 3;
    const long rp = (long)e * H * (Is >> 1), rs = (long)e * H * (Is >> 5);
    for (int hh = warp; hh < Ht; hh += NW) {
        const int h = h0 + hh;
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(l2p + rp + (long)h * (Is >> 1));
        const uint8_t*  ws = l2s + rs + (long)h * (Is >> 5);
        float acc = 0.f;
        for (int c = lane; c < Ic; c += 32) {
            const float s = __int_as_float(((unsigned)ws[c >> 2]) << 23);
            const uint32_t w = wp[c];
            const int base = c << 3;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned bb = (j < 4) ? (w >> (8*j)) : (w >> (8*(j-4)));
                acc += dqv((j<4)?((bb>>4)&0xF):(bb&0xF)) * s * as[base + j];
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(-1u, acc, o);
        if (lane == 0) atomicAdd(&y[(long)t * H + h], rw * acc);
    }
}

torch::Tensor fused_tp_moe(torch::Tensor x, torch::Tensor l1p, torch::Tensor l1s,
                           torch::Tensor l2p, torch::Tensor l2s, torch::Tensor tok,
                           torch::Tensor exp, torch::Tensor wt, int M, int H, int Is,
                           int nA, int nB, int threads) {
    TORCH_CHECK(H == 6144 && Is == 256, "specialized H=6144 Is=256");
    const int P = tok.size(0);
    auto y = torch::zeros({M, H}, torch::device(x.device()).dtype(torch::kFloat32));
    auto act = torch::empty({P, Is}, torch::device(x.device()).dtype(torch::kFloat32));
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    fc1_kernel<6144,256><<<P * nA, threads>>>(xp, l1p.data_ptr<uint8_t>(), l1s.data_ptr<uint8_t>(),
        tok.data_ptr<int>(), exp.data_ptr<int>(), act.data_ptr<float>(), nA);
    fc2_kernel<6144,256><<<P * nB, threads>>>(l2p.data_ptr<uint8_t>(), l2s.data_ptr<uint8_t>(),
        tok.data_ptr<int>(), exp.data_ptr<int>(), wt.data_ptr<float>(),
        act.data_ptr<float>(), y.data_ptr<float>(), nB);
    return y;
}
"""

_ext = load_inline(
    name='tp_mxfp4_2k_v1',
    cpp_sources="torch::Tensor fused_tp_moe(torch::Tensor x, torch::Tensor l1p, torch::Tensor l1s, torch::Tensor l2p, torch::Tensor l2s, torch::Tensor tok, torch::Tensor exp, torch::Tensor wt, int M, int H, int Is, int nA, int nB, int threads);",
    cuda_sources=_CUDA, functions=['fused_tp_moe'],
    extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=False,
)

_NA = int(os.environ.get('TP_NA', '8'))       # FC1 intermediate tiles per pair (divides Is=256)
_NB = int(os.environ.get('TP_NB', '16'))      # FC2 output tiles per pair (divides H=6144)
_THREADS = int(os.environ.get('TP_THREADS', '256'))


def tp_moe_partial(x, l1_packed, l1_scale, l2_packed, l2_scale, topk_idx, topk_w, Is):
    M, H = x.shape
    topk = topk_idx.shape[1]
    tok = torch.arange(M, device=x.device, dtype=torch.int32).repeat_interleave(topk)
    exp = topk_idx.reshape(-1).to(torch.int32)
    wt = topk_w.reshape(-1).to(torch.float32)
    return _ext.fused_tp_moe(x.to(torch.bfloat16).contiguous(),
                             l1_packed.contiguous(), l1_scale.contiguous(),
                             l2_packed.contiguous(), l2_scale.contiguous(),
                             tok, exp, wt, M, H, Is, _NA, _NB, _THREADS)
