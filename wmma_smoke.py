"""De-risk: verify nvcuda::wmma bf16 GEMM compiles+runs via load_inline in the
megamoe container, matching torch. C[M,N] = A[M,K] @ B[K,N]."""
import os
import torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_mm')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0'

CUDA = r"""
#include <torch/extension.h>
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

// One warp computes a 16x16 output tile, accumulating over K (multiple of 16).
// A [M,K] row-major, B [K,N] row-major, C [M,N] f32. Grid: (N/16, M/16), 1 warp/block.
__global__ void wmma_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                          int M, int N, int K) {
    const int tileN = blockIdx.x, tileM = blockIdx.y;
    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16,16,16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (tileM*16)*K + k, K);
        wmma::load_matrix_sync(b, B + k*N + tileN*16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + (tileM*16)*N + tileN*16, acc, N, wmma::mem_row_major);
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0), K=A.size(1), N=B.size(1);
    auto C = torch::empty({M,N}, torch::device(A.device()).dtype(torch::kFloat32));
    dim3 grid(N/16, M/16);
    wmma_gemm<<<grid, 32>>>(reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()), C.data_ptr<float>(), M,N,K);
    return C;
}
"""
ext = load_inline(name='wmma_smoke', cpp_sources="torch::Tensor gemm(torch::Tensor A, torch::Tensor B);",
                  cuda_sources=CUDA, functions=['gemm'], extra_cuda_cflags=['-O3'], verbose=False)

torch.manual_seed(0)
M,N,K = 64, 256, 512
A = torch.randn(M,K, device='cuda', dtype=torch.bfloat16)
B = torch.randn(K,N, device='cuda', dtype=torch.bfloat16)
C = ext.gemm(A.contiguous(), B.contiguous())
ref = (A.float() @ B.float())
cos = torch.nn.functional.cosine_similarity(C.flatten(), ref.flatten(), dim=0).item()
print(f"WMMA gemm cosine={cos:.5f} maxrelerr={((C-ref).abs()/ref.abs().clamp_min(1e-3)).max().item():.3f}")
print("WMMA_OK" if cos > 0.99 else "WMMA_FAIL")
