"""Nail the fp8 mma.sync m16n8k32 fragment layout. C[16,N] = A[16,K] @ B[N,K]^T,
A,B fp8 e4m3, C f32. Verify vs torch. One warp, tile over N(8) and K(32)."""
import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

CUDA = r'''
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// A[16,K] row-major, B[N,K] row-major (weights, =B^T for the mma). C[16,N] f32.
// One warp. K multiple of 32, N multiple of 8. A,B loaded to smem first.
extern "C" __global__ void gemm(const uint8_t* Ag, const uint8_t* Bg, float* C, int K, int N){
    const int t = threadIdx.x & 31, gid = t >> 2, tid = t & 3;
    extern __shared__ uint8_t sm[];
    uint8_t* As = sm;              // [16,K]
    uint8_t* Bs = sm + 16*K;       // [N,K]
    for (int i = threadIdx.x; i < 16*K; i += blockDim.x) As[i] = Ag[i];
    for (int i = threadIdx.x; i < N*K; i += blockDim.x) Bs[i] = Bg[i];
    __syncthreads();

    for (int n0 = 0; n0 < N; n0 += 8) {
        float c0=0,c1=0,c2=0,c3=0;
        for (int k0 = 0; k0 < K; k0 += 32) {
            auto ld=[&](const uint8_t* M,int row,int col)->unsigned{
                return *reinterpret_cast<const unsigned*>(M + row*K + col); };
            unsigned a0=ld(As,gid,   k0+tid*4),   a1=ld(As,gid+8, k0+tid*4);
            unsigned a2=ld(As,gid,   k0+tid*4+16), a3=ld(As,gid+8, k0+tid*4+16);
            unsigned b0=ld(Bs,n0+gid,k0+tid*4),   b1=ld(Bs,n0+gid,k0+tid*4+16);
            asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        // C store: c0->[gid][n0+tid*2], c1->[gid][n0+tid*2+1], c2->[gid+8][..], c3->[gid+8][..]
        C[gid*N + n0+tid*2]     = c0;  C[gid*N + n0+tid*2+1]     = c1;
        C[(gid+8)*N + n0+tid*2] = c2;  C[(gid+8)*N + n0+tid*2+1] = c3;
    }
}
torch::Tensor gemm_t(torch::Tensor A, torch::Tensor B){
    int K=A.size(1), N=B.size(0);
    auto C=torch::zeros({16,N},torch::device(A.device()).dtype(torch::kFloat32));
    int sh=16*K+N*K;
    gemm<<<1,32,sh>>>(A.data_ptr<uint8_t>(),B.data_ptr<uint8_t>(),C.data_ptr<float>(),K,N);
    return C;
}
'''
e = load_inline(name='fp8mma_correct', cpp_sources="torch::Tensor gemm_t(torch::Tensor A, torch::Tensor B);",
                cuda_sources=CUDA, functions=['gemm_t'], extra_cuda_cflags=['-O3'], verbose=False)
torch.manual_seed(0)
M,N,K = 16, 64, 128
Af = (torch.randint(0,7,(M,K),device='cuda')-3).to(torch.float8_e4m3fn)
Bf = (torch.randint(0,7,(N,K),device='cuda')-3).to(torch.float8_e4m3fn)
C = e.gemm_t(Af.view(torch.uint8).contiguous(), Bf.view(torch.uint8).contiguous())
ref = Af.float() @ Bf.float().t()   # [16,N]
maxerr = (C-ref).abs().max().item()
cos = torch.nn.functional.cosine_similarity(C.flatten(), ref.flatten(), dim=0).item()
print(f"FP8_MMA cos={cos:.5f} maxerr={maxerr:.3f}")
print("FP8_MMA_LAYOUT_OK" if cos>0.999 and maxerr<0.01 else "FP8_MMA_LAYOUT_WRONG")
