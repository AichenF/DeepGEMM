import os, torch
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0'
# Try mma.sync fp8 e4m3 m16n8k32 (Hopper). A[16,32] e4m3, B[32,8] e4m3, C[16,8] f32.
CUDA=r'''
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <mma.h>
// minimal: one warp does m16n8k32 fp8 mma, A/B preloaded to smem row-major, verify vs ref.
__global__ void mmafp8(const __nv_fp8_e4m3* A,const __nv_fp8_e4m3* B,float* C,int K){
  // K multiple of 32. A[16,K] row-major, B[K,8] row-major (=B^T stored [8,K]? use k-major).
  // Use ldmatrix-free manual load into fragment regs per PTX m16n8k32.e4m3 layout.
  // For a smoke test just confirm the instruction compiles+runs; correctness checked loosely.
  int lane=threadIdx.x&31;
  float c[4]={0,0,0,0};
  for(int k0=0;k0<K;k0+=32){
    unsigned a[4],b[2];
    // A fragment: each thread holds 16 e4m3 (4 regs x4 bytes). Simplified indexing.
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned v=0; for(int j=0;j<4;j++){ int row=(lane>>2)+ (i&1)*8; int col=k0+(lane&3)*4+ (i>>1)*16 + j; v|=((unsigned)(*(const unsigned char*)&A[row*K+col]))<<(8*j);} a[i]=v;}
    #pragma unroll
    for(int i=0;i<2;i++){ unsigned v=0; for(int j=0;j<4;j++){ int col=(lane>>2); int row=k0+(lane&3)*4+i*16+j; v|=((unsigned)(*(const unsigned char*)&B[row*8+col]))<<(8*j);} b[i]=v;}
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  // store (approx layout) just to force use
  int r=(lane>>2), cc=(lane&3)*2; C[r*8+cc]=c[0]; C[(r+8)*8+cc]=c[2];
}
torch::Tensor mm(torch::Tensor A,torch::Tensor B){int K=A.size(1); auto C=torch::zeros({16,8},torch::device(A.device()).dtype(torch::kFloat32));
  mmafp8<<<1,32>>>(reinterpret_cast<const __nv_fp8_e4m3*>(A.data_ptr()),reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),C.data_ptr<float>(),K); return C;}
'''
try:
  e=load_inline(name='fp8mma',cpp_sources="torch::Tensor mm(torch::Tensor A,torch::Tensor B);",cuda_sources=CUDA,functions=['mm'],extra_cuda_cflags=['-O3'],verbose=False)
  A=torch.randint(0,4,(16,32),device='cuda').to(torch.float8_e4m3fn); B=torch.randint(0,4,(32,8),device='cuda').to(torch.float8_e4m3fn)
  C=e.mm(A.contiguous(),B.contiguous())
  print("FP8_MMA_COMPILES_RUNS finite=",bool(torch.isfinite(C).all()))
except Exception as ex:
  import traceback; tb=traceback.format_exc()
  print("FP8_MMA_FAIL:", str(ex)[:120])
  print([l for l in tb.splitlines() if 'error:' in l][:3])
