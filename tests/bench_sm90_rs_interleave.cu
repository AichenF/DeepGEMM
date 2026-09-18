// Does interleaving the register-source decode with the WGMMA beat decoding a
// whole half first? Same loader, same MMA count; only the grouping differs, so
// this isolates scheduling from arithmetic.
//
// The mainloop decodes all four k-step slices of a half and then issues four
// MMAs, which leaves half 0's decode exposed -- only half 1's hides under half
// 0's MMA. Finer groups should hide more of it, at one extra fence/commit pair
// each. Measured H20-3e, 78 SMs, identical to 2 dp over three runs:
//
//   kGroupK=4  decode half then 4 MMA (shipped)   2.35 TB/s
//   kGroupK=2  two groups per half                2.40   +2.1%
//   kGroupK=1  interleaved per k-step             2.13   -9.4%
//
// So the aggressive form is a regression -- the fences cost more than the
// overlap wins -- and the mild one is inside the kernel's own noise floor.
// The mainloop's existing half-granularity grouping is the right trade; do not
// re-litigate this without new evidence.
//
//   nvcc -O3 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
//        -std=c++20 -I deep_gemm/include -I third-party/cutlass/include \
//        -o bench_sm90_rs_interleave tests/bench_sm90_rs_interleave.cu
// (-arch=sm_90a alone resolves to sm_90 and ptxas then rejects wgmma.)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>
#define CHK(x) do{auto e=(x);if(e!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)
using namespace deep_gemm;
constexpr int kRows=128,kTileBytes=kRows*(64+4),kStages=4,kTilesPerStage=2;
constexpr int kStageBytes=kTilesPerStage*kTileBytes;
constexpr int kNSwap=24, kBlockK=128, kASize=kNSwap*kBlockK;
using RSW = typename mma::sm90::FP8RSMMASelector<kNSwap>::type;
constexpr int kAccum = RSW::kNumAccum;

__device__ __forceinline__ void mbi(uint64_t*b,uint32_t c){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void mbe(uint64_t*b,uint32_t n){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(n));}
__device__ __forceinline__ void mba(uint64_t*b){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"l"(__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void mbw(uint64_t*b,uint32_t p){asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@P bra D;\nbra W;\nD:\n}\n"::"l"(__cvta_generic_to_shared(b)),"r"(p));}
__device__ __forceinline__ void b1d(void*d,const void*s,uint32_t n,uint64_t*b){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"l"(__cvta_generic_to_shared(d)),"l"(s),"r"(n),"l"(__cvta_generic_to_shared(b)):"memory");}

// kGroupK = 4 reproduces the old structure (decode whole half, then 4 MMA);
// kGroupK = 1 is the interleaved one under test.
template <int kGroupK>
__global__ __launch_bounds__(128) void run(const uint8_t* __restrict__ raw,int tiles,int kb,float* sink){
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* packed=smem; uint8_t* smem_a=smem+kStages*kStageBytes;
    uint64_t* bars=reinterpret_cast<uint64_t*>(smem_a+kASize);
    auto* lut=reinterpret_cast<mxfp4::ScaledLut*>(bars+2*kStages);
    const uint32_t tid=threadIdx.x, lane=tid&31u, warp=tid>>5;
    if(tid==0) for(int i=0;i<kStages;++i) mbi(bars+i,1);
    mxfp4::init_scaled_lut_window(lut,tid);
    asm volatile("fence.proxy.async.shared::cta;":::"memory"); __syncthreads();
    float acc[kAccum]; for(int i=0;i<kAccum;++i) acc[i]=0.f;
    const int total=tiles*kb; int stage=0; uint32_t ph=0;
    auto issue=[&](int idx,int s){ if(tid) return; uint8_t* d=packed+s*kStageBytes; mbe(bars+s,kStageBytes);
        for(int t=0;t<kTilesPerStage;++t)
            b1d(d+t*kTileBytes, raw+(uint64_t)((blockIdx.x*tiles+idx/kb)*kTilesPerStage+t)*kb*kTileBytes+(uint64_t)(idx%kb)*kTileBytes, kTileBytes, bars+s); };
    for(int s=0;s<kStages&&s<total;++s) issue(s,s);
    const uint32_t row_idx=lane>>2, wg_n=0;
    for(int i=0;i<total;++i){
        mbw(bars+stage,ph);
        const uint8_t* p=packed+stage*kStageBytes;
        #pragma unroll
        for(uint32_t half=0;half<2;++half){
            const uint32_t decode_row = wg_n + half*64u + warp*16u + row_idx + ((lane&1u)<<3);
            const auto src = mxfp4::decode_tile_row(p, decode_row);
            const uint32_t sw = ptx::ld_shared(reinterpret_cast<const uint32_t*>(src.scale));
            const uint32_t word_sel=(lane>>1)&1u; const bool keep_hi=(lane&1u)==0;
            uint32_t a_frag[4][4];
            #pragma unroll
            for(uint32_t g=0; g<4; g+=kGroupK){
                #pragma unroll
                for(uint32_t sl=g; sl<g+kGroupK; ++sl){
                    const auto* ch = src.chunk + sl*mxfp4::kBChunkStride + word_sel*4u;
                    const uint32_t wlo=ptx::ld_shared(reinterpret_cast<const uint32_t*>(ch));
                    const uint32_t whi=ptx::ld_shared(reinterpret_cast<const uint32_t*>(ch+8u));
                    const auto l=mxfp4::load_scaled_lut(lut,(sw>>(sl*8u))&0xffu);
                    mxfp4::dequant_rs_word_pair(wlo,whi,l,keep_hi,a_frag[sl]);
                    #pragma unroll
                    for(int z=0;z<4;++z) mma::sm90::warpgroup_fence_operand(a_frag[sl][z]);
                }
                #pragma unroll
                for(int z=0;z<kAccum;++z) ptx::warpgroup_fence_operand(acc[z]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for(uint32_t k=g;k<g+kGroupK;++k)
                    RSW::wgmma(a_frag[k], mma::sm90::make_smem_desc(smem_a+k*RSW::K,1), acc, k);
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for(int z=0;z<kAccum;++z) ptx::warpgroup_fence_operand(acc[z]);
            }
            ptx::warpgroup_wait<0>();
        }
        __syncthreads();
        const int nxt=i+kStages;
        if(nxt<total) issue(nxt,stage); else if(tid==0) mba(bars+stage);
        if(++stage==kStages){stage=0;ph^=1;}
    }
    if(acc[0]==1234.5f) sink[0]=acc[0];
}
template<int G> void go(const char* n,const uint8_t* buf,int tiles,int kb,float* sink,int sms,size_t bytes){
    size_t smem=kStages*kStageBytes+kASize+2*kStages*8+32*sizeof(mxfp4::ScaledLut);
    CHK(cudaFuncSetAttribute(run<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    for(int r=0;r<3;++r) run<G><<<sms,128,smem>>>(buf,tiles,kb,sink);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    for(int r=0;r<10;++r) run<G><<<sms,128,smem>>>(buf,tiles,kb,sink);
    cudaEventRecord(b);CHK(cudaDeviceSynchronize());float ms;cudaEventElapsedTime(&ms,a,b);
    printf("  %-38s %6.3f ms  %5.2f TB/s\n",n,ms/10,bytes*10/(ms*1e-3)/1e12);
}
int main(){int sms;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    const int kb=48,tiles=8; size_t bytes=(size_t)sms*tiles*kTilesPerStage*kb*kTileBytes;
    uint8_t* buf;CHK(cudaMalloc(&buf,bytes));CHK(cudaMemset(buf,0x11,bytes));
    float* sink;CHK(cudaMalloc(&sink,4));
    printf("RS decode + WGMMA (N_SWAP=%d), %d SMs, %.2f GB/launch\n",kNSwap,sms,bytes/1e9);
    go<4>("kGroupK=4  decode half, then 4 MMA",buf,tiles,kb,sink,sms,bytes);
    go<2>("kGroupK=2  two groups per half",buf,tiles,kb,sink,sms,bytes);
    go<1>("kGroupK=1  interleaved per k-step",buf,tiles,kb,sink,sms,bytes);
    return 0;}
