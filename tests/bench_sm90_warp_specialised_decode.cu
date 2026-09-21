// What the MXFP4 weight pipeline can actually reach, stage by stage.
//
// bench_sm90_decode_consumer.cu prices the decode with thread 0 issuing the next
// copy only after the decode returns -- i.e. serialised -- which understates the
// ceiling. The kernel runs a dedicated loader warp ahead of the decode against
// empty/full barriers. This reproduces that, then adds the WGMMA consumer, so
// the cost of each stage is separable.
//
// Measured H20-3e, 78 SMs:
//   loader alone                          4.16 TB/s
//   + dequant, 8 decode warps             3.09        (6 stages: no change)
//   + dequant, 4 decode warps             2.48
//   + WGMMA consuming the decoded tile    2.31
//
// The kernel achieves 2.10 (shared-memory split) and 2.29 (register-source),
// i.e. 91-99% of that last figure. So the pipeline is essentially at its limit
// and the binding resource is shared-memory bandwidth, not the dequant ALU: the
// SS path moves ~100 KB of smem per stage (TMA write + decode read + decode
// write + WGMMA read) for 17 KB of HBM, a 5.75x amplification. That is also why
// the register-source path, which skips the decoded-tile round trip, sustains a
// higher stream rate even though the split wins end-to-end by overlapping.
//
// The WGMMA here is a stand-in (it feeds the same descriptor to both operands);
// it is there to carry representative shared-memory traffic, not to compute.
//
//   nvcc -O3 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
//        -std=c++20 -I deep_gemm/include -I third-party/cutlass/include \
//        -o bench_sm90_warp_specialised_decode \
//        tests/bench_sm90_warp_specialised_decode.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>
#define CHK(x) do{auto e=(x);if(e!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)
using namespace deep_gemm;
constexpr int kRows=128,kTileBytes=kRows*(64+4),kTilesPerStage=2;
constexpr int kStageBytes=kTilesPerStage*kTileBytes, kDecoded=256*128;
__device__ __forceinline__ void mbi(uint64_t*b,uint32_t c){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void mbe(uint64_t*b,uint32_t n){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(n));}
__device__ __forceinline__ void mba(uint64_t*b){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"l"(__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void mbw(uint64_t*b,uint32_t p){asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@P bra D;\nbra W;\nD:\n}\n"::"l"(__cvta_generic_to_shared(b)),"r"(p));}
__device__ __forceinline__ void b1d(void*d,const void*s,uint32_t n,uint64_t*b){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"l"(__cvta_generic_to_shared(d)),"l"(s),"r"(n),"l"(__cvta_generic_to_shared(b)):"memory");}

using SWG = typename mma::sm90::FP8MMASelector<24>::type;
// kWork 0 = touch only (loader cap), 1 = dequant, 2 = dequant + WGMMA consumer
template <int kStages, int kDecWarps, int kWork>
__global__ __launch_bounds__(kWork==2 ? 512 : 32 + kDecWarps*32) void ws(
        const uint8_t* __restrict__ raw,int tiles,int kb,uint32_t* sink){
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* packed=smem; uint8_t* dec=smem+kStages*kStageBytes;
    uint64_t* full=reinterpret_cast<uint64_t*>(dec+2*kDecoded);
    uint64_t* empty=full+kStages;
    auto* lut=reinterpret_cast<mxfp4::ScaledLut*>(empty+kStages+(kWork==2?4:0));
    const uint32_t tid=threadIdx.x, warp=tid>>5;
    uint64_t* dfull = empty + kStages;            // decoded ring, 2 slots
    uint64_t* dempty = dfull + 2;
    if(tid==0){ for(int i=0;i<kStages;++i){ mbi(full+i,1); mbi(empty+i,kDecWarps); }
                if constexpr (kWork==2) for(int i=0;i<2;++i){ mbi(dfull+i,kDecWarps); mbi(dempty+i,4); } }
    mxfp4::init_scaled_lut(lut, tid, blockDim.x);
    asm volatile("fence.proxy.async.shared::cta;":::"memory"); __syncthreads();
    const int total=tiles*kb;
    const uint32_t kLoadWarps = (kWork==2 ? 4u : 1u);
    if(warp < kLoadWarps){                         // dedicated loader, runs ahead
      if(warp==0){
        int stage=0; uint32_t ph=0;
        for(int i=0;i<total;++i){
            if(i>=kStages) mbw(empty+stage, ph^1u);
            if(threadIdx.x==0){
                uint8_t* d=packed+stage*kStageBytes; mbe(full+stage,kStageBytes);
                for(int t=0;t<kTilesPerStage;++t)
                    b1d(d+t*kTileBytes, raw+(uint64_t)((blockIdx.x*tiles+i/kb)*kTilesPerStage+t)*kb*kTileBytes+(uint64_t)(i%kb)*kTileBytes, kTileBytes, full+stage);
            }
            if(++stage==kStages){stage=0;ph^=1;}
        }
      }
    } else if (warp < kLoadWarps + kDecWarps) {    // decode warps
        const uint32_t dtid=(warp-kLoadWarps)*32+(tid&31u);
        uint32_t acc=0; int stage=0; uint32_t ph=0;
        for(int i=0;i<total;++i){
            mbw(full+stage, ph);
            const uint8_t* p=packed+stage*kStageBytes;
            if constexpr (kWork==0){ acc^=reinterpret_cast<const uint32_t*>(p)[dtid]; }
            else {
                for(uint32_t r=dtid;r<256;r+=kDecWarps*32)
                    mxfp4::dequant_smem_b_from_packed_mode2_nibble<false>(
                        dec+(i&1)*kDecoded, p, r, lut);
            }
            if constexpr (kWork==2) {
                cutlass::arch::fence_view_async_shared();
                if((tid&31u)==0){ mba(dfull+(i&1)); }
            }
            __syncwarp();
            if((tid&31u)==0) mba(empty+stage);
            if constexpr (kWork==2) { if((tid&31u)==0 && i>=2) mbw(dempty+(i&1), ((i>>1)&1u)^1u); }
            if(++stage==kStages){stage=0;ph^=1;}
        }
        if(acc==0xdeadbeefu) sink[0]=acc;
    } else {                                       // math warpgroup: WGMMA consumer
        float ac[SWG::kNumAccum]; for(int z=0;z<SWG::kNumAccum;++z) ac[z]=0.f;
        for(int i=0;i<total;++i){
            mbw(dfull+(i&1), ((i>>1)&1u));
            ptx::warpgroup_arrive();
            #pragma unroll
            for(uint32_t k=0;k<4;++k)
                SWG::wgmma(mma::sm90::make_smem_desc(dec+(i&1)*kDecoded+k*SWG::K,1),
                           mma::sm90::make_smem_desc(dec+(i&1)*kDecoded+k*SWG::K,1), ac, k);
            ptx::warpgroup_commit_batch(); ptx::warpgroup_wait<0>();
            if((tid&31u)==0) mba(dempty+(i&1));
        }
        if(ac[0]==1234.5f) sink[0]=1;
    }
}
template<int S,int W,int K> void go(const char* n,const uint8_t* b,int t,int kb,uint32_t* s,int sms,size_t by){
    size_t sm=S*kStageBytes+2*kDecoded+(2*S+(K==2?4:0))*8+mxfp4::kScaledLutSize*sizeof(mxfp4::ScaledLut);
    CHK(cudaFuncSetAttribute(ws<S,W,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));
    const int thr=(K==2 ? 512 : 32+W*32);
    for(int r=0;r<3;++r) ws<S,W,K><<<sms,thr,sm>>>(b,t,kb,s);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a,e;cudaEventCreate(&a);cudaEventCreate(&e);cudaEventRecord(a);
    for(int r=0;r<10;++r) ws<S,W,K><<<sms,thr,sm>>>(b,t,kb,s);
    cudaEventRecord(e);CHK(cudaDeviceSynchronize());float ms;cudaEventElapsedTime(&ms,a,e);
    printf("  %-44s %6.3f ms  %5.2f TB/s\n",n,ms/10,by*10/(ms*1e-3)/1e12);
}
int main(){int sms;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    const int kb=48,t=8; size_t by=(size_t)sms*t*kTilesPerStage*kb*kTileBytes;
    uint8_t* b;CHK(cudaMalloc(&b,by));CHK(cudaMemset(b,0x11,by));
    uint32_t* s;CHK(cudaMalloc(&s,4));
    printf("warp-specialised: loader warp + decode warps, %d SMs, %.2f GB\n",sms,by/1e9);
    go<4,8,0>("4 stages, 8 dec warps, touch only (loader cap)",b,t,kb,s,sms,by);
    go<4,8,1>("4 stages, 8 dec warps, full dequant",b,t,kb,s,sms,by);
    go<6,8,1>("6 stages, 8 dec warps, full dequant",b,t,kb,s,sms,by);
    go<4,4,1>("4 stages, 4 dec warps, full dequant",b,t,kb,s,sms,by);
    go<4,8,2>("4 stages, 8 dec warps, dequant + WGMMA",b,t,kb,s,sms,by);
    return 0;}
