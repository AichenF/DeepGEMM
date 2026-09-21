// Which consumer stage throttles the MXFP4 weight stream, and what is the
// ceiling for W4A8 on a part with no FP4 MMA.
//
// bench_sm90_weight_load.cu shows the shipped tile layout streams at 4.5 TB/s
// with a trivial consumer, but the kernel only achieves 2.1-2.3. This runs the
// same loader and bolts on progressively more of the real consumer, so the
// deficit can be attributed rather than guessed at.
//
// Measured H20-3e, 78 SMs:
//   loader alone                       4.18 TB/s
//   + dequant ALU only (6 ops/word)    2.91        -30%
//   + dequant ALU at 4 ops/word        3.81        (WRONG results; slope only)
//   + shared store only                3.19        -24%
//   + both (the SS decode)             2.39        -43%   (kernel: 2.10-2.29)
//
// So the dequant ALU alone caps any such kernel near 70% of its own load path,
// and that is the ceiling to quote -- not the HBM roofline.
//
// The 4-op row exists to price instruction count: the decode is strongly
// instruction-bound, roughly 10% of stream throughput per op removed. Real
// dequant_word emits ~7-8 SASS ops per 8 output bytes (AND, two PRMT, SHF,
// IMAD.SHL, two LOP3), and each is load-bearing -- the AND is forced because a
// PRMT selector with bit 3 set selects sign-replication mode, and the shifts
// cannot fold into LOP3, which has no shifter. One op saved would be worth ~5%
// on the ceiling and ~3% in-kernel, under the noise floor. The quad-ILP decode
// variant changes nothing, so that knob is exhausted too.
//
//   nvcc -O3 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
//        -std=c++20 -I deep_gemm/include -I third-party/cutlass/include \
//        -o bench_sm90_decode_consumer tests/bench_sm90_decode_consumer.cu
// (-arch=sm_90a alone resolves to sm_90 and ptxas then rejects wgmma.)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>

#define CHK(x) do { auto e=(x); if(e!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

constexpr int kRows = 128, kTileBytes = kRows*(64+4), kStages = 4, kThreads = 256;
constexpr int kTilesPerStage = 2;                    // BLOCK_N 256 = two 128-row tiles
constexpr int kStageBytes = kTilesPerStage*kTileBytes;
constexpr int kDecodedBytes = 256*128;               // BLOCK_N x BLOCK_K FP8

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t n){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"l"(__cvta_generic_to_shared(b)),"r"(n));}
__device__ __forceinline__ void mbar_arrive(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"l"(__cvta_generic_to_shared(b)));}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t p){asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@P bra D;\nbra W;\nD:\n}\n"::"l"(__cvta_generic_to_shared(b)),"r"(p));}
__device__ __forceinline__ void bulk1d(void* d,const void* s,uint32_t n,uint64_t* b){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"l"(__cvta_generic_to_shared(d)),"l"(s),"r"(n),"l"(__cvta_generic_to_shared(b)):"memory");}

// kMode 0 = touch one word (loader alone), 1 = decode to registers (RS-like),
// 2 = decode into a shared decoded tile (SS-like)
template <int kMode>
__global__ __launch_bounds__(kThreads) void consume(
        const uint8_t* __restrict__ raw, int tiles_per_cta, int k_blocks, uint32_t* sink) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* packed = smem;
    uint8_t* decoded = smem + kStages*kStageBytes;            // only used by mode 2
    uint64_t* bars = reinterpret_cast<uint64_t*>(
        smem + kStages*kStageBytes + ((kMode==2||kMode==3||kMode==4) ? 2*kDecodedBytes : 0));
    auto* lut = reinterpret_cast<deep_gemm::mxfp4::ScaledLut*>(bars + 2*kStages);
    const uint32_t tid = threadIdx.x;
    if (tid == 0) for (int i=0;i<kStages;++i) mbar_init(bars+i,1);
    deep_gemm::mxfp4::init_scaled_lut(lut, tid, blockDim.x);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    uint32_t acc = 0;
    const int total = tiles_per_cta * k_blocks;
    int stage = 0; uint32_t ph = 0;
    auto issue = [&](int idx, int s) {
        if (tid) return;
        uint8_t* dst = packed + s*kStageBytes;
        mbar_expect(bars+s, kStageBytes);
        for (int t = 0; t < kTilesPerStage; ++t)
            bulk1d(dst + t*kTileBytes,
                   raw + (uint64_t)((blockIdx.x*tiles_per_cta + idx/k_blocks)*kTilesPerStage + t)
                         * k_blocks * kTileBytes + (uint64_t)(idx%k_blocks)*kTileBytes,
                   kTileBytes, bars+s);
    };
    for (int s = 0; s < kStages && s < total; ++s) issue(s, s);
    for (int i = 0; i < total; ++i) {
        mbar_wait(bars+stage, ph);
        const uint8_t* p = packed + stage*kStageBytes;
        if constexpr (kMode == 0) {
            acc ^= reinterpret_cast<const uint32_t*>(p)[tid];
        } else if constexpr (kMode == 1) {
            // decode ALU only: same loads and same dequant_word work, no smem store
            const auto src = deep_gemm::mxfp4::decode_tile_row(p, tid);
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(src.scale);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 v = *reinterpret_cast<const uint4*>(src.chunk + q*deep_gemm::mxfp4::kBChunkStride);
                const auto l = deep_gemm::mxfp4::load_scaled_lut(lut, (sw >> (q*8)) & 0xffu);
                uint2 d;
                d = deep_gemm::mxfp4::dequant_word(v.x, l); acc ^= d.x ^ d.y;
                d = deep_gemm::mxfp4::dequant_word(v.y, l); acc ^= d.x ^ d.y;
                d = deep_gemm::mxfp4::dequant_word(v.z, l); acc ^= d.x ^ d.y;
                d = deep_gemm::mxfp4::dequant_word(v.w, l); acc ^= d.x ^ d.y;
            }
        } else if constexpr (kMode == 5) {
            const auto src = deep_gemm::mxfp4::decode_tile_row(p, tid);
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(src.scale);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 v = *reinterpret_cast<const uint4*>(src.chunk + q*deep_gemm::mxfp4::kBChunkStride);
                const auto l = deep_gemm::mxfp4::load_scaled_lut(lut, (sw >> (q*8)) & 0xffu);
                uint32_t a, b;
                #define FAKE4(w) { a = deep_gemm::mxfp4::byte_perm_unchecked(l.x, l.y, (w)); \
                                   b = deep_gemm::mxfp4::byte_perm_unchecked(l.x, l.y, (w)); \
                                   asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(a) : "r"(w)); \
                                   asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(b) : "r"(w)); \
                                   acc ^= a ^ b; }
                FAKE4(v.x) FAKE4(v.y) FAKE4(v.z) FAKE4(v.w)
                #undef FAKE4
            }
        } else if constexpr (kMode == 3) {
            // smem store only: same 128 B written per thread, no dequant ALU
            const auto src = deep_gemm::mxfp4::decode_tile_row(p, tid);
            uint8_t* dst = decoded + (i&1)*kDecodedBytes + tid*128;
            const uint32_t sz = (tid & 7u) << 4;       // same XOR swizzle the decoder uses
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 v = *reinterpret_cast<const uint4*>(src.chunk + q*deep_gemm::mxfp4::kBChunkStride);
                *reinterpret_cast<uint4*>(dst + (((q*2  )*16) ^ sz)) = v;
                *reinterpret_cast<uint4*>(dst + (((q*2+1)*16) ^ sz)) = v;
            }
            acc ^= decoded[(i&1)*kDecodedBytes + tid];
        } else if constexpr (kMode == 2) {
            deep_gemm::mxfp4::dequant_smem_b_from_packed_mode2_nibble<false>(
                decoded + (i&1)*kDecodedBytes, p, tid, lut);
            acc ^= decoded[(i&1)*kDecodedBytes + tid];
        } else {
            deep_gemm::mxfp4::dequant_smem_b_from_packed_mode2_nibble<true>(
                decoded + (i&1)*kDecodedBytes, p, tid, lut);
            acc ^= decoded[(i&1)*kDecodedBytes + tid];
        }
        __syncthreads();
        const int nxt = i + kStages;
        if (nxt < total) issue(nxt, stage); else if (tid==0) mbar_arrive(bars+stage);
        if (++stage == kStages) { stage = 0; ph ^= 1; }
    }
    if (acc == 0xdeadbeefu) sink[0] = acc;
}

template <int M> void run(const char* name, const uint8_t* buf, int tiles, int kb, uint32_t* sink, int sms, size_t bytes) {
    size_t smem = kStages*kStageBytes + ((M==2||M==3||M==4) ? 2*kDecodedBytes : 0) + 2*kStages*8 + deep_gemm::mxfp4::kScaledLutSize*sizeof(deep_gemm::mxfp4::ScaledLut);
    CHK(cudaFuncSetAttribute(consume<M>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    for (int r=0;r<3;++r) consume<M><<<sms,kThreads,smem>>>(buf,tiles,kb,sink);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r=0;r<10;++r) consume<M><<<sms,kThreads,smem>>>(buf,tiles,kb,sink);
    cudaEventRecord(b); CHK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("  %-34s %6.2f ms  %5.2f TB/s  (smem %zu KB)\n", name, ms/10, bytes*10/(ms*1e-3)/1e12, smem/1024);
}

int main() {
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int kb = 48, tiles = 8;                       // 8 BLOCK_N-tiles x 48 k-blocks per CTA
    size_t bytes = (size_t)sms*tiles*kTilesPerStage*kb*kTileBytes;
    uint8_t* buf; CHK(cudaMalloc(&buf, bytes)); CHK(cudaMemset(buf, 0x11, bytes));
    uint32_t* sink; CHK(cudaMalloc(&sink, 4));
    printf("H20-3e %d SMs, streaming %.2f GB per launch\n", sms, bytes/1e9);
    run<0>("loader alone (1 word/thread)", buf, tiles, kb, sink, sms, bytes);
    run<1>("decode ALU only (6 ops/word)",    buf, tiles, kb, sink, sms, bytes);
    run<5>("decode ALU, 4 ops/word (WRONG)",  buf, tiles, kb, sink, sms, bytes);
    run<3>("smem store only, no decode ALU",  buf, tiles, kb, sink, sms, bytes);
    run<2>("+ decode to shared (SS, both)",   buf, tiles, kb, sink, sms, bytes);
    run<4>("SS with quad-ILP decode",         buf, tiles, kb, sink, sms, bytes);
    return 0;
}
