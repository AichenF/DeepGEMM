// Decode a real transform_mxfp4_weights_for_mega_moe_sm90 output with the
// kernel's own decoder and diff against the host reference.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <deep_gemm/quantization/mxfp4_fused_scale.cuh>

#define CHECK(e) do { cudaError_t r=(e); if(r!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);return 1;} } while(0)

__global__ void decode_rows(const uint8_t* __restrict__ packed,
                            uint8_t* __restrict__ out, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint8_t* row_ptr = packed + row * 80;
    const uint4* src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) quads[i] = src[i];
    const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(row_ptr + 64);
    const uint32_t swz = (row & 7u) << 4;
    uint8_t* dst = out + row * 128;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const auto lut = deep_gemm::mxfp4::make_scaled_lut((scale_word >> (q*8)) & 0xffu);
        const uint4 v = quads[q];
        const uint2 w0 = deep_gemm::mxfp4::dequant_word(v.x, lut);
        const uint2 w1 = deep_gemm::mxfp4::dequant_word(v.y, lut);
        const uint2 w2 = deep_gemm::mxfp4::dequant_word(v.z, lut);
        const uint2 w3 = deep_gemm::mxfp4::dequant_word(v.w, lut);
        *reinterpret_cast<uint4*>(dst + (((q*2)*16) ^ swz))   = make_uint4(w0.x,w0.y,w1.x,w1.y);
        *reinterpret_cast<uint4*>(dst + (((q*2+1)*16) ^ swz)) = make_uint4(w2.x,w2.y,w3.x,w3.y);
    }
}

static std::vector<uint8_t> load(const char* p, size_t n) {
    std::vector<uint8_t> v(n); FILE* f = fopen(p, "rb");
    if (!f || fread(v.data(), 1, n, f) != n) { printf("read fail %s\n", p); exit(2); }
    fclose(f); return v;
}

int main() {
    const int rows = 256, k = 128;
    auto packed = load("chain_packed.bin", (size_t)rows * 80);
    auto ref    = load("chain_ref.bin",    (size_t)rows * k);

    uint8_t *d_p = nullptr, *d_o = nullptr;
    CHECK(cudaMalloc(&d_p, packed.size()));
    CHECK(cudaMalloc(&d_o, ref.size()));
    CHECK(cudaMemcpy(d_p, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    decode_rows<<<(rows+127)/128, 128>>>(d_p, d_o, rows);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(ref.size());
    CHECK(cudaMemcpy(got.data(), d_o, got.size(), cudaMemcpyDeviceToHost));

    long bad = 0, shown = 0;
    for (int r = 0; r < rows; ++r) {
        const uint32_t swz = (r & 7u) << 4;
        for (int i = 0; i < k; ++i) {
            const int phys = ((i / 16) * 16 ^ swz) + i % 16;   // undo store swizzle
            const uint8_t g = got[(size_t)r*k + phys], w = ref[(size_t)r*k + i];
            if (g != w) { if (++bad <= 5 && shown++ < 5)
                printf("  row=%d k=%d got=0x%02x want=0x%02x\n", r, i, g, w); }
        }
    }
    printf("decode chain over real transform output: %ld/%ld mismatches -> %s\n",
           bad, (long)rows*k, bad == 0 ? "PASS" : "FAIL");
    cudaFree(d_p); cudaFree(d_o);
    return bad == 0 ? 0 : 1;
}
