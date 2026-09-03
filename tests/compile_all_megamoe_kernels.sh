#!/usr/bin/env bash
# Compile-gate the MegaMoE kernels. No GPU needed; nvcc alone catches what a
# hand-resolved merge breaks.
#
# Three things this gets right, each of which silently produces a false PASS
# if you get it wrong:
#   1. Templates must be INSTANTIATED. A bare `#include` of a header whose
#      kernel is a template only syntax-checks it and proves almost nothing.
#      Instantiation mirrors the JIT's generated `__instantiate_kernel`.
#   2. Use -gencode arch=compute_NNa,code=sm_NNa, not -arch=sm_NNa. The latter
#      resolves to the non-`a` target and ptxas then rejects wgmma. This is the
#      same trap csrc/jit/compiler.hpp works around.
#   3. Kernel bodies are guarded by __CUDA_ARCH__ ranges, so compiling one for
#      the wrong arch compiles it to nothing and "passes" vacuously. Every
#      instantiated probe therefore asserts the emitted PTX is non-trivial.
#
# Usage: bash tests/compile_all_megamoe_kernels.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d third-party/cutlass/include ]] \
  || { echo "run: git submodule update --init --depth 1 third-party/cutlass" >&2; exit 1; }

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
# -std=c++20: the kernels use floating-point non-type template parameters.
flags=(-std=c++20 --expt-relaxed-constexpr -I deep_gemm/include -I third-party/cutlass/include)
fail=0

# instantiate <label> <compute_arch> <expect> <source>
# expect is either `gated-out` (the __CUDA_ARCH__ guard must erase the body) or
# a PTX instruction mnemonic that MUST appear -- checking for a characteristic
# instruction, rather than a line count, is what distinguishes real codegen from
# a body that compiled to nothing.
instantiate() {
  local label=$1 arch=$2 expect=$3 src=$4 hits
  printf '%-40s %-8s ' "$label" "sm_$arch"
  printf '%s\n' "$src" > "$work/tu.cu"
  if ! nvcc -c "${flags[@]}" -gencode=arch=compute_$arch,code=sm_$arch \
       -o /dev/null "$work/tu.cu" 2>"$work/err"; then
    echo 'FAIL (compile)'; sed -n '1,5p' "$work/err" >&2; fail=1; return
  fi
  nvcc -ptx "${flags[@]}" -gencode=arch=compute_$arch,code=sm_$arch \
       -o "$work/out.ptx" "$work/tu.cu" 2>/dev/null || true
  if [[ "$expect" == gated-out ]]; then
    hits=$(grep -c -E 'wgmma|prmt' "$work/out.ptx" 2>/dev/null || true)
    if (( hits > 0 )); then
      echo "FAIL (expected body gated out, found $hits kernel instructions)"; fail=1
    else echo 'PASS (gated out as expected)'; fi
    return
  fi
  hits=$(grep -c -E "$expect" "$work/out.ptx" 2>/dev/null || true)
  if (( hits == 0 )); then
    echo "FAIL (vacuous: no '$expect' in PTX, body gated out for this arch)"; fail=1
  else echo "PASS ($hits x $expect in PTX)"; fi
}

# must_fail <label> <arch> <error regex> <source> -- asserts a compile failure,
# so an architectural limit is pinned by a test rather than by a comment.
must_fail() {
  local label=$1 arch=$2 pattern=$3 src=$4
  printf '%-40s %-8s ' "$label" "sm_$arch"
  printf '%s\n' "$src" > "$work/tu.cu"
  if nvcc -c "${flags[@]}" -gencode=arch=compute_$arch,code=sm_$arch -o /dev/null "$work/tu.cu" 2>"$work/err"; then
    echo "FAIL (expected a compile error matching /$pattern/, but it built)"; fail=1
  elif grep -qE "$pattern" "$work/err"; then
    echo "PASS (rejected as expected: $(grep -oE "$pattern" "$work/err" | head -1))"
  else
    echo "FAIL (failed for an unexpected reason)"; sed -n '1,5p' "$work/err" >&2; fail=1
  fi
}

syntax_only() {
  local label=$1 arch=$2 src=$3
  printf '%-40s %-8s ' "$label" "sm_$arch"
  printf '%s\n' "$src" > "$work/tu.cu"
  if nvcc -c "${flags[@]}" -gencode=arch=compute_$arch,code=sm_$arch -o /dev/null "$work/tu.cu" 2>"$work/err"
  then echo 'PASS (syntax only)'; else echo FAIL; sed -n '1,5p' "$work/err" >&2; fail=1; fi
}

# Tuning row from csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe.hpp, >256-token
# case: block_m=128 block_n=128 experts_per_wave=48 stages=6.
nvfp4_src='#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_h200_fused.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_mega_moe_h200_fused_impl<
        2048, 48, 128, 128, 8192, 8192, 6, 10.0f, true,
        false, false, true, false>);
    (void)ptr;
}'

instantiate 'sm90_nvfp4_mega_moe_h200'   90a  wgmma     "$nvfp4_src"
# Pins down, rather than assumes, that this kernel is Hopper-only: its body is
# #if'd to 900 <= __CUDA_ARCH__ < 1000, so it cannot serve SM120 prefill.
instantiate 'sm90_nvfp4_mega_moe_h200'   120a gated-out "$nvfp4_src"

# Same tuning row against the MXFP4 port. Without this the gate compiled only
# the NVFP4 kernel and the standalone dequant probe, so the MXFP4 MegaMoE kernel
# itself -- the thing the port adds -- was never codegen'd for any arch.
mxfp4_src='#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_mxfp4_mega_moe_h200_fused_impl<
        2048, 48, 128, 128, 8192, 8192, 6, 10.0f, true,
        false, false, true, false>);
    (void)ptr;
}'

instantiate 'sm90_mxfp4_mega_moe_h200'   90a  wgmma     "$mxfp4_src"
instantiate 'sm90_mxfp4_mega_moe_h200'   120a gated-out "$mxfp4_src"

# Template-only headers, not instantiated here: catches parse/merge damage but
# not per-arch codegen.
syntax_only 'sm90_fp8_mega_moe'          90a  '#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>'
sm100_src='#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        2048, 6144, 2048, 384, 8, 8, 128, 128, 128, 64, 128, 128,
        8192, 8192, 3, 128, 128, 128, 132, 8, 10.0f, true>);
    (void)ptr;
}'
instantiate 'sm100_fp8_fp4_mega_moe'     100a tcgen05   "$sm100_src"
# Records why prefill MegaMoE cannot be enabled on RTX 6000D. sglang picks this
# kernel for any _device_sm >= 100, and its own guard is __CUDA_ARCH__ >= 1000,
# so SM120 (1200) does NOT skip the body -- it compiles tcgen05 and ptxas
# rejects it. The failure therefore lands at serving-time JIT, not at config
# validation. Expect a compile failure here until an SM120 MegaMoE exists.
must_fail 'sm100_fp8_fp4_mega_moe'       120a "tcgen05.*not supported" "$sm100_src"
syntax_only 'scheduler/mega_moe'         90a  '#include <deep_gemm/scheduler/mega_moe.cuh>'

# The dequant bridges are non-template inline functions called from a probe
# kernel, so these ARE real instantiations on both architectures.
dequant_src='#include <deep_gemm/quantization/mxfp4_fused_scale.cuh>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>
__global__ void probe(const uint32_t* in, const uint8_t* sf, uint2* out) {
    const auto lut = deep_gemm::mxfp4::make_scaled_lut(sf[threadIdx.x]);
    out[0] = deep_gemm::mxfp4::dequant_word(in[threadIdx.x], lut);
    out[1] = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair(in[threadIdx.x], sf[0]);
}'

instantiate 'dequant nvfp4+mxfp4'        90a  prmt "$dequant_src"
instantiate 'dequant nvfp4+mxfp4'        120a prmt "$dequant_src"

exit "$fail"
