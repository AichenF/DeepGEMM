#!/usr/bin/env bash
# Build the SM120 MegaMoE correctness and performance executables for one
# shape configuration.
#
#   build.sh <output-dir> [-DCAKE_MOE_... ...]
#
# Runs nvcc inside the pinned CUDA 13.3 container so the toolchain matches the
# qualified artifacts. Records ptxas resource usage and SASS for every build so
# a configuration change can be audited without rerunning the GPU.
set -Eeuo pipefail

readonly source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly image="${CAKE_BUILD_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
readonly deps_root="${CAKE_DEPS_ROOT:-/root/dsv4/deps}"

if [[ $# -lt 1 ]]; then
  printf 'usage: %s <output-dir> [nvcc -D flags...]\n' "$0" >&2
  exit 2
fi

output_dir="$1"
shift
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

config_flags=("$@")
printf '%s\n' "${config_flags[@]:-}" > "${output_dir}/config-flags.txt"

docker run --rm --network none \
  -v "${source_dir}:/src:ro" \
  -v "${output_dir}:/out" \
  -v "${deps_root}:/deps:ro" \
  -w /src "${image}" bash -lc '
set -Eeuo pipefail
read -r -a config <<< "'"${config_flags[*]:-}"'"
common=(
  -std=c++20 -O3 --expt-relaxed-constexpr
  --relocatable-device-code=false
  --generate-code=arch=compute_120a,code=sm_120a
  -lineinfo -Xptxas=-v,-warn-spills
  -I/src
  -I/deps/deep_gemm/include
  -I/deps/third-party/cutlass/include
  -I/deps/third-party/cutlass/tools/util/include
)
libs=(
  -L/usr/local/cuda/lib64/stubs
  -L/usr/lib/x86_64-linux-gnu
  -lnccl -lcuda -lcudart -lcudadevrt -Xcompiler=-pthread
)
nvcc=/usr/local/cuda/bin/nvcc
cuobjdump=/usr/local/cuda/bin/cuobjdump

"${nvcc}" --version > /out/nvcc-version.txt
for target in correctness perf; do
  case "${target}" in
    correctness) entry=deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu ;;
    perf)        entry=deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_perf_host.cu ;;
  esac
  "${nvcc}" "${common[@]}" "${config[@]}" "/src/${entry}" "${libs[@]}" \
    -o "/out/${target}" \
    > "/out/${target}.stdout" 2> "/out/${target}.ptxas.log"
  "${cuobjdump}" --dump-resource-usage "/out/${target}" > "/out/${target}-resource.txt"
  "${cuobjdump}" --dump-sass "/out/${target}" > "/out/${target}-sass.txt"
done

cd /out
sha256sum correctness perf nvcc-version.txt \
  correctness.ptxas.log correctness.stdout correctness-resource.txt correctness-sass.txt \
  perf.ptxas.log perf.stdout perf-resource.txt perf-sass.txt > sha256.txt
'

printf 'built %s\n' "${output_dir}"
