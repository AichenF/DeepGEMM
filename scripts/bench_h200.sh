#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LABEL=${1:-h200}
HOST=${DG_AKO_H200_HOST:-viking-prod-298}
ROOT_BASE=${DG_AKO_ROOT_BASE:-/home/scratch.aichenf_wwfo/greencontext}
REMOTE_REPO=${DG_AKO_REMOTE_REPO:-${ROOT_BASE}/worktrees/megamoe_nvfp4_dev}
SEED_REPO=${DG_AKO_SEED_REPO:-${ROOT_BASE}/worktrees/megamoe_nvfp4_dev_m}
VENV=${DG_AKO_VENV:-${ROOT_BASE}/venvs/torchcu132-final}
RESULT_ROOT=${DG_AKO_RESULT_ROOT:-${ROOT_BASE}/results/megamoe_nvfp4_dev_ako}
MATRIX_RUNNER=${DG_AKO_MATRIX_RUNNER:-${ROOT_BASE}/megamoe_nvfp4_ba7ee09_v4_shape_matrix_runner.py}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_REVISION=$(git rev-parse HEAD)

ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" \
    "bash -lc 'mkdir -p \"${REMOTE_REPO}\"; \
        if test ! -d \"${REMOTE_REPO}/third-party/cutlass/include/cute\"; then \
            mkdir -p \"${REMOTE_REPO}/third-party\"; \
            cp -a \"${SEED_REPO}/third-party/cutlass\" \"${REMOTE_REPO}/third-party/\"; \
            cp -a \"${SEED_REPO}/third-party/fmt\" \"${REMOTE_REPO}/third-party/\"; \
        fi'"

rsync -az --delete csrc/ "${HOST}:${REMOTE_REPO}/csrc/"
rsync -az --delete --exclude '_C*.so' deep_gemm/ "${HOST}:${REMOTE_REPO}/deep_gemm/"
rsync -az --delete scripts/ "${HOST}:${REMOTE_REPO}/scripts/"
rsync -az --delete tests/ "${HOST}:${REMOTE_REPO}/tests/"
rsync -az --delete docs/experiments/sm90_nvfp4_standard_prepack/ \
    "${HOST}:${REMOTE_REPO}/docs/experiments/sm90_nvfp4_standard_prepack/"
rsync -az HINTS.md ITERATIONS.md develop.sh setup.py \
    "${HOST}:${REMOTE_REPO}/"

set +e
ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" \
    "bash -lc 'cd \"${REMOTE_REPO}\" && \
        DG_AKO_LABEL=\"${LABEL}\" \
        DG_AKO_SOURCE_REVISION=\"${SOURCE_REVISION}\" \
        DG_AKO_ROOT_BASE=\"${ROOT_BASE}\" \
        DG_AKO_REMOTE_REPO=\"${REMOTE_REPO}\" \
        DG_AKO_VENV=\"${VENV}\" \
        DG_AKO_RESULT_ROOT=\"${RESULT_ROOT}\" \
        DG_AKO_MATRIX_RUNNER=\"${MATRIX_RUNNER}\" \
        DG_AKO_REBUILD=\"${DG_AKO_REBUILD:-1}\" \
        DG_AKO_RUN_CORRECTNESS=\"${DG_AKO_RUN_CORRECTNESS:-1}\" \
        DG_AKO_RUN_PERF=\"${DG_AKO_RUN_PERF:-1}\" \
        DG_AKO_NUM_TESTS=\"${DG_AKO_NUM_TESTS:-50}\" \
        DG_AKO_SHAPES=\"${DG_AKO_SHAPES:-flash pro mimo_pro}\" \
        DG_AKO_M_VALUES=\"${DG_AKO_M_VALUES:-8 16 32 64 128}\" \
        DG_AKO_CORRECTNESS_M_VALUES=\"${DG_AKO_CORRECTNESS_M_VALUES:-8 128}\" \
        DG_AKO_CORRECTNESS_SHAPES=\"${DG_AKO_CORRECTNESS_SHAPES:-flash pro mimo_pro}\" \
        DG_AKO_CORRECTNESS_REPEATS=\"${DG_AKO_CORRECTNESS_REPEATS:-1}\" \
        DG_AKO_MAX_TOKENS_PER_RANK=\"${DG_AKO_MAX_TOKENS_PER_RANK:-128}\" \
        DG_AKO_NVFP4_BLOCK_N=\"${DG_AKO_NVFP4_BLOCK_N:-}\" \
        DG_AKO_FLUSH_L2_BYTES=\"${DG_AKO_FLUSH_L2_BYTES:-8000000000}\" \
        DG_AKO_SEED=\"${DG_AKO_SEED:-101}\" \
        bash scripts/ako_sm90_nvfp4_unified_smallm_h200_remote.sh'" \
    2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e

TRAJ_DIR=trajectory/${TIMESTAMP}_${LABEL}
mkdir -p "${TRAJ_DIR}"
copy_trajectory_file() {
    local source=$1
    mkdir -p "${TRAJ_DIR}/$(dirname "${source}")"
    cp "${source}" "${TRAJ_DIR}/${source}"
}
for source in \
    HINTS.md \
    deep_gemm/mega/__init__.py \
    csrc/apis/mega.hpp \
    csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe.hpp \
    csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp \
    csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe_small_m.hpp \
    csrc/jit_kernels/impls/sm90_nvfp4_mega_moe_small_m.hpp \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe.cuh \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_mode2_dequant.cuh \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_small_m.cuh \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_small_m_fused_body.inl \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_split_l1_body.inl \
    deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_split_l2_body.inl \
    deep_gemm/include/deep_gemm/quantization/nvfp4_dequant.cuh
do
    copy_trajectory_file "${source}"
done
mv _bench_output.txt "${TRAJ_DIR}/output.txt"
git diff --binary > "${TRAJ_DIR}/candidate.patch"
git status --short > "${TRAJ_DIR}/git-status.txt"
echo "Trajectory saved to: ${TRAJ_DIR}"

exit "${BENCH_EXIT}"
