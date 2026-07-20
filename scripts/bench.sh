#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LABEL=${1:-baseline}
HOST=${DG_AKO_HOST:-root@10.6.131.8}
CONTAINER=${DG_AKO_CONTAINER:-mega_moe_box}
REMOTE_REPO=${DG_AKO_REMOTE_REPO:-/root/fac/megamoe/DeepGEMM_megamoe_nvfp4_dev}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_REVISION=$(git rev-parse HEAD)

ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" \
    "test -d '${REMOTE_REPO}' || cp -a /root/fac/megamoe/DeepGEMM_megamoe_nvfp4_opt '${REMOTE_REPO}'; \
     if test ! -d '${REMOTE_REPO}/third-party/cutlass/include/cute'; then \
         rm -rf '${REMOTE_REPO}/third-party/cutlass' '${REMOTE_REPO}/third-party/fmt'; \
         cp -a /root/fac/megamoe/DeepGEMM_megamoe_nvfp4_ba7ee09_acc/third-party/cutlass \
             '${REMOTE_REPO}/third-party/cutlass'; \
         cp -a /root/fac/megamoe/DeepGEMM_megamoe_nvfp4_ba7ee09_acc/third-party/fmt \
             '${REMOTE_REPO}/third-party/fmt'; \
     fi"

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
    "docker exec \
        -e DG_AKO_LABEL='${LABEL}' \
        -e DG_AKO_SOURCE_REVISION='${SOURCE_REVISION}' \
        -e DG_AKO_REMOTE_REPO='${REMOTE_REPO}' \
        -e DG_AKO_REBUILD='${DG_AKO_REBUILD:-1}' \
        -e DG_AKO_RUN_CORRECTNESS='${DG_AKO_RUN_CORRECTNESS:-1}' \
        -e DG_AKO_RUN_PERF='${DG_AKO_RUN_PERF:-1}' \
        -e DG_AKO_NUM_TESTS='${DG_AKO_NUM_TESTS:-50}' \
        -e DG_AKO_SHAPES='${DG_AKO_SHAPES:-flash pro mimo_pro}' \
        -e DG_AKO_M_VALUES='${DG_AKO_M_VALUES:-8 16 32 64 128}' \
        '${CONTAINER}' \
        bash -lc 'cd ${REMOTE_REPO} && bash scripts/ako_sm90_nvfp4_unified_smallm_remote.sh'" \
    2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e

TRAJ_DIR=trajectory/${TIMESTAMP}_${LABEL}
mkdir -p "${TRAJ_DIR}"
cp HINTS.md "${TRAJ_DIR}/"
cp deep_gemm/mega/__init__.py "${TRAJ_DIR}/"
cp csrc/apis/mega.hpp "${TRAJ_DIR}/"
cp csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp "${TRAJ_DIR}/"
cp deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_fused_body.inl "${TRAJ_DIR}/"
mv _bench_output.txt "${TRAJ_DIR}/output.txt"
git diff --binary > "${TRAJ_DIR}/candidate.patch"
git status --short > "${TRAJ_DIR}/git-status.txt"
echo "Trajectory saved to: ${TRAJ_DIR}"

exit "${BENCH_EXIT}"
