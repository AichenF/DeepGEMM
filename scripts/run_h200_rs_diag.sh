#!/usr/bin/env bash

set -euo pipefail

M=${1:-64}
NUM_TESTS=${2:-3}
RUN_NAME=${3:-smoke_m64}

ROOT=/home/scratch.aichenf_wwfo/greencontext
REPO=${4:-${ROOT}/worktrees/megamoe_nvfp4_rs_diag_h200_20260718}
RESULT=${ROOT}/results/mimo_h200_rs_diag_20260718/${RUN_NAME}
VENV=${ROOT}/venvs/torchcu132-final
SITE=${VENV}/lib/python3.12/site-packages

mkdir -p "${RESULT}" /tmp/aichenf_rs_diag

export HOME=/tmp/aichenf_rs_diag
export CUDA_HOME=${SITE}/nvidia/cu13
export PATH=${CUDA_HOME}/bin:${VENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DG_USE_LOCAL_VERSION=0
export DG_BENCH_FLUSH_L2_BYTES=8000000000
export DG_BENCH_REUSE_OUTPUT=1
export DG_BENCH_PRINT_SAMPLES=1
export DG_BENCH_REUSE_FLUSH_BUFFER=1
export DG_BENCH_DISCARD_FIRST_ACTIVE=0
export DG_BENCH_SEED_OFFSET=0
export DG_SM90_MOE_SET_NUM_SMS=132
export DG_SM90_NVFP4_FORCE_REGISTER_WEIGHT_RS=${DG_SM90_NVFP4_FORCE_REGISTER_WEIGHT_RS:-1}
export DG_V4_MATRIX_RUNNER=${ROOT}/megamoe_nvfp4_ba7ee09_v4_shape_matrix_runner.py
export DG_JIT_CACHE_DIR=${RESULT}/jit
export TORCH_EXTENSIONS_DIR=${RESULT}/torch_extensions

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname) M=${M} NUM_TESTS=${NUM_TESTS} COLD_L2_BYTES=${DG_BENCH_FLUSH_L2_BYTES} NUM_SMS=${DG_SM90_MOE_SET_NUM_SMS} FORCE_RS=${DG_SM90_NVFP4_FORCE_REGISTER_WEIGHT_RS}"

"${VENV}/bin/python" \
    "${REPO}/docs/experiments/sm90_nvfp4_standard_prepack/run_v4_repo.py" \
    --repo "${REPO}" \
    --script "${REPO}/docs/experiments/sm90_nvfp4_standard_prepack/bench_nibble_group_candidate.py" \
    --impl-key ours_nvfp4 \
    --impl ours_nvfp4 \
    --shape mimo_pro \
    --m "${M}" \
    --cap 8192 \
    --num-tests "${NUM_TESTS}" \
    --stat median \
    --num-processes 8 \
    --nvfp4-block-n 256 \
    2>&1 | tee "${RESULT}/run.log"

echo "RUN_COMPLETE=1"
