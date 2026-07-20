#!/usr/bin/env bash
set -euo pipefail

ROOT_BASE=${DG_AKO_ROOT_BASE:-/home/scratch.aichenf_wwfo/greencontext}
REPO=${DG_AKO_REMOTE_REPO:-${ROOT_BASE}/worktrees/megamoe_nvfp4_dev}
VENV=${DG_AKO_VENV:-${ROOT_BASE}/venvs/torchcu132-final}
PY=${VENV}/bin/python
SITE=${VENV}/lib/python3.12/site-packages
LABEL=${DG_AKO_LABEL:-h200}
RESULT_ROOT=${DG_AKO_RESULT_ROOT:-${ROOT_BASE}/results/megamoe_nvfp4_dev_ako}
RESULT=${RESULT_ROOT}/${LABEL}
MATRIX_RUNNER=${DG_AKO_MATRIX_RUNNER:-${ROOT_BASE}/megamoe_nvfp4_ba7ee09_v4_shape_matrix_runner.py}

export CUDA_HOME=${SITE}/nvidia/cu13
export PATH=${CUDA_HOME}/bin:${VENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DG_USE_LOCAL_VERSION=0
export DG_NUM_LOCAL_RANKS=8
export DG_BENCH_FLUSH_L2_BYTES=${DG_AKO_FLUSH_L2_BYTES:-8000000000}
export DG_BENCH_REUSE_OUTPUT=1
export DG_BENCH_PRINT_SAMPLES=1
export DG_BENCH_REUSE_FLUSH_BUFFER=1
export DG_BENCH_DISCARD_FIRST_ACTIVE=0
export DG_BENCH_SEED_OFFSET=${DG_AKO_SEED:-101}
export DG_V4_MATRIX_RUNNER=${MATRIX_RUNNER}

mkdir -p "${RESULT}/jit" "${RESULT}/torch_extensions" "${RESULT}/logs"
export DG_JIT_CACHE_DIR=${RESULT}/jit
export TORCH_EXTENSIONS_DIR=${RESULT}/torch_extensions

exec > >(tee -a "${RESULT}/driver.log") 2>&1

echo "RUN_META label=${LABEL} host=$(hostname) source=${DG_AKO_SOURCE_REVISION:-synced-worktree}"
"${PY}" -c 'import torch; print(f"PYTORCH={torch.__version__} CUDA={torch.version.cuda} GPUS={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader

if [[ ${DG_AKO_REBUILD:-1} == 1 ]]; then
    (
        cd "${REPO}"
        bash develop.sh
    )
fi

run_correctness() {
    local shape=$1 hidden=$2 intermediate_hidden=$3 local_experts=$4 topk=$5
    "${PY}" "${REPO}/tests/test_nvfp4_mega_moe_sm90_correctness.py" \
        --batches ${DG_AKO_CORRECTNESS_M_VALUES:-8 128} \
        --hidden "${hidden}" \
        --intermediate-hidden "${intermediate_hidden}" \
        --num-experts "${local_experts}" \
        --num-topk "${topk}" \
        --num-max-tokens-per-rank 128 \
        --nvfp4-block-n 256 \
        --weight-scales 0.05 \
        --global-scale-modes none expert \
        --seed 42 2>&1 | tee "${RESULT}/logs/correctness_${shape}.log"
}

if [[ ${DG_AKO_RUN_CORRECTNESS:-1} == 1 ]]; then
    run_correctness flash 4096 2048 32 6
    run_correctness pro 7168 3072 48 6
    run_correctness mimo_pro 6144 2048 48 8
    echo "CORRECT=True"
fi

if [[ ${DG_AKO_RUN_PERF:-1} == 1 ]]; then
    read -r -a shapes <<< "${DG_AKO_SHAPES:-flash pro mimo_pro}"
    read -r -a m_values <<< "${DG_AKO_M_VALUES:-8 16 32 64 128}"
    for shape in "${shapes[@]}"; do
        for m in "${m_values[@]}"; do
            log=${RESULT}/logs/${shape}_M${m}.log
            "${PY}" "${REPO}/docs/experiments/sm90_nvfp4_standard_prepack/run_v4_repo.py" \
                --repo "${REPO}" \
                --script "${REPO}/tests/bench_nvfp4_mega_moe_sm90.py" \
                --impl-key ours_nvfp4 \
                --impl ours_nvfp4 \
                --shape "${shape}" \
                --m "${m}" \
                --cap 8192 \
                --num-tests "${DG_AKO_NUM_TESTS:-50}" \
                --stat median \
                --num-processes 8 \
                2>&1 | tee "${log}"
        done
    done
fi

echo "RUN_COMPLETE=1"
