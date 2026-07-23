#!/usr/bin/env bash
set -euo pipefail

REPO=${DG_AKO_REMOTE_REPO:-/root/fac/megamoe/DeepGEMM_megamoe_nvfp4_dev}
RESULT_ROOT=${DG_AKO_RESULT_ROOT:-/root/fac/megamoe/results/megamoe_nvfp4_dev_ako}
LABEL=${DG_AKO_LABEL:-baseline}
RESULT=${RESULT_ROOT}/${LABEL}
MATRIX_RUNNER=${DG_AKO_MATRIX_RUNNER:-/root/fac/scripts/megamoe/v4_shape_matrix_runner.py}
PY=${DG_AKO_PYTHON:-python}
NVFP4_BLOCK_N=${DG_AKO_NVFP4_BLOCK_N:-}

block_n_args=()
if [[ -n ${NVFP4_BLOCK_N} ]]; then
    block_n_args=(--nvfp4-block-n "${NVFP4_BLOCK_N}")
fi

export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DG_USE_LOCAL_VERSION=0
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
        --num-max-tokens-per-rank "${DG_AKO_MAX_TOKENS_PER_RANK:-128}" \
        "${block_n_args[@]}" \
        --weight-scales 0.05 \
        --global-scale-modes none expert \
        --repeats "${DG_AKO_CORRECTNESS_REPEATS:-1}" \
        --seed 42 2>&1 | tee "${RESULT}/logs/correctness_${shape}.log"
}

if [[ ${DG_AKO_RUN_CORRECTNESS:-1} == 1 ]]; then
    read -r -a correctness_shapes <<< "${DG_AKO_CORRECTNESS_SHAPES:-flash pro mimo_pro}"
    for shape in "${correctness_shapes[@]}"; do
        case "${shape}" in
            flash) run_correctness flash 4096 2048 32 6 ;;
            pro) run_correctness pro 7168 3072 48 6 ;;
            mimo_pro) run_correctness mimo_pro 6144 2048 48 8 ;;
            *) echo "Unknown correctness shape: ${shape}" >&2; exit 2 ;;
        esac
    done
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
                "${block_n_args[@]}" \
                2>&1 | tee "${log}"
        done
    done
fi

echo "RUN_COMPLETE=1"
