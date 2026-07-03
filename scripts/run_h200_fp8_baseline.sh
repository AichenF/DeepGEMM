#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to a result directory on the H200 host}
OURS=${OURS:?set OURS to the split-FP8 worktree}
PR=${PR:?set PR to the pinned PR323 worktree}
VENV=${VENV:?set VENV to the Python environment}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to CUDA 13.2}
RUNNER=${RUNNER:-$OURS/scripts/h200_fp8_matrix_runner.py}
NUM_TESTS=${NUM_TESTS:-10}
SEED=${SEED:-101}
CACHE_TAG=${CACHE_TAG:-baseline}

export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export DG_USE_LOCAL_VERSION=0
export DG_OURS_REPO="$OURS"
export DG_OURS_BENCH="$OURS/tests/bench_mega_moe_sm90.py"
export DG_PR323_REPO="$PR"
export DG_PR323_BUILD="$PR/build"
export DG_PR323_BENCH="$PR/sgl_deep_gemm/tests/test_mega_moe_hopper.py"
export DG_BENCH_FLUSH_L2_BYTES=0
export TMPDIR="$ROOT/tmp"

# An inherited development shell must not accidentally force a candidate.
unset DG_SM90_MOE_EXPERTS_PER_WAVE DG_SM90_MOE_NUM_STAGES
unset DG_SM90_MOE_FORCE_BLOCK_M DG_SM90_MOE_FORCE_EPILOGUE_WG
unset DG_SM90_MOE_SPLIT_MN DG_SM90_MOE_BN256_2WG DG_SM90_MOE_SWAP_AB
unset DG_SM90_MOE_DISPATCH_WARPS DG_SM90_MOE_DIRECT_L2_SCATTER
unset DG_SM90_MOE_L2_NMAJOR DG_SM90_MOE_ONE_WARP_CLEANUP

mkdir -p "$ROOT/logs" "$ROOT/jit/$CACHE_TAG" "$ROOT/torch_extensions" "$TMPDIR"

run_one() {
    local impl=$1 shape=$2 m=$3 leg=$4
    local log="$ROOT/logs/baseline_${shape}_M${m}_S${SEED}_L${leg}_${impl}_N${NUM_TESTS}.log"
    if grep -q '^RUN_EXIT=0$' "$log" 2>/dev/null; then
        printf 'SKIP_COMPLETE %s\n' "$log"
        return
    fi
    export DG_JIT_CACHE_DIR="$ROOT/jit/$CACHE_TAG/$impl"
    export TORCH_EXTENSIONS_DIR="$ROOT/torch_extensions/$impl"
    mkdir -p "$DG_JIT_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
    printf 'RUN_META_JSON {"impl":"%s","shape":"%s","m":%s,"seed":%s,"leg":%s,"num_tests":%s,"cache_tag":"%s"}\n' \
        "$impl" "$shape" "$m" "$SEED" "$leg" "$NUM_TESTS" "$CACHE_TAG" | tee "$log"
    set +e
    timeout 3600 "$VENV/bin/python" "$RUNNER" \
        --impl "$impl" --shape "$shape" --m "$m" --cap 8192 \
        --seed "$SEED" --num-tests "$NUM_TESTS" --stat median \
        --num-processes 8 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'RUN_EXIT=%s\n' "$status" | tee -a "$log"
    return "$status"
}

for shape in flash pro; do
    index=0
    for m in 8 16 32 64 128 256 512 1024 2048 4096 8192; do
        if (( index % 2 == 0 )); then
            run_one ours "$shape" "$m" 1
            run_one pr323 "$shape" "$m" 2
        else
            run_one pr323 "$shape" "$m" 1
            run_one ours "$shape" "$m" 2
        fi
        index=$((index + 1))
    done
done
