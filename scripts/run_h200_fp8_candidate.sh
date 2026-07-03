#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to the campaign result directory}
OURS=${OURS:?set OURS to the split-FP8 worktree}
VENV=${VENV:?set VENV to the Python environment}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to CUDA 13.2}
CANDIDATE=${CANDIDATE:?set a filesystem-safe candidate name}
SHAPE=${SHAPE:?set SHAPE to flash or pro}
MS=${MS:?set a space-separated list of M points}
RUNNER=${RUNNER:-$ROOT/scripts/h200_fp8_matrix_runner.py}
NUM_TESTS=${NUM_TESTS:-10}
SEED=${SEED:-101}
OBSERVATIONS=${OBSERVATIONS:-1}

CASE_ROOT="$ROOT/candidates/$CANDIDATE"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 DG_USE_LOCAL_VERSION=0 DG_BENCH_FLUSH_L2_BYTES=0
export DG_OURS_REPO="$OURS"
export DG_OURS_BENCH="$OURS/tests/bench_mega_moe_sm90.py"
export TMPDIR="$CASE_ROOT/tmp"
export DG_JIT_CACHE_DIR="$CASE_ROOT/jit"
export TORCH_EXTENSIONS_DIR="$CASE_ROOT/torch_extensions"
mkdir -p "$CASE_ROOT/logs" "$TMPDIR" "$DG_JIT_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

{
    printf 'candidate=%s\nshape=%s\nms=%s\nseed=%s\nnum_tests=%s\n' \
        "$CANDIDATE" "$SHAPE" "$MS" "$SEED" "$NUM_TESTS"
    env | LC_ALL=C sort | grep '^DG_SM90_MOE_' || true
} > "$CASE_ROOT/config.txt"

for observation in $(seq 1 "$OBSERVATIONS"); do
    for m in $MS; do
        log="$CASE_ROOT/logs/${CANDIDATE}_${SHAPE}_M${m}_S${SEED}_O${observation}_N${NUM_TESTS}.log"
        if grep -q '^RUN_EXIT=0$' "$log" 2>/dev/null; then
            printf 'SKIP_COMPLETE %s\n' "$log"
            continue
        fi
        printf 'RUN_META_JSON {"impl":"ours","candidate":"%s","shape":"%s","m":%s,"seed":%s,"observation":%s,"num_tests":%s}\n' \
            "$CANDIDATE" "$SHAPE" "$m" "$SEED" "$observation" "$NUM_TESTS" | tee "$log"
        set +e
        timeout 3600 "$VENV/bin/python" "$RUNNER" \
            --impl ours --shape "$SHAPE" --m "$m" --cap 8192 \
            --seed "$SEED" --num-tests "$NUM_TESTS" --stat median \
            --num-processes 8 2>&1 | tee -a "$log"
        status=${PIPESTATUS[0]}
        set -e
        printf 'RUN_EXIT=%s\n' "$status" | tee -a "$log"
        if (( status != 0 )); then
            exit "$status"
        fi
    done
done
