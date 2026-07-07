#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to the campaign result directory}
OURS=${OURS:?set OURS to the split-FP8 worktree}
PR=${PR:?set PR to the pinned PR323 worktree}
VENV=${VENV:?set VENV to the Python environment}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to CUDA 13.2}
CANDIDATE=${CANDIDATE:?set a filesystem-safe candidate name}
SHAPE=${SHAPE:?set SHAPE to flash or pro}
M=${M:?set the token-count point}

RUNNER=${RUNNER:-$ROOT/scripts/h200_fp8_matrix_runner.py}
CANDIDATE_RUNNER=${CANDIDATE_RUNNER:-$ROOT/scripts/run_h200_fp8_candidate.sh}
NUM_TESTS=${NUM_TESTS:-20}
SEED=${SEED:-101}
OBSERVATIONS=${OBSERVATIONS:-3}

CASE_ROOT="$ROOT/candidates/$CANDIDATE"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 DG_USE_LOCAL_VERSION=0 DG_BENCH_FLUSH_L2_BYTES=0
export DG_OURS_REPO="$OURS"
export DG_OURS_BENCH="$OURS/tests/bench_mega_moe_sm90.py"
export DG_PR323_REPO="$PR"
export DG_PR323_BUILD="$PR/build"
export DG_PR323_BENCH="$PR/sgl_deep_gemm/tests/test_mega_moe_hopper.py"
export TMPDIR=${DG_CANDIDATE_TMPDIR:-/tmp/aichenf_sm90_fp8_interleaved}
mkdir -p "$CASE_ROOT/logs" "$TMPDIR"

run_ours_through() {
    local observation=$1
    ROOT="$ROOT" OURS="$OURS" VENV="$VENV" CUDA_HOME="$CUDA_HOME" \
        CANDIDATE="$CANDIDATE" SHAPE="$SHAPE" MS="$M" SEED="$SEED" \
        NUM_TESTS="$NUM_TESTS" OBSERVATIONS="$observation" \
        DG_CANDIDATE_TMPDIR="$TMPDIR" \
        bash "$CANDIDATE_RUNNER"
}

run_pr() {
    local observation=$1
    local log="$CASE_ROOT/logs/${CANDIDATE}_${SHAPE}_M${M}_S${SEED}_O${observation}_N${NUM_TESTS}_pr323.log"
    if grep -q '^RUN_EXIT=0$' "$log" 2>/dev/null; then
        printf 'SKIP_COMPLETE %s\n' "$log"
        return
    fi
    printf 'RUN_META_JSON {"impl":"pr323","candidate":"%s","shape":"%s","m":%s,"seed":%s,"observation":%s,"num_tests":%s}\n' \
        "$CANDIDATE" "$SHAPE" "$M" "$SEED" "$observation" "$NUM_TESTS" | tee "$log"
    set +e
    timeout 3600 "$VENV/bin/python" "$RUNNER" \
        --impl pr323 --shape "$SHAPE" --m "$M" --cap 8192 \
        --seed "$SEED" --num-tests "$NUM_TESTS" --stat median \
        --num-processes 8 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'RUN_EXIT=%s\n' "$status" | tee -a "$log"
    return "$status"
}

for observation in $(seq 1 "$OBSERVATIONS"); do
    if (( observation % 2 )); then
        run_ours_through "$observation"
        run_pr "$observation"
    else
        run_pr "$observation"
        run_ours_through "$observation"
    fi
done
