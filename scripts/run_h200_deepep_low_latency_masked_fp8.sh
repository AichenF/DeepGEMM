#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to the campaign result directory}
OURS=${OURS:?set OURS to this DeepGEMM worktree}
VENV=${VENV:?set VENV to the Python environment}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}

CANDIDATE=${CANDIDATE:-deepep_low_latency_masked_fp8_cold_l2}
SHAPES=${SHAPES:-"flash pro mimo_pro"}
MS=${MS:-"8 16 32 64 128"}
SEED=${SEED:-101}
OBSERVATIONS=${OBSERVATIONS:-3}
WARMUPS=${WARMUPS:-5}
SAMPLES=${SAMPLES:-20}
FLUSH_L2_BYTES=${FLUSH_L2_BYTES:-8000000000}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-9000}
DEEPEP_PYTHONPATH=${DEEPEP_PYTHONPATH:-}

CASE_ROOT="$ROOT/candidates/$CANDIDATE"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 DG_USE_LOCAL_VERSION=0 DG_TEST_USE_SOURCE_TREE=1
if [[ -n "$DEEPEP_PYTHONPATH" ]]; then
    export PYTHONPATH="$DEEPEP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
export DG_JIT_CACHE_DIR=${DG_LL_JIT_CACHE_DIR:-$CASE_ROOT/jit}
export EP_JIT_CACHE_DIR=${EP_LL_JIT_CACHE_DIR:-$CASE_ROOT/deep_ep_jit}
export TRITON_CACHE_DIR=${DG_LL_TRITON_CACHE_DIR:-$CASE_ROOT/triton_cache}
export TORCH_EXTENSIONS_DIR="$CASE_ROOT/torch_extensions"
export TMPDIR=${DG_LL_TMPDIR:-/tmp/aichenf_deepep_ll_masked_fp8}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-1024}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
mkdir -p "$CASE_ROOT/logs" "$DG_JIT_CACHE_DIR" "$EP_JIT_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

{
    printf 'candidate=%s\nshapes=%s\nms=%s\nseed=%s\n' \
        "$CANDIDATE" "$SHAPES" "$MS" "$SEED"
    printf 'observations=%s\nwarmups=%s\nsamples=%s\nflush_l2_bytes=%s\n' \
        "$OBSERVATIONS" "$WARMUPS" "$SAMPLES" "$FLUSH_L2_BYTES"
    printf 'deepep_pythonpath=%s\nnvshmem_qp_depth=%s\n' \
        "$DEEPEP_PYTHONPATH" "$NVSHMEM_QP_DEPTH"
    printf 'triton_cache_dir=%s\n' "$TRITON_CACHE_DIR"
    printf 'branch=%s\nhead=%s\n' \
        "$(git -C "$OURS" branch --show-current)" \
        "$(git -C "$OURS" rev-parse HEAD)"
} >"$CASE_ROOT/config.txt"

index=0
for shape in $SHAPES; do
    for m in $MS; do
        log="$CASE_ROOT/logs/${CANDIDATE}_${shape}_M${m}_S${SEED}_O${OBSERVATIONS}_N${SAMPLES}.log"
        if grep -q '^RUN_EXIT=0$' "$log" 2>/dev/null; then
            printf 'SKIP_COMPLETE %s\n' "$log"
            index=$((index + 1))
            continue
        fi
        export MASTER_PORT=$((MASTER_PORT_BASE + index))
        printf 'RUN_META_JSON {"schema_version":1,"impl":"deepep_low_latency_masked_fp8","candidate":"%s","shape":"%s","m":%s,"cap":%s,"seed":%s,"observations":%s,"num_tests":%s,"flush_l2_bytes":%s}\n' \
            "$CANDIDATE" "$shape" "$m" "$m" "$SEED" "$OBSERVATIONS" \
            "$SAMPLES" "$FLUSH_L2_BYTES" | tee "$log"
        set +e
        timeout 3600 "$VENV/bin/python" \
            "$OURS/tests/bench_deepep_low_latency_masked_fp8_sm90.py" \
            --shape "$shape" --m "$m" --cap "$m" --seed "$SEED" \
            --num-processes 8 --observations "$OBSERVATIONS" \
            --warmups "$WARMUPS" --samples "$SAMPLES" \
            --flush-l2-bytes "$FLUSH_L2_BYTES" 2>&1 | tee -a "$log"
        status=${PIPESTATUS[0]}
        set -e
        printf 'RUN_EXIT=%s\n' "$status" | tee -a "$log"
        if (( status != 0 )); then
            exit "$status"
        fi
        index=$((index + 1))
    done
done
