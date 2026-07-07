#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to the campaign result directory}
OURS=${OURS:?set OURS to this DeepGEMM worktree}
VENV=${VENV:?set VENV to the Python environment}
CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}

CANDIDATE=${CANDIDATE:-deepep_grouped_fp8_cold_l2}
SHAPES=${SHAPES:-"flash pro mimo_pro"}
MS=${MS:-"8 16 32 64 128 256 512 1024 2048 4096 8192"}
SEED=${SEED:-101}
OBSERVATIONS=${OBSERVATIONS:-3}
WARMUPS=${WARMUPS:-5}
SAMPLES=${SAMPLES:-20}
FLUSH_L2_BYTES=${FLUSH_L2_BYTES:-8000000000}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-8500}
DEEPEP_PYTHONPATH=${DEEPEP_PYTHONPATH:-}
DEEPEP_BUFFER=${DEEPEP_BUFFER:-elastic}
DEEPEP_DO_CPU_SYNC=${DEEPEP_DO_CPU_SYNC:-1}
BASELINE_CAP=${BASELINE_CAP:-match_m}

CASE_ROOT="$ROOT/candidates/$CANDIDATE"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$VENV/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 DG_USE_LOCAL_VERSION=0 DG_TEST_USE_SOURCE_TREE=1
if [[ -n "$DEEPEP_PYTHONPATH" ]]; then
    export PYTHONPATH="$DEEPEP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
export DG_JIT_CACHE_DIR=${DG_BASELINE_JIT_CACHE_DIR:-$CASE_ROOT/jit}
export EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-$CASE_ROOT/deep_ep_jit}
export TORCH_EXTENSIONS_DIR="$CASE_ROOT/torch_extensions"
export TMPDIR=${DG_BASELINE_TMPDIR:-/tmp/aichenf_deepep_fp8_base}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export EP_DISABLE_GIN=${EP_DISABLE_GIN:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
mkdir -p "$CASE_ROOT/logs" "$DG_JIT_CACHE_DIR" "$EP_JIT_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

{
    printf 'candidate=%s\nshapes=%s\nms=%s\nseed=%s\n' \
        "$CANDIDATE" "$SHAPES" "$MS" "$SEED"
    printf 'observations=%s\nwarmups=%s\nsamples=%s\nflush_l2_bytes=%s\n' \
        "$OBSERVATIONS" "$WARMUPS" "$SAMPLES" "$FLUSH_L2_BYTES"
    printf 'deepep_pythonpath=%s\ndeepep_buffer=%s\ndeepep_do_cpu_sync=%s\n' \
        "$DEEPEP_PYTHONPATH" "$DEEPEP_BUFFER" "$DEEPEP_DO_CPU_SYNC"
    printf 'baseline_cap=%s\nep_disable_gin=%s\nnccl_nvls_enable=%s\n' \
        "$BASELINE_CAP" "$EP_DISABLE_GIN" "$NCCL_NVLS_ENABLE"
    printf 'deep_ep_jit_cache_dir=%s\n' "$EP_JIT_CACHE_DIR"
    printf 'branch=%s\nhead=%s\n' \
        "$(git -C "$OURS" branch --show-current)" \
        "$(git -C "$OURS" rev-parse HEAD)"
} >"$CASE_ROOT/config.txt"

index=0
for shape in $SHAPES; do
    for m in $MS; do
        if [[ "$BASELINE_CAP" == "match_m" ]]; then
            cap=$m
        else
            cap=$BASELINE_CAP
        fi
        log="$CASE_ROOT/logs/${CANDIDATE}_${shape}_M${m}_S${SEED}_O${OBSERVATIONS}_N${SAMPLES}.log"
        if grep -q '^RUN_EXIT=0$' "$log" 2>/dev/null; then
            printf 'SKIP_COMPLETE %s\n' "$log"
            index=$((index + 1))
            continue
        fi
        export MASTER_PORT=$((MASTER_PORT_BASE + index))
        printf 'RUN_META_JSON {"schema_version":2,"impl":"deepep_grouped_fp8","candidate":"%s","shape":"%s","m":%s,"cap":%s,"seed":%s,"observations":%s,"num_tests":%s,"flush_l2_bytes":%s,"do_cpu_sync":%s}\n' \
            "$CANDIDATE" "$shape" "$m" "$cap" "$SEED" "$OBSERVATIONS" "$SAMPLES" "$FLUSH_L2_BYTES" "$DEEPEP_DO_CPU_SYNC" | tee "$log"
        set +e
        timeout 3600 "$VENV/bin/python" "$OURS/tests/bench_deepep_grouped_fp8_sm90.py" \
            --shape "$shape" --m "$m" --cap "$cap" --seed "$SEED" \
            --num-processes 8 --observations "$OBSERVATIONS" \
            --warmups "$WARMUPS" --samples "$SAMPLES" \
            --flush-l2-bytes "$FLUSH_L2_BYTES" \
            --deepep-buffer "$DEEPEP_BUFFER" \
            --do-cpu-sync "$DEEPEP_DO_CPU_SYNC" 2>&1 | tee -a "$log"
        status=${PIPESTATUS[0]}
        set -e
        printf 'RUN_EXIT=%s\n' "$status" | tee -a "$log"
        if (( status != 0 )); then
            exit "$status"
        fi
        index=$((index + 1))
    done
done
