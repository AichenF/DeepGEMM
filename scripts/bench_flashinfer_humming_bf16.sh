#!/bin/bash
# Reproducible TP4 BF16-serving comparison against FlashInfer CutlassHummingConfig.
set -euo pipefail

BENCH_REPO=/home/xutingz/fac/DeepGEMM_tp
BENCH_ENV=/home/xutingz/fac/flashinfer_humming_torch211_env
BENCH_CACHE=/home/xutingz/fac/flashinfer_humming_cache_torch211
BENCH_MS="${MS:-8,16,32,64,128}"
BENCH_OUTER="${OUTER:-10}"
BENCH_REPLAYS="${REPLAYS:-200}"
BENCH_WARMUP="${WARMUP_REPLAYS:-10}"
BENCH_GPUS="${GPU_LIST:-0,1,2,3}"
BENCH_PAIR="${PAIR_GRANULARITY:-batch}"
BENCH_ROUTE="${ROUTE_PATTERN:-random}"
BENCH_CUSTOM_QUANT="${CUSTOM_INPUT_QUANT:-group128}"
BENCH_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
BENCH_PORT=$((20000 + $(date +%N) / 1000 % 40000))
BENCH_RESULT_DIR="${BENCH_REPO}/results/flashinfer_humming_bf16"
BENCH_LOG="${BENCH_RESULT_DIR}/${BENCH_TAG}.log"

mkdir -p "${BENCH_RESULT_DIR}"
cd "${BENCH_REPO}"
source "${BENCH_ENV}/bin/activate"
export CUDA_VISIBLE_DEVICES="${BENCH_GPUS}"
export TORCH_CUDA_ARCH_LIST=9.0a
export MASTER_PORT="${BENCH_PORT}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE="${BENCH_CACHE}"
export V4_SINGLE_LAUNCH_TP4=0
# The restored source still parses the dormant single-launch template while
# building the selected multi-kernel extension.  Declare its compact W13
# argument record so discarded C++ branches remain well-formed; these two
# flags do not change any kernel launched by the multi-kernel path.
export V4_SINGLE_LAUNCH_W13_PHASE_NOINLINE=1
export V4_SINGLE_LAUNCH_W13_PHASE_COMPACT_ABI=1
export PYTHONPATH=/home/xutingz/fac/.tpmoe_tmp:/home/xutingz/fac/sglang_carv2_overlay_2538def:/tmp/tpmoe_pydeps:${BENCH_REPO}:${BENCH_REPO}/bench:/lustre/raplab/client/xutingz/workspace/dsv4pro_public_sbo_repro_20260826/source/humming-v0.1.12

EXTRA_ARGS=()
if [[ "${NO_AUTOTUNE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-autotune)
fi

python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  bench/v4_flash_tp_flashinfer_humming_bf16_graph.py \
  --ms "${BENCH_MS}" \
  --route-pattern "${BENCH_ROUTE}" \
  --outer "${BENCH_OUTER}" \
  --replays "${BENCH_REPLAYS}" \
  --warmup-replays "${BENCH_WARMUP}" \
  --pair-granularity "${BENCH_PAIR}" \
  --custom-input-quant "${BENCH_CUSTOM_QUANT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${BENCH_LOG}"

echo "RESULT_LOG=${BENCH_LOG}"
