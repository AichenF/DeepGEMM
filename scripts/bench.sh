#!/bin/bash
# AKO signal benchmark for the scheme-A TP4 tile-warp-specialized MegaMoE.
# Run only inside the `megamoe` container:
#   bash scripts/bench.sh iter-N
# Optional signal controls: MS, OUTER, REPLAYS, WARMUP_REPLAYS, GPU_LIST.
set -euo pipefail

AKO_LABEL="${1:-run}"
AKO_REPO=/home/xutingz/fac/DeepGEMM_tp
AKO_MS="${MS:-8,16,32,64,128}"
AKO_OUTER="${OUTER:-2}"
AKO_REPLAYS="${REPLAYS:-5}"
AKO_WARMUP_REPLAYS="${WARMUP_REPLAYS:-2}"
AKO_GPU_LIST="${GPU_LIST:-1,2,3,4}"
AKO_NANOSECONDS="$(date +%N)"
AKO_MASTER_PORT=$((20000 + (10#${AKO_NANOSECONDS} / 1000 % 40000)))
AKO_OUTPUT=_bench_output.txt
AKO_TRAJ_DIR="trajectory/${AKO_LABEL}"

cd "${AKO_REPO}"
export CUDA_VISIBLE_DEVICES="${AKO_GPU_LIST}"
export TORCH_CUDA_ARCH_LIST=9.0a
export V4_SINGLE_LAUNCH_TP4=1
export MASTER_PORT="${AKO_MASTER_PORT}"
export PYTHONPATH=/home/xutingz/fac/.tpmoe_tmp:/home/xutingz/fac/sglang_carv2_overlay_2538def:/tmp/tpmoe_pydeps:/home/xutingz/fac/DeepGEMM_tp:/home/xutingz/fac/DeepGEMM_tp/bench:/lustre/raplab/client/xutingz/workspace/dsv4pro_public_sbo_repro_20260826/source/humming-v0.1.12

/home/xutingz/workspace/miniforge3/bin/torchrun \
  --standalone --nproc_per_node=4 \
  bench/v4_flash_tp_single_vs_multi_graph.py \
  --candidate tp-tile-ws \
  --route-pattern random \
  --ms "${AKO_MS}" \
  --outer "${AKO_OUTER}" \
  --replays "${AKO_REPLAYS}" \
  --warmup-replays "${AKO_WARMUP_REPLAYS}" \
  --pair-granularity replay \
  2>&1 | tee "${AKO_OUTPUT}"

/home/xutingz/workspace/miniforge3/bin/python - <<'PY'
import json
from pathlib import Path

lines = Path("_bench_output.txt").read_text().splitlines()
correctness = [
    json.loads(line.split(" ", 1)[1])
    for line in lines
    if line.startswith("SINGLE_MULTI_CORRECTNESS ")
]
summaries = [
    json.loads(line.split(" ", 1)[1])
    for line in lines
    if line.startswith("SINGLE_MULTI_SUMMARY ")
]
if not summaries:
    raise SystemExit("missing SINGLE_MULTI_SUMMARY")
summary = summaries[-1]
strict = bool(correctness) and all(
    record["candidate_final"]["allreduce_ok"] for record in correctness
)
loose = bool(correctness) and all(
    record["candidate_accept"] for record in correctness
)
runtime = float(summary["candidate_geometric_mean_ms"])
reference = float(summary["control_geometric_mean_ms"])
print("COMPILED=True")
print(f"CORRECT={strict} (strict_allreduce; loose_accept={loose})")
print(f"RUNTIME={runtime:.9f} ms (candidate TP4 max-rank cold-L2 geomean)")
print(f"REF_RUNTIME={reference:.9f} ms (paired multi-kernel control)")
print(f"SPEEDUP={reference / runtime:.6f}x")
PY

mkdir -p "${AKO_TRAJ_DIR}"
cp v4_flash_tp_tile_ws_body.inl "${AKO_TRAJ_DIR}/"
cp v4_flash_tp_tile_ws_megamoe.py "${AKO_TRAJ_DIR}/"
cp v4_flash_tp_native_megamoe.py "${AKO_TRAJ_DIR}/"
cp "${AKO_OUTPUT}" "${AKO_TRAJ_DIR}/"
