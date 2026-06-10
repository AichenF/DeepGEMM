#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
if [[ -n "${apps}" ]]; then
  echo "Refusing to run: GPU compute processes are present." >&2
  echo "${apps}" >&2
  exit 1
fi

echo "GPU preflight passed: no compute processes."

log_dir="${NVFP4_GATE_LOG_DIR:-/tmp/deepgemm_nvfp4_ako_gate_logs}"
mkdir -p "${log_dir}"
nvfp4_log="${log_dir}/bench_nvfp4.log"
w8a8_log="${log_dir}/bench_w8a8.log"
bench_batches=(${NVFP4_GATE_BENCH_BATCHES:-256 512 1024 2048 4096 8192})
bench_num_tests="${NVFP4_GATE_NUM_TESTS:-5}"

MASTER_PORT=29719 \
python3 tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --num-experts 4 \
  --num-topk 2 \
  --hidden 512 \
  --intermediate-hidden 256 \
  --batches 8 32 \
  --weight-scales 0.001 0.05 0.3

MASTER_PORT=29720 \
DG_JIT_CACHE_DIR=/tmp/deepgemm_nvfp4_ako_postpatch_correct \
python3 tests/test_nvfp4_mega_moe_sm90_correctness.py

MASTER_PORT=29721 \
DG_JIT_CACHE_DIR=/tmp/deepgemm_nvfp4_ako_postpatch_nvfp4 \
python3 tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches "${bench_batches[@]}" \
  --num-tests "${bench_num_tests}" | tee "${nvfp4_log}"

MASTER_PORT=29722 \
DG_JIT_CACHE_DIR=/tmp/deepgemm_nvfp4_ako_postpatch_w8a8 \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 \
  --batches "${bench_batches[@]}" \
  --num-tests "${bench_num_tests}" | tee "${w8a8_log}"

NVFP4_LOG="${nvfp4_log}" \
W8A8_LOG="${w8a8_log}" \
NVFP4_W8A8_RATIO_MAX="${NVFP4_W8A8_RATIO_MAX:-1.25}" \
python3 - <<'PY'
import os
import re
import sys

nvfp4_text = open(os.environ["NVFP4_LOG"], encoding="utf-8").read()
w8a8_text = open(os.environ["W8A8_LOG"], encoding="utf-8").read()
ratio_max = float(os.environ["NVFP4_W8A8_RATIO_MAX"])

nvfp4 = {
    int(token): float(us)
    for token, us in re.findall(r"tokens=\s*(\d+)\b.*?nvfp4=\s*([0-9]+(?:\.[0-9]+)?)us", nvfp4_text)
}
w8a8 = {
    int(token): float(us)
    for token, us in re.findall(r"tokens=\s*(\d+)\b.*?\s([0-9]+(?:\.[0-9]+)?)\s+us", w8a8_text)
}
common = sorted(set(nvfp4) & set(w8a8))
if not common:
    print("Failed to parse benchmark latency from gate logs.", file=sys.stderr)
    sys.exit(2)

failed = False
for token in common:
    ratio = nvfp4[token] / w8a8[token]
    print(f"M={token}: NVFP4/W8A8 latency ratio {ratio:.3f} ({nvfp4[token]:.1f}us / {w8a8[token]:.1f}us)")
    if ratio > ratio_max:
        failed = True
if failed:
    print(f"FAIL: at least one ratio exceeds threshold {ratio_max:.3f}", file=sys.stderr)
    sys.exit(1)
print(f"PASS: all ratios <= threshold {ratio_max:.3f}")
PY
