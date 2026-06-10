#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)"
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
  --batches 32 \
  --num-tests 5 | tee "${nvfp4_log}"

MASTER_PORT=29722 \
DG_JIT_CACHE_DIR=/tmp/deepgemm_nvfp4_ako_postpatch_w8a8 \
python3 tests/bench_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 32 \
  --num-tests 5 | tee "${w8a8_log}"

NVFP4_LOG="${nvfp4_log}" \
W8A8_LOG="${w8a8_log}" \
NVFP4_W8A8_RATIO_MAX="${NVFP4_W8A8_RATIO_MAX:-1.10}" \
python3 - <<'PY'
import os
import re
import sys

nvfp4_text = open(os.environ["NVFP4_LOG"], encoding="utf-8").read()
w8a8_text = open(os.environ["W8A8_LOG"], encoding="utf-8").read()
ratio_max = float(os.environ["NVFP4_W8A8_RATIO_MAX"])

nvfp4_match = re.search(r"nvfp4=\s*([0-9]+(?:\.[0-9]+)?)us", nvfp4_text)
w8a8_match = re.search(r"tokens=\s*32\b.*?\s([0-9]+(?:\.[0-9]+)?)\s+us", w8a8_text)
if not nvfp4_match or not w8a8_match:
    print("Failed to parse benchmark latency from gate logs.", file=sys.stderr)
    sys.exit(2)

nvfp4_us = float(nvfp4_match.group(1))
w8a8_us = float(w8a8_match.group(1))
ratio = nvfp4_us / w8a8_us
print(f"NVFP4/W8A8 latency ratio: {ratio:.3f} ({nvfp4_us:.1f}us / {w8a8_us:.1f}us)")
if ratio > ratio_max:
    print(f"FAIL: ratio {ratio:.3f} exceeds threshold {ratio_max:.3f}", file=sys.stderr)
    sys.exit(1)
print(f"PASS: ratio {ratio:.3f} <= threshold {ratio_max:.3f}")
PY
