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

NVFP4_LOG="${nvfp4_log}" \
NVFP4_GATE_BENCH_BATCHES="${NVFP4_GATE_BENCH_BATCHES:-256 512 1024 2048 4096 8192}" \
python3 - <<'PY'
import os
import re
import sys

nvfp4_text = open(os.environ["NVFP4_LOG"], encoding="utf-8").read()
expected = [int(token) for token in os.environ["NVFP4_GATE_BENCH_BATCHES"].split()]

nvfp4 = {
    int(token): float(us)
    for token, us in re.findall(r"tokens=\s*(\d+)\b.*?nvfp4=\s*([0-9]+(?:\.[0-9]+)?)us", nvfp4_text)
}
missing = [token for token in expected if token not in nvfp4]
if missing:
    print(f"Failed to parse NVFP4 benchmark latency for M={missing}.", file=sys.stderr)
    sys.exit(2)

failed = False
for token in expected:
    latency = nvfp4[token]
    print(f"M={token}: NVFP4 latency {latency:.1f}us")
    if latency <= 0:
        failed = True
if failed:
    print("FAIL: at least one parsed NVFP4 latency is non-positive.", file=sys.stderr)
    sys.exit(1)
print("PASS: parsed all requested NVFP4 benchmark latencies.")
PY
