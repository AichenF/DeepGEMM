#!/usr/bin/env bash
# Formal Flash/Pro final-kernel performance run against the frozen Aichen baseline.
#
# Closed protocol: prove the sealed 75186dd source, apply only the sealed
# 3-file/16-site CUDA 13.2 dependent-template compatibility patch, build once,
# run one excluded complete warmup per model, then 30 fresh complete processes
# per model in alternating Flash/Pro order.
# Performance never controls this runner's exit status; evidence integrity does.
set -Eeuo pipefail

EXPECTED_COMMIT="75186dde9dac140c053c9007ace0ce7cce41150c"
EXPECTED_TREE="7d362b6f6069164fc326faddb2a99e8b158bb5cf"
PUBLISHED_CODE_COMMIT="a8f17ad58f97b49b33ed63a4ed7a7304c897d4af"
PUBLISHED_CODE_TREE="80d517297b932911f785b6ffaf1b6adf608f73c8"
BASE_SOURCE_ID="DeepGEMM final release@${EXPECTED_COMMIT} tree=${EXPECTED_TREE}"
SOURCE_ID="${BASE_SOURCE_ID} + cuda132-dependent-template-compat-3files-16sites"

IMAGE="nvcr.io/nvidia/pytorch@sha256:f572dd504a3fef02277c21f228977f100f7831576ac73140a250c473f74d3ad3"
SEAL_ROOT="/home/xuechengw/MegaMoe/artifacts/final_mimo_cuda132_20260716"
ARTIFACT_ROOT="/home/xuechengw/MegaMoe/artifacts/final_flash_pro_cuda132_20260716"
SOURCE_TAR="${SEAL_ROOT}/deepgemm-75186dd-cuda132-source-with-submodules.tar.gz"
SOURCE_TAR_SHA256="401f50a608becd12b486478f65e6d2ae2a427323e83661c1bd92baf50e70c904"
SOURCE_TOP="deepgemm-75186dd-cuda132-source"
COMPAT_PATCH="${SEAL_ROOT}/deepgemm-75186dd-cuda132-dependent-template-compat.patch"
COMPAT_PATCH_SHA256="333154d5254467c5e4a399d5286b414c405dca7566f927f93cca76ea30fcb07d"
PRE_MANIFEST="${SEAL_ROOT}/source_pre_patch_manifest.sha256"
PRE_MANIFEST_SHA256="1bddc2b997d1614967506eb9988f1a067e009e9c68c57bef74ece1bba4339292"
POST_MANIFEST="${SEAL_ROOT}/source_post_patch_manifest.sha256"
POST_MANIFEST_SHA256="5e2ee5de49ad91aee454a51a8118ef285a1ca3415de0f5d8a9172f7e7f03cab9"
HARNESS_MANIFEST="${SEAL_ROOT}/protected_harness_manifest.sha256"
HARNESS_MANIFEST_SHA256="dba0ee5e82221f00452e9b5fbe424e64007a1f3f581c1ecb210ba43e7e82b434"
PATCH_FILE_HASHES="${SEAL_ROOT}/cuda132_patch_file_hashes.tsv"
PATCH_FILE_HASHES_SHA256="a1f03bbbfb1c07ac438e0d607f06310ba427f9cafece28303d2aeeb3fcd551b4"
SEALING_METADATA="${SEAL_ROOT}/SEALING_METADATA.md"
SEALING_METADATA_SHA256="6e2e864c7dec9d3421dc3e86d231451a84ef7a6f8c48985f91644e41cdd6b41e"

PARSER="${ARTIFACT_ROOT}/summarize_final_flash_pro_vs_baseline_30.py"
PARSER_SHA256="4a6c1c69c769b0514d1f4f76f5a8e1f1ca3aefb72a7047604dcc1a8e3a7af7dd"
COMMON_PARSER="${SEAL_ROOT}/summarize_final_mimo_vs_baseline_30.py"
COMMON_PARSER_SHA256="7b8ca133b0bad86a032b50dd75022d8f519f31d1dea756aeba778990bed51ee9"
P2P_CHECK="/home/xuechengw/MegaMoe/artifacts/mimo25pro_20260715/all_pairs_mapped_p2p.py"
P2P_CHECK_SHA256="7817be83f62c91f6416b02b2ea6674a3b45f5d211478f8b4ae435be32ca14ad1"
BASELINE_DIR="/home/xuechengw/MegaMoe/artifacts/aichen_imageperf_three_models_cuda132_20260716/job3119858_20260716_034631"
BASELINE_HOST="viking-prod-217"
BASELINE_GPU_SNAPSHOT_SHA256="fcce6db2577c6e9c6579941422e58c312fd97921c8620cf26e9e5cbc3bbdb3b4"
BASELINE_LEDGER_SHA256="5cda919490917371551f5e979eb63323ec302d6ebfb8b7df279248e3d13fb8f0"
BASELINE_ARTIFACT_MANIFEST_SHA256="e29d50eec961c77adbaedeb60f7906d41a2d70d1a1e738cf2ca3873e99b19071"

EXPECTED_M="8 16 32 64 128 256 512 1024 2048 4096 8192"
EXPECTED_ROUNDS=30
NUM_TESTS=20
NODE_ROOT_BASE="${NODE_ROOT_BASE:-/tmp/xuechengw}"
SOURCE_STATE="absent"

fail() {
  echo "ERROR: $*" >&2
  return 1
}

sha256_value() {
  sha256sum "$1" | awk '{print $1}'
}

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing required file ${path}"
  [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || fail "unsealed hash placeholder for ${path}: ${expected}"
  [[ "$(sha256_value "${path}")" == "${expected}" ]] || fail "hash mismatch for ${path}"
}

check_manifest_files() {
  local manifest="$1" check_log="$2"
  (cd "${SRC}" && sha256sum --check "${manifest}") > "${check_log}"
}

observe_full_source_manifest() {
  local destination="$1"
  (
    cd "${SRC}"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) > "${destination}"
}

check_exact_source_manifest() {
  local manifest="$1"
  local phase="$2"
  local observed="${OUTDIR}/source_${phase}_observed.sha256"
  check_manifest_files "${manifest}" "${OUTDIR}/source_${phase}_check.log"
  observe_full_source_manifest "${observed}"
  cmp -s "${manifest}" "${observed}" || fail "full source file set/hash mismatch at ${phase}"
}

capture_protected_source() {
  local phase="$1"
  check_manifest_files "${POST_MANIFEST}" "${OUTDIR}/source_postpatch_${phase}_check.log" || return
  check_manifest_files "${HARNESS_MANIFEST}" "${OUTDIR}/protected_harness_${phase}_check.log" || return
}

observe_jit_manifest() {
  local destination="$1"
  (
    cd "${JIT_CACHE}"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) > "${destination}"
}

find_jit_sources() {
  local pattern="$1" destination="$2" source rc
  : > "${destination}"
  while IFS= read -r -d '' source; do
    if grep -q -F -- "${pattern}" "${source}"; then
      printf '%s\n' "${source}" >> "${destination}"
    else
      rc=$?
      (( rc == 1 )) || fail "cannot inspect JIT source ${source} (grep exit ${rc})"
    fi
  done < <(find "${JIT_CACHE}" -type f -name kernel.cu -print0)
  LC_ALL=C sort -o "${destination}" "${destination}"
}

hash_listed_files() {
  local paths_file="$1" destination="$2"
  [[ -s "${paths_file}" ]] || fail "empty JIT source list ${paths_file}"
  xargs -r sha256sum < "${paths_file}" > "${destination}"
}

HOST_TAG="$(hostname | tr -c 'A-Za-z0-9_.-' '_')"
START_TAG="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="job${SLURM_JOB_ID}_${START_TAG}"
  EXECUTION_MODE="slurm_allocation_host_docker"
else
  RUN_ID="host${HOST_TAG}_${START_TAG}"
  EXECUTION_MODE="direct_host_docker"
fi
RUN_ROOT="${NODE_ROOT_BASE}/final-flash-pro-cuda132-${RUN_ID}"
SRC="${RUN_ROOT}/${SOURCE_TOP}"
NODE_OUT="${RUN_ROOT}/out"
JIT_CACHE="${NODE_OUT}/dg_jit_cache"
P2P_NODE="${RUN_ROOT}/all_pairs_mapped_p2p.py"
OUTDIR="${ARTIFACT_ROOT}/${RUN_ID}"
PARSER_SNAPSHOT="${OUTDIR}/summarize_final_flash_pro_vs_baseline_30.snapshot.py"
COMMON_PARSER_SNAPSHOT="${OUTDIR}/common_final_parser.py"
RUNNER_SNAPSHOT="${OUTDIR}/run_final_flash_pro_cuda132_30.snapshot.sh"

[[ ! -e "${RUN_ROOT}" ]] || { echo "ERROR: run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ ! -e "${OUTDIR}" ]] || { echo "ERROR: artifact directory exists: ${OUTDIR}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}" "${JIT_CACHE}" "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/runner.log") 2>&1

on_exit() {
  local rc=$? protocol_status="NOT_COMPLETED" run_status="EXECUTION_FAIL"
  trap - EXIT
  set +e
  if [[ "${SOURCE_STATE}" == "postpatch" && -f "${SRC}/tests/bench_nvfp4_mega_moe_sm90.py" ]]; then
    capture_protected_source after_exit || rc=90
  fi
  [[ -f "${OUTDIR}/protocol_status.txt" ]] && protocol_status="$(tr -d '[:space:]' < "${OUTDIR}/protocol_status.txt")"
  if (( rc == 0 )) && [[ "${protocol_status}" == "COMPLETED" ]]; then
    run_status="PASS"
  fi
  printf '%s\n' "${rc}" > "${OUTDIR}/exit_code.txt"
  printf '%s\n' "${run_status}" > "${OUTDIR}/run_status.txt"
  find "${OUTDIR}" -maxdepth 1 -type f ! -name runner.log ! -name artifact_sha256.txt -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > "${OUTDIR}/artifact_sha256.txt"
  echo "protocol_status=${protocol_status}"
  echo "run_status=${run_status}"
  echo "artifacts=${OUTDIR}"
  echo "node_workspace=${RUN_ROOT}"
  exit "${rc}"
}
trap on_exit EXIT

run_logged() {
  local log_file="$1"; shift
  local rc
  set +e
  "$@" > "${log_file}" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    cat "${log_file}" >&2
    fail "command failed with exit=${rc}; see ${log_file}"
  fi
}

gpu_smi() {
  docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" "$@"
}

assert_gpus_idle() {
  local label="$1" pids
  pids="$(gpu_smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  if [[ -n "${pids}" ]]; then
    printf '%s\tFAIL\t%s\t%s\n' "${label}" "$(date --iso-8601=seconds)" "${pids//$'\n'/,}" >> "${OUTDIR}/gpu_idle_checks.tsv"
    fail "unexpected GPU compute processes at ${label}: ${pids//$'\n'/,}"
  fi
  printf '%s\tPASS\t%s\t-\n' "${label}" "$(date --iso-8601=seconds)" >> "${OUTDIR}/gpu_idle_checks.tsv"
}

capture_gpu_snapshot() {
  local path="$1"
  gpu_smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.memory,driver_version,pstate --format=csv > "${path}"
}

capture_telemetry() {
  local phase="$1"
  capture_gpu_snapshot "${OUTDIR}/telemetry_${phase}_core.csv"
  docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" -q > "${OUTDIR}/telemetry_${phase}_nvidia_smi_q.txt" 2>&1
  docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" nvlink --status > "${OUTDIR}/telemetry_${phase}_nvlink_status.txt" 2>&1
  if ! docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" nvlink --error-counters > "${OUTDIR}/telemetry_${phase}_nvlink_errors.txt" 2>&1; then
    docker run --rm --gpus all --entrypoint nvidia-smi "${IMAGE}" nvlink -e >> "${OUTDIR}/telemetry_${phase}_nvlink_errors.txt" 2>&1
  fi
}

bench_command() {
  local model="$1" hidden intermediate experts topk
  case "${model}" in
    flash) hidden=4096; intermediate=2048; experts=256; topk=6 ;;
    pro) hidden=7168; intermediate=3072; experts=384; topk=6 ;;
    *) fail "unknown model ${model}"; return 1 ;;
  esac
  printf '%s' "set -euo pipefail
bad_dg=\"\$(env | awk -F= '\$1 ~ /^DG_/ && \$1 != \"DG_JIT_CACHE_DIR\" {print \$1}')\"
if [[ -n \"\${bad_dg}\" ]]; then echo \"unexpected DG environment: \${bad_dg}\" >&2; exit 3; fi
if [[ \"\${DG_JIT_CACHE_DIR:-}\" != \"/out/dg_jit_cache\" ]]; then echo bad DG_JIT_CACHE_DIR >&2; exit 3; fi
if env | grep -E '^(SGLANG_MEGAMOE_NVFP4_REQUANTIZE|NSYS_ITERS)='; then echo unexpected external benchmark override >&2; exit 3; fi
exec python3 tests/bench_nvfp4_mega_moe_sm90.py --num-processes 8 --batches 8 16 32 64 128 256 512 1024 2048 4096 8192 --hidden ${hidden} --intermediate-hidden ${intermediate} --num-experts ${experts} --num-topk ${topk} --num-tests ${NUM_TESTS}"
}

run_process() {
  local label="$1" model="$2" kind="$3" command log_file started_at ended_at log_sha
  command="$(bench_command "${model}")"
  log_file="${OUTDIR}/${label}.log"
  [[ ! -e "${log_file}" ]] || fail "refusing to replace ${log_file}"
  assert_gpus_idle "before_${label}"
  capture_gpu_snapshot "${OUTDIR}/${label}_gpu_before.csv"
  started_at="$(date --iso-8601=seconds)"
  echo "=== ${label} model=${model} kind=${kind} start=${started_at} ==="
  run_logged "${log_file}" \
    docker run --rm --network=host --ipc=host --gpus all --cap-add SYS_NICE \
      -v "${SRC}:/workspace/DeepGEMM" -v "${NODE_OUT}:/out" \
      -w /workspace/DeepGEMM -e PYTHONPATH=/workspace/DeepGEMM \
      -e TORCH_CUDA_ARCH_LIST=9.0a -e DG_JIT_CACHE_DIR=/out/dg_jit_cache \
      "${IMAGE}" bash -lc "${command}"
  if [[ "${kind}" == "excluded_warmup" ]]; then
    python3 "${PARSER_SNAPSHOT}" --validate-log "${log_file}" --model "${model}" --allow-compile
  else
    python3 "${PARSER_SNAPSHOT}" --validate-log "${log_file}" --model "${model}"
  fi
  ended_at="$(date --iso-8601=seconds)"
  assert_gpus_idle "after_${label}"
  log_sha="$(sha256_value "${log_file}")"
  printf '%s\t%s\t%s\t%s\t0\t%s\n' "${label}" "${kind}" "${started_at}" "${ended_at}" "${log_sha}" >> "${OUTDIR}/process_ledger.tsv"
  echo "=== ${label} end=${ended_at} sha256=${log_sha} ==="
}

echo "=== Final 75186dd Flash/Pro CUDA 13.2: 2 warmups + 60 measured processes ==="
echo "start=$(date --iso-8601=seconds)"
echo "run_id=${RUN_ID} execution_mode=${EXECUTION_MODE} host=$(hostname)"
echo "source=${SOURCE_ID}"
echo "image=${IMAGE}"
echo "shapes=flash:H4096/I2048/E256/topk6;pro:H7168/I3072/E384/topk6;ranks8 ordered_m=${EXPECTED_M}"
echo "protocol=one excluded warmup/model; 30 fresh complete measured processes/model; alternating FP/PF; num_tests=${NUM_TESTS}"
echo "sample_policy=no retries/deletion/replacement/best-of-N; performance is not an exit gate"

for required_pair in \
  "${SOURCE_TAR}|${SOURCE_TAR_SHA256}" \
  "${COMPAT_PATCH}|${COMPAT_PATCH_SHA256}" \
  "${PRE_MANIFEST}|${PRE_MANIFEST_SHA256}" \
  "${POST_MANIFEST}|${POST_MANIFEST_SHA256}" \
  "${HARNESS_MANIFEST}|${HARNESS_MANIFEST_SHA256}" \
  "${PATCH_FILE_HASHES}|${PATCH_FILE_HASHES_SHA256}" \
  "${SEALING_METADATA}|${SEALING_METADATA_SHA256}" \
  "${PARSER}|${PARSER_SHA256}" \
  "${COMMON_PARSER}|${COMMON_PARSER_SHA256}" \
  "${P2P_CHECK}|${P2P_CHECK_SHA256}"; do
  require_hash "${required_pair%%|*}" "${required_pair#*|}"
done
python3 -m py_compile "${PARSER}" "${COMMON_PARSER}"
python3 "${PARSER}" --self-test > "${OUTDIR}/parser_self_test.log"
python3 "${PARSER}" --validate-baseline --baseline-dir "${BASELINE_DIR}" > "${OUTDIR}/baseline_raw_preflight.log"

require_hash "${BASELINE_DIR}/gpu_before.csv" "${BASELINE_GPU_SNAPSHOT_SHA256}"
require_hash "${BASELINE_DIR}/process_ledger.tsv" "${BASELINE_LEDGER_SHA256}"
require_hash "${BASELINE_DIR}/artifact_sha256.txt" "${BASELINE_ARTIFACT_MANIFEST_SHA256}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  [[ "${PULL_IF_MISSING:-1}" == "1" ]] || fail "pinned CUDA 13.2 image is missing"
  docker pull "${IMAGE}"
fi
docker image inspect --format 'Id={{.Id}} RepoDigests={{json .RepoDigests}} Created={{.Created}}' "${IMAGE}" > "${OUTDIR}/container_image.inspect.txt"
grep -F "${IMAGE##*@}" "${OUTDIR}/container_image.inspect.txt" >/dev/null || fail "resolved image digest mismatch"

install -m 0444 "${PARSER}" "${PARSER_SNAPSHOT}"
install -m 0444 "${COMMON_PARSER}" "${COMMON_PARSER_SNAPSHOT}"
install -m 0444 "$0" "${RUNNER_SNAPSHOT}"
install -m 0444 "${P2P_CHECK}" "${P2P_NODE}"
for identity_file in "${COMPAT_PATCH}" "${PRE_MANIFEST}" "${POST_MANIFEST}" \
                     "${HARNESS_MANIFEST}" "${PATCH_FILE_HASHES}" "${SEALING_METADATA}"; do
  install -m 0444 "${identity_file}" "${OUTDIR}/$(basename "${identity_file}")"
done
printf '%s\n' "${SOURCE_ID}" > "${OUTDIR}/source_identity.txt"
printf '%s\n' "${IMAGE}" > "${OUTDIR}/container_image.requested.txt"
printf 'label\tstatus\tobserved_at\tcompute_pids\n' > "${OUTDIR}/gpu_idle_checks.tsv"
printf 'label\tkind\tstarted_at\tended_at\texit_code\tlog_sha256\n' > "${OUTDIR}/process_ledger.tsv"

{
  echo "runner_sha256=$(sha256_value "$0")"
  echo "parser_sha256=$(sha256_value "${PARSER}")"
  echo "common_parser_sha256=$(sha256_value "${COMMON_PARSER}")"
  echo "base_source=${BASE_SOURCE_ID}"
  echo "executed_source_variant=${SOURCE_ID}"
  echo "published_code_commit=${PUBLISHED_CODE_COMMIT}"
  echo "published_code_tree=${PUBLISHED_CODE_TREE}"
  echo "stock_source=false"
  echo "compatibility_patch_sha256=${COMPAT_PATCH_SHA256}"
  echo "compatibility_patch_files=3 compatibility_patch_edits=16 syntax_only=true"
  echo "source_tar_sha256=${SOURCE_TAR_SHA256}"
  echo "source_pre_manifest_sha256=${PRE_MANIFEST_SHA256}"
  echo "source_post_manifest_sha256=${POST_MANIFEST_SHA256}"
  echo "container_image=${IMAGE}"
  echo "execution_mode=${EXECUTION_MODE}"
  echo "required_host=${BASELINE_HOST} exact_GPU_UUID_order=true"
  echo "required_cuda=torch13.2 PATH_nvcc13.2 DeepGEMM_JIT_nvcc13.2"
  echo "shapes=flash:H4096:I2048:E256:topk6:ranks8;pro:H7168:I3072:E384:topk6:ranks8"
  echo "m_order=${EXPECTED_M}"
  echo "protocol=1 excluded complete warmup/model + 30 fresh complete measured processes/model; odd rounds Flash-Pro; even rounds Pro-Flash"
  echo "num_tests=${NUM_TESTS}"
  echo "block_n_policy=release-auto; no --nvfp4-block-n; both models BN256 for M<=1024 and BN128 for M>=2048"
  echo "dispatch_policy=Flash BN256 grouped-nibble and BN128 split; Pro BN256 standard fused and BN128 split"
  echo "environment_policy=only DG_JIT_CACHE_DIR allowed among DG_*"
  echo "primary_statistic=rank0 median30; mean-rank and max-rank also reported"
  echo "sample_policy=no retry, deletion, replacement, best-of-N, or latency selection"
  echo "performance_exit_gate=false"
  echo "frozen_baseline_dir=${BASELINE_DIR}"
} > "${OUTDIR}/identity_manifest.txt"

echo "=== Same-node hardware and CUDA 13.2 preflight ==="
hostname > "${OUTDIR}/hostname.txt"
[[ "$(hostname)" == "${BASELINE_HOST}" ]] || fail "final run is not on frozen baseline host ${BASELINE_HOST}"
if [[ "${EXECUTION_MODE}" == "slurm_allocation_host_docker" ]]; then
  timeout 15s scontrol show job "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
  grep -Eq '(^|[[:space:]])JobState=RUNNING([[:space:]]|$)' "${OUTDIR}/slurm_job.txt" || fail "Slurm job is not RUNNING"
  grep -Eq '(^|[[:space:]])NumNodes=1([[:space:]]|$)' "${OUTDIR}/slurm_job.txt" || fail "Slurm allocation is not one node"
  grep -Eq 'AllocTRES=.*gres/gpu=8.*gres/gpu:h200=8' "${OUTDIR}/slurm_job.txt" || fail "Slurm allocation is not exactly 8 H200"
  grep -F "NodeList=$(hostname)" "${OUTDIR}/slurm_job.txt" >/dev/null || fail "Slurm NodeList differs from host"
else
  printf 'execution_mode=direct_host_docker\n' > "${OUTDIR}/slurm_job.txt"
fi

capture_gpu_snapshot "${OUTDIR}/gpu_before.csv"
awk -F', *' 'NR>1 {print $1", "$2", "$3", "$14}' "${BASELINE_DIR}/gpu_before.csv" > "${OUTDIR}/baseline_gpu_identity.csv"
gpu_smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader > "${OUTDIR}/final_gpu_identity.csv"
cmp -s "${OUTDIR}/baseline_gpu_identity.csv" "${OUTDIR}/final_gpu_identity.csv" || fail "GPU index/name/UUID/driver differ from baseline"
gpu_smi topo -m > "${OUTDIR}/topology.txt"
gpu_count="$(gpu_smi --query-gpu=name --format=csv,noheader | awk '$0 ~ /NVIDIA H200/ {n++} END{print n+0}')"
[[ "${gpu_count}" == 8 ]] || fail "expected exactly 8 visible H200 GPUs, got ${gpu_count}"
awk '/^GPU[0-7][[:space:]]/ {row=substr($1,4)+0; rows++; for(col=0;col<8;col++){v=$(col+2); want=(col==row?"X":"NV18"); if(v!=want){bad=1}}} END{if(rows!=8||bad)exit 1}' "${OUTDIR}/topology.txt" || fail "node is not all-pairs NV18"
assert_gpus_idle preflight
capture_telemetry before

docker run --rm "${IMAGE}" env | sort > "${OUTDIR}/container_base_environment.txt"
run_logged "${OUTDIR}/container_smoke.log" docker run --rm --gpus all "${IMAGE}" bash -lc 'set -euo pipefail
python3 -c "import json,torch,sys; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; assert str(torch.version.cuda)==\"13.2\"; assert str(torch.__version__).startswith(\"2.11.0a0\"); assert list(torch.cuda.nccl.version())==[2,29,7]; assert len(names)==8 and all(\"H200\" in x for x in names); print(\"CUDA132_ENV_JSON=\"+json.dumps({\"python\":sys.version,\"torch\":torch.__version__,\"cuda\":torch.version.cuda,\"nccl\":torch.cuda.nccl.version(),\"devices\":names},separators=(\",\",\":\")))"
nvcc --version | tee /tmp/nvcc.txt; grep -Eq "release 13\\.2([,.]|$)" /tmp/nvcc.txt; echo CUDA132_NVCC_PASS'
run_logged "${OUTDIR}/mapped_p2p.log" docker run --rm --network=host --ipc=host --gpus all --cap-add SYS_NICE -v "${P2P_NODE}:/workspace/all_pairs_mapped_p2p.py:ro" "${IMAGE}" python3 /workspace/all_pairs_mapped_p2p.py
[[ "$(grep -cFx 'ALL_PAIRS_MAPPED_P2P_PASS pairs=56' "${OUTDIR}/mapped_p2p.log")" == 1 ]] || fail "mapped P2P 56/56 did not pass"
assert_gpus_idle after_p2p

echo "=== Restore and prove exact final source, then apply sealed syntax patch ==="
tar -tzf "${SOURCE_TAR}" > "${OUTDIR}/source_tar_members.txt"
awk -v top="${SOURCE_TOP}/" '$0 !~ "^"top {bad=1} /(^|\/)\.\.($|\/)/ {bad=1} END{exit bad}' "${OUTDIR}/source_tar_members.txt" || fail "source tar has unexpected top-level or traversal member"
tar -xzf "${SOURCE_TAR}" -C "${RUN_ROOT}"
[[ -d "${SRC}" ]] || fail "source tar did not create ${SRC}"
SOURCE_STATE="prepatch"
check_exact_source_manifest "${PRE_MANIFEST}" prepatch
check_manifest_files "${HARNESS_MANIFEST}" "${OUTDIR}/protected_harness_prepatch_check.log"
[[ "$(find "${SRC}/deep_gemm" -maxdepth 1 -type f -name '_C*.so' | wc -l)" == 0 ]] || fail "source tar contains prebuilt extension"
[[ "$(find "${JIT_CACHE}" -mindepth 1 | wc -l)" == 0 ]] || fail "JIT cache is not initially empty"

git apply --numstat "${COMPAT_PATCH}" > "${OUTDIR}/compat_patch_numstat.tsv"
cat > "${OUTDIR}/compat_patch_expected_numstat.tsv" <<'EOF'
6	6	deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_nibble_group_fused_body.inl
6	6	deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_fused_body.inl
4	4	deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_split_l1_body.inl
EOF
cmp -s "${OUTDIR}/compat_patch_expected_numstat.tsv" "${OUTDIR}/compat_patch_numstat.tsv" || fail "compat patch is not exactly 3 files / 16 replacements"
(cd "${SRC}" && git apply --check --whitespace=error-all "${COMPAT_PATCH}")
(cd "${SRC}" && git apply --whitespace=error-all "${COMPAT_PATCH}")
SOURCE_STATE="postpatch"
check_exact_source_manifest "${POST_MANIFEST}" postpatch
check_manifest_files "${HARNESS_MANIFEST}" "${OUTDIR}/protected_harness_postpatch_check.log"

echo "=== Build native extension once with CUDA 13.2 ==="
run_logged "${OUTDIR}/build.log" docker run --rm --network=host --ipc=host --gpus all --cap-add SYS_NICE -v "${SRC}:/workspace/DeepGEMM" -w /workspace/DeepGEMM "${IMAGE}" bash -lc 'set -euo pipefail; MAX_JOBS=64 DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 TORCH_CUDA_ARCH_LIST=9.0a python3 setup.py build_ext --inplace --force; PYTHONPATH=/workspace/DeepGEMM python3 -c "import deep_gemm,deep_gemm._C; print(deep_gemm._C.__file__)"'
mapfile -t built_extensions < <(find "${SRC}/deep_gemm" -maxdepth 1 -type f -name '_C*.so')
[[ "${#built_extensions[@]}" == 1 ]] || fail "expected one built extension, got ${#built_extensions[@]}"
sha256sum "${built_extensions[0]}" > "${OUTDIR}/built_extension_sha256.txt"
run_logged "${OUTDIR}/deep_gemm_cuda_resolution.log" docker run --rm --network=host --ipc=host --gpus all -v "${SRC}:/workspace/DeepGEMM" -w /workspace/DeepGEMM -e PYTHONPATH=/workspace/DeepGEMM "${IMAGE}" python3 -c 'import deep_gemm,os,re,subprocess; h=deep_gemm._find_cuda_home(); o=subprocess.check_output([os.path.join(h,"bin","nvcc"),"--version"],text=True); assert re.search(r"release 13\.2(?:,|\s|$)",o); print("DEEP_GEMM_CUDA_HOME="+h); print(o,end=""); print("DEEP_GEMM_JIT_NVCC_PASS release=13.2")'
run_logged "${OUTDIR}/final_layout_policy.log" docker run --rm --gpus all -v "${SRC}:/workspace/DeepGEMM" -w /workspace/DeepGEMM -e PYTHONPATH=/workspace/DeepGEMM "${IMAGE}" python3 -c 'import os; assert "DG_SM90_NVFP4_NIBBLE_GROUP" not in os.environ; from deep_gemm.mega import choose_nvfp4_block_n_for_mega_moe_sm90 as c; ms=(8,16,32,64,128,256,512,1024,2048,4096,8192); want=[256]*8+[128]*3; flash=[c(m,6,32,2048) for m in ms]; pro=[c(m,6,48,3072) for m in ms]; assert flash==want,(flash,want); assert pro==want,(pro,want); print("FINAL_FLASH_PRO_RELEASE_AUTO_POLICY_PASS block_n=256_for_M_le_1024 block_n=128_for_M_ge_2048 grouped_default_env_unset")'
assert_gpus_idle after_build
capture_protected_source before_measurement

bench_command flash > "${OUTDIR}/benchmark_command_flash.txt"
bench_command pro > "${OUTDIR}/benchmark_command_pro.txt"
docker run --rm --network=host --ipc=host --gpus all --cap-add SYS_NICE -v "${SRC}:/workspace/DeepGEMM" -v "${NODE_OUT}:/out" -w /workspace/DeepGEMM -e PYTHONPATH=/workspace/DeepGEMM -e TORCH_CUDA_ARCH_LIST=9.0a -e DG_JIT_CACHE_DIR=/out/dg_jit_cache "${IMAGE}" env | sort > "${OUTDIR}/benchmark_environment.txt"

echo "=== One complete Flash warmup (excluded) ==="
run_process warmup_flash flash excluded_warmup
observe_jit_manifest "${OUTDIR}/jit_cache_after_flash_warmup.sha256"
[[ -s "${OUTDIR}/jit_cache_after_flash_warmup.sha256" ]] || fail "Flash warmup did not populate JIT cache"
find_jit_sources 'sm90_nvfp4_mega_moe_nibble_group_fused_impl<' "${OUTDIR}/flash_grouped_nibble_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_fused_impl<' "${OUTDIR}/flash_standard_fused_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_split_l1_impl<' "${OUTDIR}/flash_split_l1_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_split_l2_impl<' "${OUTDIR}/flash_split_l2_jit_sources.txt"
[[ -s "${OUTDIR}/flash_grouped_nibble_jit_sources.txt" ]] || fail "Flash warmup did not prove grouped-nibble fused JIT path"
[[ ! -s "${OUTDIR}/flash_standard_fused_jit_sources.txt" ]] || fail "Flash unexpectedly compiled standard fused JIT path"
[[ -s "${OUTDIR}/flash_split_l1_jit_sources.txt" ]] || fail "Flash warmup did not prove split L1 JIT path"
[[ -s "${OUTDIR}/flash_split_l2_jit_sources.txt" ]] || fail "Flash warmup did not prove split L2 JIT path"
hash_listed_files "${OUTDIR}/flash_grouped_nibble_jit_sources.txt" "${OUTDIR}/flash_grouped_nibble_jit_sources.sha256"

echo "=== One complete Pro warmup (excluded) ==="
run_process warmup_pro pro excluded_warmup
observe_jit_manifest "${OUTDIR}/jit_cache_after_all_warmups.sha256"
find_jit_sources 'sm90_nvfp4_mega_moe_nibble_group_fused_impl<' "${OUTDIR}/all_grouped_nibble_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_fused_impl<' "${OUTDIR}/pro_standard_fused_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_split_l1_impl<' "${OUTDIR}/all_split_l1_jit_sources.txt"
find_jit_sources 'sm90_nvfp4_mega_moe_split_l2_impl<' "${OUTDIR}/all_split_l2_jit_sources.txt"
[[ -s "${OUTDIR}/pro_standard_fused_jit_sources.txt" ]] || fail "Pro warmup did not prove standard fused JIT path"
[[ -s "${OUTDIR}/all_split_l1_jit_sources.txt" ]] || fail "Pro/Flash warmups did not prove split L1 JIT path"
[[ -s "${OUTDIR}/all_split_l2_jit_sources.txt" ]] || fail "Pro/Flash warmups did not prove split L2 JIT path"
hash_listed_files "${OUTDIR}/all_grouped_nibble_jit_sources.txt" "${OUTDIR}/all_grouped_nibble_jit_sources.sha256"
cmp -s "${OUTDIR}/flash_grouped_nibble_jit_sources.sha256" "${OUTDIR}/all_grouped_nibble_jit_sources.sha256" || fail "Pro warmup unexpectedly added grouped-nibble JIT sources"
: > "${OUTDIR}/pro_single_active_dispatch_warp_jit_sources.txt"
while IFS= read -r source; do
  if grep -q -F -- '/* kSingleActiveDispatchWarp */ true' "${source}"; then
    printf '%s\n' "${source}" >> "${OUTDIR}/pro_single_active_dispatch_warp_jit_sources.txt"
  else
    rc=$?
    (( rc == 1 )) || fail "cannot inspect Pro JIT source ${source} (grep exit ${rc})"
  fi
done < "${OUTDIR}/pro_standard_fused_jit_sources.txt"
LC_ALL=C sort -o "${OUTDIR}/pro_single_active_dispatch_warp_jit_sources.txt" "${OUTDIR}/pro_single_active_dispatch_warp_jit_sources.txt"
[[ -s "${OUTDIR}/pro_single_active_dispatch_warp_jit_sources.txt" ]] || fail "Pro warmup did not prove a standard fused single-active-dispatch-warp instance"
(
  cd "${JIT_CACHE}"
  find . -type f -name kernel.cu -print0 | LC_ALL=C sort -z \
    | tar --null --files-from=- -czf "${OUTDIR}/jit_kernel_sources_after_warmups.tar.gz"
)
sha256sum "${OUTDIR}/jit_kernel_sources_after_warmups.tar.gz" > "${OUTDIR}/jit_kernel_sources_after_warmups.sha256"

echo "=== 30 fresh complete measured processes per model, alternating order ==="
for round in $(seq 1 "${EXPECTED_ROUNDS}"); do
  if (( round % 2 == 1 )); then
    model_order=(flash pro)
  else
    model_order=(pro flash)
  fi
  for model in "${model_order[@]}"; do
    printf -v label 'round_%02d_%s' "${round}" "${model}"
    run_process "${label}" "${model}" measured
  done
done

observe_jit_manifest "${OUTDIR}/jit_cache_after_measurements.sha256"
cmp -s "${OUTDIR}/jit_cache_after_all_warmups.sha256" "${OUTDIR}/jit_cache_after_measurements.sha256" || fail "JIT cache changed during formal measurements"
capture_gpu_snapshot "${OUTDIR}/gpu_after.csv"
capture_telemetry after
assert_gpus_idle final
capture_protected_source after_measurement

echo "=== Strict raw-log baseline-versus-final recomputation ==="
python3 "${PARSER_SNAPSHOT}" --run-dir "${OUTDIR}" --baseline-dir "${BASELINE_DIR}" | tee "${OUTDIR}/summary_stdout.txt"
printf 'COMPLETED\n' > "${OUTDIR}/protocol_status.txt"
echo "finish=$(date --iso-8601=seconds) protocol=COMPLETED performance=REPORTED_NOT_GATED"
echo "SUCCESS: trustworthy final Flash/Pro 30-process comparison completed"
