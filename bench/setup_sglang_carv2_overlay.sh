#!/usr/bin/env bash
set -euo pipefail

# Build a private compatibility overlay for the symmetric-memory CARv2 used by
# this benchmark.  The source checkout is copied, never modified in place.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_pkg=${SGLANG_CARV2_SOURCE:-/home/xutingz/workspace/codex_correct_forks_20260731/sglang/python/sglang}
overlay_root=${SGLANG_CARV2_OVERLAY:-/home/xutingz/fac/sglang_carv2_overlay_2538def}
patch_file=${script_dir}/patches/sglang_carv2_tvmffi011_ipc.patch
target=${overlay_root}/sglang/jit_kernel/csrc/distributed/ipc.cuh

if [[ ! -f ${source_pkg}/srt/distributed/device_communicators/custom_all_reduce_v2.py ]]; then
  echo "missing symmetric-memory CARv2 source package: ${source_pkg}" >&2
  exit 1
fi

if [[ ! -d ${overlay_root}/sglang ]]; then
  mkdir -p "${overlay_root}"
  cp -a "${source_pkg}" "${overlay_root}/sglang"
fi

if grep -q 'to_ipc_handle(get<0>(pair))' "${target}"; then
  patch -d "${overlay_root}/sglang" -p1 < "${patch_file}"
fi

if ! grep -q 'to_ipc_handle(pair.get<0>())' "${target}"; then
  echo "CARv2 overlay has an unexpected ipc.cuh: ${target}" >&2
  exit 1
fi

echo "SGLANG_CARV2_OVERLAY=${overlay_root}"
