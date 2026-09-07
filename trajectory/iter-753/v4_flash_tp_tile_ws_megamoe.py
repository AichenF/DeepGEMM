"""TP tile-warp-specialized Hopper MegaMoE experiment.

This isolated import selects the two-CTA-per-SM persistent schedule whose
microtask is one routed BM8 block and one local intermediate K128 slice.
W13, SwiGLU/FP8 quantization, and every W2 N256 tile for that slice execute
back-to-back; the FP8 intermediate remains CTA-local in shared memory.
"""

from __future__ import annotations

import os
import sys

import torch as _torch


if "v4_flash_tp_native_megamoe" in sys.modules:
    raise RuntimeError(
        "import v4_flash_tp_tile_ws_megamoe before the native module so "
        "its dedicated JIT configuration cannot be shadowed"
    )

_TILE_WS_ENV = {
    "V4_NATIVE_TP_TILE_WS": "1",
    "V4_NATIVE_TP_TILE_N128": "1",
    "V4_NATIVE_TP_TILE_CLUSTER_PAIR": "1",
    "V4_NATIVE_H20_EXACT_OUTER": "0",
    "V4_NATIVE_REGISTER_DEQUANT": "1",
    "V4_NATIVE_RS_K128_BATCH": "1",
    "V4_NATIVE_RS_K64_COMMIT_GROUPS": "0",
    "V4_NATIVE_TWO_CTA_PER_SM": "1",
    "V4_NATIVE_RS_HALF_PREFETCH": "0",
    "V4_NATIVE_NORMALIZED_WEIGHT_SCALE": "1",
    "V4_NATIVE_RS_SCALE_WORD_CACHE": "0",
    "V4_NATIVE_SPLIT_WEIGHT_SCALE_TMA": "0",
    "V4_NATIVE_TILE_WEIGHT_SCALE_TMA": "0",
    "V4_NATIVE_SINGLE_L1_WARMUP_WAVE": "0",
    "V4_NATIVE_L1_WARMUP_WAVES": "0",
    "V4_NATIVE_DUAL_ACTIVE_DISPATCH": "1",
    "V4_NATIVE_TP_LOCAL_BARRIER_FASTPATH": "1",
    "V4_NATIVE_TP_LOCAL_DISPATCH_FASTPATH": "0",
    "V4_NATIVE_TP_LOCAL_DIRECT_COPY": "0",
    "V4_NATIVE_TP_LOCAL_ROUTE_BUILD": "1",
    "V4_NATIVE_TP_LOCAL_PARALLEL_COMBINE_CHUNKS": "1",
    "V4_NATIVE_FOLD_GLOBAL_SCALES": "0",
    "V4_NATIVE_FOLD_W13_GLOBAL_SCALE": "0",
    "V4_NATIVE_FOLD_W2_GLOBAL_SCALE": "0",
}
for _name, _value in _TILE_WS_ENV.items():
    os.environ[_name] = _value

_torch_abi = _torch.__version__.split("+")[0].replace(".", "_")
os.environ["TORCH_EXTENSIONS_DIR"] = os.environ.get(
    "V4_TP_TILE_WS_TORCH_EXTENSIONS_DIR",
    f"/tmp/torch_ext_v4_tp_tile_ws_t{_torch_abi}",
)

from v4_flash_tp_native_megamoe import *  # noqa: E402,F403


TP_TILE_WS = True
TP_TILE_WS_TORCH_EXTENSIONS_DIR = os.environ["TORCH_EXTENSIONS_DIR"]
TP_TILE_WS_CTA_COUNT = 234
TP_TILE_WS_THREADS = 384
TP_TILE_WS_PHYSICAL_N = 128
TP_TILE_WS_CLUSTER_SIZE = 2
TP_TILE_WS_TASK = "route_bm8_x_intermediate_k128_cluster_pair"
