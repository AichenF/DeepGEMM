"""Exact Hopper/H20 outer-pipeline experiment for V4 Flash TP4 MegaMoE.

This is a deliberately isolated JIT configuration of the existing native
MXFP4 kernel.  It selects the Hopper reference outer schedule while reusing
the current TP input, workspace, local reduction, and embedded communication
contracts.  Import this module instead of ``v4_flash_tp_native_megamoe``.
"""

from __future__ import annotations

import os
import sys

import torch as _torch


if "v4_flash_tp_native_megamoe" in sys.modules:
    raise RuntimeError(
        "import v4_flash_tp_h20_exact_megamoe before the native module so "
        "its dedicated JIT configuration cannot be shadowed"
    )

_EXACT_ENV = {
    "V4_NATIVE_H20_EXACT_OUTER": "1",
    "V4_NATIVE_REGISTER_DEQUANT": "0",
    "V4_NATIVE_RS_K128_BATCH": "0",
    "V4_NATIVE_RS_K64_COMMIT_GROUPS": "0",
    "V4_NATIVE_TWO_CTA_PER_SM": "0",
    "V4_NATIVE_RS_HALF_PREFETCH": "0",
    "V4_NATIVE_NORMALIZED_WEIGHT_SCALE": "0",
    "V4_NATIVE_RS_SCALE_WORD_CACHE": "0",
    "V4_NATIVE_SPLIT_WEIGHT_SCALE_TMA": "0",
    "V4_NATIVE_TILE_WEIGHT_SCALE_TMA": "0",
    "V4_NATIVE_SINGLE_L1_WARMUP_WAVE": "0",
    "V4_NATIVE_L1_WARMUP_WAVES": "0",
    "V4_NATIVE_DUAL_ACTIVE_DISPATCH": "0",
    "V4_NATIVE_FOLD_GLOBAL_SCALES": "0",
    "V4_NATIVE_FOLD_W13_GLOBAL_SCALE": "0",
    "V4_NATIVE_FOLD_W2_GLOBAL_SCALE": "0",
}
for _name, _value in _EXACT_ENV.items():
    os.environ[_name] = _value

# The repository's flat control and this experiment can be imported by
# different PyTorch environments on the shared host.  Keep the exact cubin
# cache ABI-specific instead of accepting a same-named extension built by a
# different torch installation.
_torch_abi = _torch.__version__.split("+")[0].replace(".", "_")
os.environ["TORCH_EXTENSIONS_DIR"] = os.environ.get(
    "V4_H20_EXACT_TORCH_EXTENSIONS_DIR",
    f"/tmp/torch_ext_v4_tp_h20_exact_t{_torch_abi}",
)

from v4_flash_tp_native_megamoe import *  # noqa: E402,F403


H20_EXACT_OUTER = True
H20_EXACT_TORCH_EXTENSIONS_DIR = os.environ["TORCH_EXTENSIONS_DIR"]
H20_EXACT_CTA_COUNT = 78
H20_EXACT_THREADS = 384
H20_EXACT_EXPERTS_PER_WAVE = {8: 16, 16: 16, 32: 16, 64: 16, 128: 32}
