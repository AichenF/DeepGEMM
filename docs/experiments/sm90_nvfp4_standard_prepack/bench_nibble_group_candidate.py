"""Expose the nibble-group candidate to the multi-rank matrix runner."""

import importlib.util
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BENCH_PATH = os.path.join(REPO_ROOT, "tests", "bench_nvfp4_mega_moe_sm90.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("nvfp4_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BENCH = _load_bench_module()
init_dist = _BENCH.init_dist
get_arch_major = _BENCH.get_arch_major
dist_print = _BENCH.dist_print


def _install_candidate():
    import deep_gemm
    from nibble_group_candidate import (
        nvfp4_nibble_group_mega_moe,
        transform_nvfp4_weights_for_mega_moe_sm90_nibble_group,
    )

    deep_gemm.nvfp4_mega_moe = nvfp4_nibble_group_mega_moe
    deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90 = (
        transform_nvfp4_weights_for_mega_moe_sm90_nibble_group
    )


def _run_one_config(*args, **kwargs):
    _install_candidate()
    _BENCH.bench_kineto = globals().get("bench_kineto", _BENCH.bench_kineto)
    args[0].nvfp4_block_n = 256
    return _BENCH._run_one_config(*args, **kwargs)
