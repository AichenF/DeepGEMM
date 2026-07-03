"""Expose the Pro NVFP4 candidate to the multi-rank matrix runner."""

import importlib.util
import os
import sys


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
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
    from pro_candidate import nvfp4_pro_candidate_mega_moe

    deep_gemm.nvfp4_mega_moe = nvfp4_pro_candidate_mega_moe


def _run_one_config(*args, **kwargs):
    _install_candidate()
    _BENCH.bench_kineto = globals().get("bench_kineto", _BENCH.bench_kineto)
    args[0].nvfp4_block_n = 256
    return _BENCH._run_one_config(*args, **kwargs)
