"""Adapt the optimized W8A8 benchmark to the common H20 matrix runner."""

import importlib.util
import os

import torch.distributed as dist


BENCH_PATH = os.environ.get(
    "DG_W8A8_OPT_BENCH",
    "/root/fac/megamoe/DeepGEMM_fp8_split_swap_ref/tests/bench_mega_moe_sm90.py",
)


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("w8a8_opt_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BENCH = _load_bench_module()
init_dist = _BENCH.init_dist
get_arch_major = _BENCH.get_arch_major
dist_print = _BENCH.dist_print


def _all_rank_dist_print(message, **_kwargs):
    print(f"W8A8_RANK={dist.get_rank()} {message}", flush=True)


def _run_one_config(
    args,
    num_tokens,
    num_max_tokens_per_rank,
    hidden,
    intermediate_hidden,
    num_experts,
    num_topk,
    num_ranks,
    rank_idx,
    group,
    activation_clamp,
    fast_math,
    print_perf=True,
):
    config_name = "flash" if intermediate_hidden == 2048 else "pro"
    case_name = (
        f"{config_name}:{num_tokens}:{hidden}:{intermediate_hidden}:"
        f"{num_experts}:{num_topk}"
    )
    seed_offset = int(os.environ.get("DG_BENCH_SEED_OFFSET", "0"))
    args.seed = (
        seed_offset
        + rank_idx
        - rank_idx * 1000003
        - _BENCH._stable_name_seed(case_name)
    )
    args.num_repeats = 1
    _BENCH.bench_kineto = globals().get("bench_kineto", _BENCH.bench_kineto)
    _BENCH.dist_print = _all_rank_dist_print
    return _BENCH._run_one_config(
        args,
        config_name,
        num_tokens,
        num_max_tokens_per_rank,
        hidden,
        intermediate_hidden,
        num_experts,
        num_topk,
        num_ranks,
        rank_idx,
        group,
        activation_clamp,
        fast_math,
        print_perf=print_perf,
    )
