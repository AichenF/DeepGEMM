#!/usr/bin/env python3
"""Common 8-rank H200 timing harness for split FP8 and pinned PR323.

The production implementation is intentionally timed as one logical MoE call:
all matched L1/L2 ``sm90_fp8_mega_moe`` events are grouped per invocation and
summed.  PR323 remains an unmodified, single-kernel black-box reference.
"""

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


SHAPES = {
    "flash": {
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
    },
}


def load_module(script_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_bench_kineto(impl: str, stat: str):
    def bench_kineto(
        fn,
        kernel_names,
        num_tests: int = 30,
        suppress_kineto_output: bool = False,
        trace_path: str = None,
        flush_l2: bool = True,
        with_multiple_kernels: bool = False,
        barrier=None,
    ):
        import torch
        import torch.distributed as dist

        names = (kernel_names,) if isinstance(kernel_names, str) else tuple(kernel_names)
        flush_l2_bytes = int(os.environ.get("DG_BENCH_FLUSH_L2_BYTES", "0"))
        flush_l2_size = max(0, flush_l2_bytes // 4)

        fn()
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            acc_events=True,
        )
        with profiler:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_l2 and flush_l2_size:
                        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                    if barrier is not None:
                        torch.cuda._sleep(int(2e7))
                        barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

        raw_events = []
        for event in profiler.events():
            name = getattr(event, "name", "")
            if any(kernel_name in name for kernel_name in names):
                elapsed = float(
                    getattr(event, "device_time_total", 0.0)
                    or getattr(event, "cuda_time_total", 0.0)
                    or 0.0
                )
                if elapsed > 0:
                    raw_events.append((name, elapsed))

        raw_us = [elapsed for _, elapsed in raw_events]

        if not raw_us:
            raise RuntimeError(f"no CUDA events matched {names}")

        if impl == "ours":
            if not with_multiple_kernels:
                raise RuntimeError("split FP8 timing must enable with_multiple_kernels")
            if len(raw_us) % num_tests:
                raise RuntimeError(
                    f"cannot group {len(raw_us)} events into {num_tests} MoE calls"
                )
            kernels_per_call = len(raw_us) // num_tests
            samples_us = [
                sum(raw_us[i * kernels_per_call:(i + 1) * kernels_per_call])
                for i in range(num_tests)
            ]
        else:
            samples_us = raw_us[-num_tests:]
            kernels_per_call = 1

        value_us = (
            statistics.median(samples_us)
            if stat == "median"
            else statistics.mean(samples_us)
        )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if int(os.environ.get("DG_BENCH_PRINT_KERNEL_BREAKDOWN", "0")):
            by_kind = {}
            for name, elapsed in raw_events:
                if "l1_impl" in name:
                    kind = "l1"
                elif "l2_no_reduce" in name or "l2_impl" in name:
                    kind = "l2"
                else:
                    kind = name
                by_kind.setdefault(kind, []).append(elapsed)
            print(
                "KERNEL_BREAKDOWN_JSON "
                + json.dumps(
                    {
                        "impl": impl,
                        "rank": rank,
                        "kernels": {
                            kind: {
                                "count": len(values),
                                "median_us": statistics.median(values),
                                "mean_us": statistics.mean(values),
                            }
                            for kind, values in by_kind.items()
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            "BENCH_STAT_JSON "
            + json.dumps(
                {
                    "impl": impl,
                    "rank": rank,
                    "stat": stat,
                    "num_tests": num_tests,
                    "num_samples": len(samples_us),
                    "returned_us": value_us,
                    "mean_us": statistics.mean(samples_us),
                    "median_us": statistics.median(samples_us),
                    "min_us": min(samples_us),
                    "max_us": max(samples_us),
                    "raw_event_count": len(raw_us),
                    "kernels_per_call": kernels_per_call,
                    "samples_us": samples_us,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if trace_path is not None:
            profiler.export_chrome_trace(trace_path)
        return value_us / 1e6

    return bench_kineto


def make_args(impl: str, shape: dict, num_tests: int):
    common = dict(
        ncu_profile_only=False,
        num_processes=8,
        local_rank_idx=None,
        hidden=shape["hidden"],
        intermediate_hidden=shape["intermediate_hidden"],
        num_experts=shape["num_experts"],
        num_topk=shape["num_topk"],
        activation_clamp=10.0,
        fast_math=1,
        masked_ratio=0.0,
    )
    if impl == "ours":
        return SimpleNamespace(**common, num_tests=num_tests, seed=0)
    return SimpleNamespace(**common, num_bench_tests=num_tests)


def worker(local_rank: int, num_processes: int, cfg: dict):
    import torch
    import torch.distributed as dist

    impl = cfg["impl"]
    if impl == "ours":
        repo = os.environ["DG_OURS_REPO"]
        script = os.environ.get(
            "DG_OURS_BENCH", str(Path(repo) / "tests" / "bench_mega_moe_sm90.py")
        )
        sys.path.insert(0, repo)
    else:
        repo = os.environ["DG_PR323_REPO"]
        build = os.environ.get("DG_PR323_BUILD", str(Path(repo) / "build"))
        script = os.environ.get(
            "DG_PR323_BENCH",
            str(Path(repo) / "sgl_deep_gemm" / "tests" / "test_mega_moe_hopper.py"),
        )
        sys.path.insert(0, build)
        sys.path.insert(1, repo)
        sys.path.insert(2, str(Path(script).parent))
        os.chdir("/tmp")

    module = load_module(script, f"h200_fp8_bench_{impl}")
    module.bench_kineto = make_bench_kineto(impl, cfg["stat"])

    def all_rank_print(message, **_kwargs):
        print(f"MATRIX_RANK={dist.get_rank()} {message}", flush=True)

    module.dist_print = all_rank_print
    shape = SHAPES[cfg["shape"]]
    args = make_args(impl, shape, cfg["num_tests"])
    rank_idx, num_ranks, group = module.init_dist(local_rank, num_processes)
    torch.manual_seed(rank_idx + cfg["seed"])
    random.seed(rank_idx + cfg["seed"])

    module.dist_print(
        f"RUN_CONFIG impl={impl} shape={cfg['shape']} M={cfg['m']} "
        f"cap={cfg['cap']} seed={cfg['seed']} stat={cfg['stat']} "
        f"num_tests={cfg['num_tests']}",
        once_in_node=True,
    )
    if impl == "ours":
        case_name = (
            f"{cfg['shape']}:{cfg['m']}:{shape['hidden']}:"
            f"{shape['intermediate_hidden']}:{shape['num_experts']}:"
            f"{shape['num_topk']}"
        )
        # bench_mega_moe_sm90 adds rank and case-name offsets internally.
        # Cancel them so both implementations consume seed + rank.
        args.seed = (
            cfg["seed"]
            + rank_idx
            - rank_idx * 1000003
            - module._stable_name_seed(case_name)
        )
        module._run_one_config(
            args,
            cfg["shape"],
            cfg["m"],
            cfg["cap"],
            shape["hidden"],
            shape["intermediate_hidden"],
            shape["num_experts"],
            shape["num_topk"],
            num_ranks,
            rank_idx,
            group,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
    else:
        module._run_fused_only_config(
            args,
            cfg["m"],
            cfg["cap"],
            shape["hidden"],
            shape["intermediate_hidden"],
            shape["num_experts"],
            shape["num_topk"],
            num_ranks,
            rank_idx,
            group,
        )

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", required=True, choices=("ours", "pr323"))
    parser.add_argument("--shape", required=True, choices=tuple(SHAPES))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tests", type=int, required=True)
    parser.add_argument("--stat", choices=("mean", "median"), default="median")
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    import torch.multiprocessing as mp

    mp.spawn(worker, args=(args.num_processes, vars(args)), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
