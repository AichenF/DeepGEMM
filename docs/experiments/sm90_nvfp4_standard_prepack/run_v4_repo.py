"""Run the existing H20 matrix harness against an explicit repository path."""

import argparse
import importlib.util
import os
import sys


def _load_runner(name: str):
    runner_path = os.environ.get(
        "DG_V4_MATRIX_RUNNER",
        "/root/fac/scripts/megamoe/v4_shape_matrix_runner.py",
    )
    sys.path.insert(0, os.path.dirname(runner_path))
    spec = importlib.util.spec_from_file_location(name, runner_path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = runner
    spec.loader.exec_module(runner)
    runner.SHAPES.setdefault(
        "mimo_pro",
        {
            "hidden": 6144,
            "intermediate_hidden": 2048,
            "num_experts": 384,
            "num_topk": 8,
        },
    )
    return runner


def _worker_with_repo_override(local_rank: int, num_processes: int, cfg: dict):
    runner = _load_runner("v4_shape_matrix_runner_worker")
    impl = os.environ["DG_V4_IMPL_OVERRIDE"]
    runner.REPOS[impl]["repo"] = os.environ["DG_V4_REPO_OVERRIDE"]
    runner.REPOS[impl]["script"] = os.environ["DG_V4_SCRIPT_OVERRIDE"]
    runner.worker(local_rank, num_processes, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--script")
    parser.add_argument("--impl-key", default="ours_nvfp4")
    known, remaining = parser.parse_known_args()

    default_script = (
        "bench_nvfp4_mega_moe_sm90.py"
        if known.impl_key == "ours_nvfp4"
        else "bench_mega_moe_sm90.py"
    )
    script = known.script or os.path.join(known.repo, "tests", default_script)
    os.environ["DG_V4_IMPL_OVERRIDE"] = known.impl_key
    os.environ["DG_V4_REPO_OVERRIDE"] = known.repo
    os.environ["DG_V4_SCRIPT_OVERRIDE"] = script
    runner = _load_runner("v4_shape_matrix_runner")
    runner.worker = _worker_with_repo_override
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
