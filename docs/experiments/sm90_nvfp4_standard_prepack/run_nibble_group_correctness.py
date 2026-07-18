"""Run focused exact-NVFP4 correctness with the nibble-group candidate."""

import importlib.util
import os
import sys

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEST_PATH = os.path.join(REPO_ROOT, "tests", "test_nvfp4_mega_moe_sm90_correctness.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_correctness_module():
    spec = importlib.util.spec_from_file_location("nvfp4_correctness_gate", TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _worker(local_rank, num_local_ranks, args):
    import deep_gemm
    from deep_gemm.utils.dist import init_dist
    from nibble_group_candidate import (
        nvfp4_nibble_group_mega_moe,
        transform_nvfp4_weights_for_mega_moe_sm90_nibble_group,
    )

    deep_gemm.nvfp4_mega_moe = nvfp4_nibble_group_mega_moe
    deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90 = (
        transform_nvfp4_weights_for_mega_moe_sm90_nibble_group
    )
    correctness = _load_correctness_module()
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        for weight_scale in args.weight_scales:
            for global_scale_mode in args.global_scale_modes:
                for m_tokens in args.batches:
                    correctness._run_case(
                        args, m_tokens, weight_scale, global_scale_mode,
                        rank_idx, group,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    correctness = _load_correctness_module()
    args = correctness._parse_args()
    if args.intermediate_hidden > 2048:
        raise ValueError("nibble-group candidate is Flash-only")
    args.nvfp4_block_n = 256
    num_local_ranks = int(os.environ.get("DG_NUM_LOCAL_RANKS", "1"))
    torch.multiprocessing.spawn(
        _worker,
        args=(num_local_ranks, args),
        nprocs=num_local_ranks,
        join=True,
    )
