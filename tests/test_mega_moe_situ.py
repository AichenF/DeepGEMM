"""Deterministic single-GPU metamorphic regression for FP8xFP4 MegaMoE SiTU."""

import os
import socket
import sys
import unittest
from typing import Dict, NamedTuple, Optional, Tuple

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


NUM_RANKS = 1
NUM_EXPERTS = 4
NUM_TOPK = 1
NUM_TOKENS = 64
HIDDEN = 1024
INTERMEDIATE = 512


class Case(NamedTuple):
    activation: str
    situ_beta: Optional[float]
    situ_linear_beta: Optional[float]


CASES = {
    'swiglu': Case('swiglu', None, None),
    'situ_hi': Case('situ', 4096.0, 4096.0),
    'situ_gate_low': Case('situ', 0.25, 4096.0),
    'situ_linear_low': Case('situ', 4096.0, 0.5),
    'situ_tiny': Case('situ', 1e-40, 1e-40),
}


def _cast_weights_to_fp4(
        bf16_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = bf16_weights.shape
    w = torch.empty(
        (num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    w_sf = torch.empty(
        (num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for group_idx in range(num_groups):
        w[group_idx], w_sf[group_idx] = per_token_cast_to_fp4(
            bf16_weights[group_idx], use_ue8m0=True, gran_k=32)
    w_sf = deep_gemm.transform_sf_into_required_layout(
        w_sf, n, k, (1, 32), num_groups)
    return w, w_sf


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_f = actual.float()
    reference_f = reference.float()
    denominator = max(
        torch.linalg.vector_norm(reference_f).item(), 1e-12)
    return (
        torch.linalg.vector_norm(actual_f - reference_f).item()
        / denominator)


def _validate_suite(
        results: Dict[str, torch.Tensor], fast_math: bool
) -> None:
    mode = f'fast_math={fast_math}'
    for case_name, y in results.items():
        if not bool(torch.isfinite(y.float()).all().item()):
            raise AssertionError(f'{mode}/{case_name} contains NaN or Inf')

    swiglu = results['swiglu']
    if torch.linalg.vector_norm(swiglu.float()).item() <= 1e-6:
        raise AssertionError(f'{mode}/swiglu is identically zero or degenerate')
    hi = _relative_l2(results['situ_hi'], swiglu)
    gate_low = _relative_l2(results['situ_gate_low'], swiglu)
    linear_low = _relative_l2(results['situ_linear_low'], swiglu)
    low_vs_low = _relative_l2(
        results['situ_gate_low'], results['situ_linear_low'])

    low_floor = max(0.05, 4.0 * hi)
    low_pair_floor = max(0.025, 2.0 * hi)
    if hi >= 0.05:
        raise AssertionError(
            f'{mode}: SiTU(4096,4096) is not close to SwiGLU '
            f'(relative L2={hi:.6f})')
    if gate_low <= low_floor:
        raise AssertionError(
            f'{mode}: SiTU(.25,4096) is not significantly different '
            f'from SwiGLU ({gate_low:.6f} <= {low_floor:.6f})')
    if linear_low <= low_floor:
        raise AssertionError(
            f'{mode}: SiTU(4096,.5) is not significantly different '
            f'from SwiGLU ({linear_low:.6f} <= {low_floor:.6f})')
    if low_vs_low <= low_pair_floor:
        raise AssertionError(
            f'{mode}: the two low-beta controls are not distinct '
            f'({low_vs_low:.6f} <= {low_pair_floor:.6f})')


def _worker(local_rank: int, master_port: int) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'

    buffer = None
    try:
        _, _, group = init_dist(local_rank, NUM_RANKS)
        generator = torch.Generator(device='cuda')
        generator.manual_seed(20260730)

        def fixed_randn(
                shape: Tuple[int, ...], scale: float
        ) -> torch.Tensor:
            value = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device='cuda',
                generator=generator)
            return value.mul_(scale)

        x_bf16 = fixed_randn((NUM_TOKENS, HIDDEN), 0.5)
        l1_bf16 = fixed_randn(
            (NUM_EXPERTS, INTERMEDIATE * 2, HIDDEN), 0.05)
        l1_bf16[0].zero_()
        l2_bf16 = fixed_randn(
            (NUM_EXPERTS, HIDDEN, INTERMEDIATE), 0.05)
        topk_idx = (
            torch.arange(NUM_TOKENS, device='cuda', dtype=torch.long)
            % NUM_EXPERTS
        ).view(NUM_TOKENS, NUM_TOPK)
        topk_weights = torch.ones(
            (NUM_TOKENS, NUM_TOPK), device='cuda', dtype=torch.float)

        x_fp8, x_sf = per_token_cast_to_fp8(
            x_bf16,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True)
        transformed_l1, transformed_l2 = (
            deep_gemm.transform_weights_for_mega_moe(
                _cast_weights_to_fp4(l1_bf16),
                _cast_weights_to_fp4(l2_bf16),
                activation='situ'))
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            NUM_EXPERTS,
            NUM_TOKENS,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE,
            mma_type='fp8xfp4',
            activation='situ')

        def run_case(case: Case, fast_math: bool) -> torch.Tensor:
            buffer.buffer.zero_()
            buffer.x[:NUM_TOKENS].copy_(x_fp8)
            buffer.x_sf[:NUM_TOKENS].copy_(x_sf)
            buffer.topk_idx[:NUM_TOKENS].copy_(topk_idx)
            buffer.topk_weights[:NUM_TOKENS].copy_(topk_weights)

            y = torch.full(
                (NUM_TOKENS, HIDDEN),
                float('nan'),
                dtype=torch.bfloat16,
                device='cuda')
            kernel_kwargs = {
                'y': y,
                'l1_weights': transformed_l1,
                'l2_weights': transformed_l2,
                'sym_buffer': buffer,
                'activation': case.activation,
                'fast_math': fast_math,
            }
            if case.activation == 'situ':
                kernel_kwargs.update(
                    situ_beta=case.situ_beta,
                    situ_linear_beta=case.situ_linear_beta)
            deep_gemm.fp8_fp4_mega_moe(**kernel_kwargs)
            torch.cuda.synchronize()
            return y

        for fast_math in (True, False):
            results = {
                case_name: run_case(case, fast_math)
                for case_name, case in CASES.items()
            }
            _validate_suite(results, fast_math)
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def test_situ_metamorphic() -> None:
    if torch.cuda.device_count() < NUM_RANKS:
        raise unittest.SkipTest('requires one CUDA device')
    if torch.cuda.get_device_capability(0)[0] != 10:
        raise unittest.SkipTest('requires an SM100 CUDA device')
    torch.multiprocessing.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=NUM_RANKS,
        join=True)


if __name__ == '__main__':
    test_situ_metamorphic()
