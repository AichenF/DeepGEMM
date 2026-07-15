"""Eight-rank MiMo2.5-Pro shape gate for the SM90 NVFP4 MegaMoE kernel.

This is deliberately separate from the generic correctness and benchmark
harnesses.  It compares the standard NVFP4 decoder with the grouped-nibble
candidate at the exact MiMo2.5-Pro MoE shape and checks both against an
independent, dequantized PyTorch reference.

The default command is a release gate and therefore requires one node with
eight H200 GPUs.  No benchmark number is produced by this script.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_nvfp4 import (
    dequantize_nvfp4_to_fp32,
    nvfp4_group_nibbles_for_mega_moe_sm90,
    nvfp4_is_nibble_grouped_for_mega_moe_sm90,
    quantize_to_nvfp4,
)
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


HIDDEN = 6144
INTERMEDIATE_HIDDEN = 2048
NUM_EXPERTS = 384
NUM_TOPK = 8
NUM_RANKS = 8
NUM_LOCAL_EXPERTS = NUM_EXPERTS // NUM_RANKS
BLOCK_N = 256
GROUP_SIZE = 16
NIBBLE_POLICY_ENV = "DG_SM90_NVFP4_NIBBLE_GROUP"

PackedWeight = Tuple[torch.Tensor, torch.Tensor]


@contextlib.contextmanager
def _nibble_policy(enabled: bool):
    """Temporarily set decoder policy without leaking state to another case."""
    previous = os.environ.get(NIBBLE_POLICY_ENV)
    os.environ[NIBBLE_POLICY_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(NIBBLE_POLICY_ENV, None)
        else:
            os.environ[NIBBLE_POLICY_ENV] = previous


def _empty_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _all_gather_stack(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.stack(gathered, dim=0)


def _all_reduce_scalar(value: float, op: dist.ReduceOp, group: dist.ProcessGroup) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op, group=group)
    return float(tensor.item())


def _all_reduce_int(value: int, op: dist.ReduceOp, group: dist.ProcessGroup) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=op, group=group)
    return int(tensor.item())


def _make_routes(
    pattern: str,
    num_tokens: int,
    source_rank: int,
    seed: int,
    zipf_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create unique global expert IDs and positive, normalized route weights."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + source_rank * 10_007 + num_tokens * 101)
    indices = torch.empty((num_tokens, NUM_TOPK), dtype=torch.int64, device="cpu")

    if pattern == "uniform":
        # Every token sends one route to every rank.  The local expert varies by
        # token/source/seed, so this is balanced without being a repeated toy row.
        for token in range(num_tokens):
            for slot in range(NUM_TOPK):
                owner = (slot + source_rank + token) % NUM_RANKS
                local = (
                    seed * 13 + source_rank * 17 + token * 29 + owner * 7
                ) % NUM_LOCAL_EXPERTS
                indices[token, slot] = owner * NUM_LOCAL_EXPERTS + local
    elif pattern == "hot":
        # One fixed hot expert per rank.  This produces severe expert skew while
        # still exercising all seven remote peers for every source token.
        for token in range(num_tokens):
            for slot in range(NUM_TOPK):
                owner = (slot + source_rank) % NUM_RANKS
                local = (seed * 5 + owner * 11) % NUM_LOCAL_EXPERTS
                indices[token, slot] = owner * NUM_LOCAL_EXPERTS + local
    elif pattern == "zipf":
        popularity = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float64, device="cpu")
        popularity = popularity.pow(-zipf_alpha)
        for token in range(num_tokens):
            row = torch.multinomial(
                popularity,
                NUM_TOPK,
                replacement=False,
                generator=generator,
            )
            owners = torch.div(row, NUM_LOCAL_EXPERTS, rounding_mode="floor")
            if bool(torch.all(owners == source_rank)):
                # Zipf is intentionally skewed, but the gate must never degrade
                # into a local-only test for a source rank.
                remote_owner = (source_rank + 1) % NUM_RANKS
                candidate = remote_owner * NUM_LOCAL_EXPERTS + (
                    seed + source_rank * 3 + token * 7
                ) % NUM_LOCAL_EXPERTS
                while bool(torch.any(row == candidate)):
                    candidate = remote_owner * NUM_LOCAL_EXPERTS + (
                        (candidate + 1) % NUM_LOCAL_EXPERTS
                    )
                row[-1] = candidate
            indices[token] = row
    else:
        raise ValueError(f"unknown routing pattern: {pattern}")

    sorted_indices = indices.sort(dim=-1).values
    if num_tokens and bool(torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1])):
        raise AssertionError(f"{pattern}: duplicate expert in a top-{NUM_TOPK} row")
    if not bool(torch.all((indices >= 0) & (indices < NUM_EXPERTS))):
        raise AssertionError(f"{pattern}: expert index outside [0, {NUM_EXPERTS})")

    logits = torch.randn(
        (num_tokens, NUM_TOPK), generator=generator, dtype=torch.float32, device="cpu"
    ) * 0.6
    weights = torch.softmax(logits, dim=-1)
    return indices.to(device="cuda"), weights.to(device="cuda")


@dataclass
class GatheredCase:
    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    expected_local_stats: torch.Tensor
    remote_routes: int
    touched_experts: int
    touched_owner_ranks: int
    max_expert_load: int


def _gather_case(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    rank_idx: int,
    group: dist.ProcessGroup,
) -> GatheredCase:
    num_tokens = x_fp8.size(0)
    x_ref = (
        x_fp8.float().view(num_tokens, HIDDEN // 128, 128)
        * x_sf.float().unsqueeze(-1)
    ).view(num_tokens, HIDDEN)
    gathered_x = _all_gather_stack(x_ref, group)
    gathered_idx = _all_gather_stack(topk_idx, group)
    gathered_weights = _all_gather_stack(topk_weights, group)

    owners = torch.div(gathered_idx, NUM_LOCAL_EXPERTS, rounding_mode="floor")
    per_source_remote = (owners != torch.arange(
        NUM_RANKS, dtype=torch.int64, device="cuda"
    ).view(NUM_RANKS, 1, 1)).sum(dim=(1, 2))
    if bool(torch.any(per_source_remote == 0)):
        raise AssertionError("at least one source rank has no cross-rank route")

    local_begin = rank_idx * NUM_LOCAL_EXPERTS
    local_mask = (gathered_idx >= local_begin) & (
        gathered_idx < local_begin + NUM_LOCAL_EXPERTS
    )
    local_ids = (gathered_idx[local_mask] - local_begin).to(torch.int64)
    expected_stats = torch.bincount(local_ids, minlength=NUM_LOCAL_EXPERTS).to(torch.int32)
    expert_load = torch.bincount(gathered_idx.flatten(), minlength=NUM_EXPERTS)
    return GatheredCase(
        x=gathered_x,
        topk_idx=gathered_idx,
        topk_weights=gathered_weights,
        expected_local_stats=expected_stats,
        remote_routes=int(per_source_remote.sum().item()),
        touched_experts=int((expert_load != 0).sum().item()),
        touched_owner_ranks=int(torch.unique(owners).numel()),
        max_expert_load=int(expert_load.max().item()),
    )


class ReferenceWeights:
    """Exact NVFP4 values, materialized either fully or one expert at a time."""

    def __init__(
        self,
        l1_row: PackedWeight,
        l2_row: PackedWeight,
        mode: str,
    ) -> None:
        self.mode = mode
        self.l1_row: Optional[PackedWeight] = l1_row
        self.l2_row: Optional[PackedWeight] = l2_row
        self.l1_full: Optional[torch.Tensor] = None
        self.l2_full: Optional[torch.Tensor] = None
        if mode == "full":
            self.l1_full = dequantize_nvfp4_to_fp32(*l1_row, group_size=GROUP_SIZE)
            self.l2_full = dequantize_nvfp4_to_fp32(*l2_row, group_size=GROUP_SIZE)
            self.l1_row = None
            self.l2_row = None

    def expert(self, local_expert: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "full":
            assert self.l1_full is not None and self.l2_full is not None
            return self.l1_full[local_expert], self.l2_full[local_expert]
        assert self.l1_row is not None and self.l2_row is not None
        l1 = dequantize_nvfp4_to_fp32(
            self.l1_row[0][local_expert:local_expert + 1],
            self.l1_row[1][local_expert:local_expert + 1],
            group_size=GROUP_SIZE,
        )[0]
        l2 = dequantize_nvfp4_to_fp32(
            self.l2_row[0][local_expert:local_expert + 1],
            self.l2_row[1][local_expert:local_expert + 1],
            group_size=GROUP_SIZE,
        )[0]
        return l1, l2


@dataclass
class WeightBundle:
    standard_l1: PackedWeight
    standard_l2: PackedWeight
    grouped_l1: PackedWeight
    grouped_l2: PackedWeight
    reference: ReferenceWeights


def _random_bf16(shape: Iterable[int], scale: float) -> torch.Tensor:
    value = torch.empty(tuple(shape), dtype=torch.bfloat16, device="cuda")
    value.normal_()
    value.mul_(scale)
    return value


def _build_weights(
    seed: int,
    rank_idx: int,
    weight_scale: float,
    reference_mode: str,
) -> WeightBundle:
    torch.manual_seed(seed * 10_000_019 + rank_idx * 1_000_003)
    torch.cuda.reset_peak_memory_stats()

    l1_bf = _random_bf16(
        (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_HIDDEN, HIDDEN), weight_scale
    )
    l1_row = quantize_to_nvfp4(l1_bf, group_size=GROUP_SIZE)
    del l1_bf
    _empty_cuda_cache()

    l2_bf = _random_bf16(
        (NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE_HIDDEN), weight_scale
    )
    l2_row = quantize_to_nvfp4(l2_bf, group_size=GROUP_SIZE)
    del l2_bf
    _empty_cuda_cache()

    # The standard copy must remain ungrouped.  Runtime calls below also hold
    # this policy at zero so the compatibility fallback cannot mutate it.
    with _nibble_policy(False):
        standard_l1, standard_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
            l1_row,
            l2_row,
            block_n=BLOCK_N,
            group_size=GROUP_SIZE,
        )
    if nvfp4_is_nibble_grouped_for_mega_moe_sm90(standard_l1[0]):
        raise AssertionError("standard L1 unexpectedly uses grouped nibbles")
    if nvfp4_is_nibble_grouped_for_mega_moe_sm90(standard_l2[0]):
        raise AssertionError("standard L2 unexpectedly uses grouped nibbles")

    grouped_l1 = (standard_l1[0].clone(), standard_l1[1])
    grouped_l2 = (standard_l2[0].clone(), standard_l2[1])
    nvfp4_group_nibbles_for_mega_moe_sm90(grouped_l1[0])
    nvfp4_group_nibbles_for_mega_moe_sm90(grouped_l2[0])
    if not nvfp4_is_nibble_grouped_for_mega_moe_sm90(grouped_l1[0]):
        raise AssertionError("grouped L1 is missing its storage-layout marker")
    if not nvfp4_is_nibble_grouped_for_mega_moe_sm90(grouped_l2[0]):
        raise AssertionError("grouped L2 is missing its storage-layout marker")

    reference = ReferenceWeights(l1_row, l2_row, reference_mode)
    if reference_mode == "full":
        del l1_row, l2_row
        _empty_cuda_cache()
    torch.cuda.synchronize()
    return WeightBundle(
        standard_l1=standard_l1,
        standard_l2=standard_l2,
        grouped_l1=grouped_l1,
        grouped_l2=grouped_l2,
        reference=reference,
    )


def _run_kernel(
    buffer,
    l1_weights: PackedWeight,
    l2_weights: PackedWeight,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    args: argparse.Namespace,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x_fp8.size(0)
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    output = torch.full(
        (num_tokens, HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    stats = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
    dist.barrier(group=group)
    with _nibble_policy(False):
        deep_gemm.nvfp4_mega_moe(
            output,
            l1_weights,
            l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=stats,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return output, stats


def _reference_output(
    gathered: GatheredCase,
    reference: ReferenceWeights,
    rank_idx: int,
    activation_clamp: float,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    num_tokens = gathered.x.size(1)
    output = torch.zeros(
        (NUM_RANKS * num_tokens, HIDDEN), dtype=torch.float32, device="cuda"
    )
    local_begin = rank_idx * NUM_LOCAL_EXPERTS
    local_mask = (gathered.topk_idx >= local_begin) & (
        gathered.topk_idx < local_begin + NUM_LOCAL_EXPERTS
    )
    touched_local = torch.unique(
        gathered.topk_idx[local_mask] - local_begin
    ).cpu().tolist()

    for local_expert in touched_local:
        global_expert = local_begin + int(local_expert)
        positions = (gathered.topk_idx == global_expert).nonzero(as_tuple=False)
        source = positions[:, 0]
        token = positions[:, 1]
        slot = positions[:, 2]
        input_rows = gathered.x[source, token]
        route_weights = gathered.topk_weights[source, token, slot]
        l1_weight, l2_weight = reference.expert(int(local_expert))

        l1_out = input_rows @ l1_weight.t()
        gate, up = l1_out.split(INTERMEDIATE_HIDDEN, dim=-1)
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
        intermediate = gate * torch.sigmoid(gate) * up
        intermediate.mul_(route_weights.unsqueeze(-1))
        contribution = intermediate @ l2_weight.t()
        flat_rows = source * num_tokens + token
        output.index_add_(0, flat_rows, contribution)

        if reference.mode == "touched":
            del l1_weight, l2_weight

    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    if reference.mode == "touched":
        _empty_cuda_cache()
    return output.view(NUM_RANKS, num_tokens, HIDDEN)[rank_idx]


def _assert_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    group: dist.ProcessGroup,
) -> None:
    local_ok = int(torch.equal(actual, expected))
    global_ok = _all_reduce_int(local_ok, dist.ReduceOp.MIN, group)
    if not global_ok:
        local_max = int((actual.to(torch.int64) - expected.to(torch.int64)).abs().max().item())
        global_max = _all_reduce_int(local_max, dist.ReduceOp.MAX, group)
        raise AssertionError(f"{label}: receive statistics mismatch; global max delta={global_max}")


def _compare_outputs(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    group: dist.ProcessGroup,
) -> Tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    local_finite = int(bool(
        torch.isfinite(actual).all() and torch.isfinite(expected).all()
    ))
    finite = _all_reduce_int(local_finite, dist.ReduceOp.MIN, group)
    max_abs = _all_reduce_scalar(float(diff.max().item()), dist.ReduceOp.MAX, group)
    total_abs = _all_reduce_scalar(float(diff.sum().item()), dist.ReduceOp.SUM, group)
    total_values = _all_reduce_int(diff.numel(), dist.ReduceOp.SUM, group)
    mean_abs = total_abs / total_values
    if not finite:
        raise AssertionError(f"{label}: non-finite kernel output")
    if max_abs > atol:
        raise AssertionError(f"{label}: max_abs={max_abs:.6e} exceeds atol={atol:.6e}")
    return max_abs, mean_abs


def _check_reference_accuracy(
    kernel: torch.Tensor,
    reference: torch.Tensor,
    args: argparse.Namespace,
    group: dist.ProcessGroup,
) -> Dict[str, float]:
    kernel_fp32 = kernel.float()
    diff = (kernel_fp32 - reference).abs()
    cosine = torch.nn.functional.cosine_similarity(kernel_fp32, reference, dim=-1)
    local_finite = int(bool(
        torch.isfinite(kernel_fp32).all() and torch.isfinite(reference).all()
    ))
    finite = _all_reduce_int(local_finite, dist.ReduceOp.MIN, group)
    max_abs = _all_reduce_scalar(float(diff.max().item()), dist.ReduceOp.MAX, group)
    abs_sum = _all_reduce_scalar(float(diff.sum().item()), dist.ReduceOp.SUM, group)
    num_values = _all_reduce_int(diff.numel(), dist.ReduceOp.SUM, group)
    cosine_min = _all_reduce_scalar(float(cosine.min().item()), dist.ReduceOp.MIN, group)
    cosine_sum = _all_reduce_scalar(float(cosine.sum().item()), dist.ReduceOp.SUM, group)
    num_tokens = _all_reduce_int(cosine.numel(), dist.ReduceOp.SUM, group)
    kernel_norm_sq = _all_reduce_scalar(
        float(kernel_fp32.square().sum().item()), dist.ReduceOp.SUM, group
    )
    reference_norm_sq = _all_reduce_scalar(
        float(reference.square().sum().item()), dist.ReduceOp.SUM, group
    )
    cosine_mean = cosine_sum / num_tokens
    norm_ratio = (kernel_norm_sq / max(reference_norm_sq, 1e-30)) ** 0.5
    if not finite:
        raise AssertionError("kernel or independent reference contains non-finite values")
    if cosine_mean < args.cosine_mean_threshold:
        raise AssertionError(
            f"reference cosine mean {cosine_mean:.6f} < {args.cosine_mean_threshold:.6f}"
        )
    if cosine_min < args.cosine_min_threshold:
        raise AssertionError(
            f"reference cosine min {cosine_min:.6f} < {args.cosine_min_threshold:.6f}"
        )
    if not (args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max):
        raise AssertionError(
            f"reference norm ratio {norm_ratio:.6f} outside "
            f"[{args.norm_ratio_min:.6f}, {args.norm_ratio_max:.6f}]"
        )
    return {
        "max_abs": max_abs,
        "mean_abs": abs_sum / num_values,
        "cosine_min": cosine_min,
        "cosine_mean": cosine_mean,
        "norm_ratio": norm_ratio,
    }


def _run_case(
    args: argparse.Namespace,
    buffer,
    weights: WeightBundle,
    seed: int,
    pattern: str,
    num_tokens: int,
    rank_idx: int,
    group: dist.ProcessGroup,
) -> None:
    input_seed = seed * 2_000_033 + rank_idx * 20_011 + num_tokens * 211
    torch.manual_seed(input_seed)
    x_bf = _random_bf16((num_tokens, HIDDEN), 1.0)
    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    del x_bf
    topk_idx, topk_weights = _make_routes(
        pattern, num_tokens, rank_idx, seed, args.zipf_alpha
    )
    gathered = _gather_case(
        x_fp8, x_sf, topk_idx, topk_weights, rank_idx, group
    )
    if pattern in ("uniform", "hot") and gathered.touched_owner_ranks != NUM_RANKS:
        raise AssertionError(f"{pattern}: expected routes to all {NUM_RANKS} owner ranks")
    if pattern == "zipf" and gathered.touched_owner_ranks < 2:
        raise AssertionError("zipf: routes touched fewer than two owner ranks")

    first_outputs: Dict[str, torch.Tensor] = {}
    first_stats: Dict[str, torch.Tensor] = {}
    candidate_max_abs = 0.0
    candidate_mean_abs = 0.0
    for repeat in range(args.repeats):
        order = ("standard", "grouped") if repeat % 2 == 0 else ("grouped", "standard")
        outputs: Dict[str, torch.Tensor] = {}
        stats: Dict[str, torch.Tensor] = {}
        for path in order:
            l1 = weights.standard_l1 if path == "standard" else weights.grouped_l1
            l2 = weights.standard_l2 if path == "standard" else weights.grouped_l2
            outputs[path], stats[path] = _run_kernel(
                buffer,
                l1,
                l2,
                x_fp8,
                x_sf,
                topk_idx,
                topk_weights,
                args,
                group,
            )
            _assert_stats(path, stats[path], gathered.expected_local_stats, group)
            if repeat == 0:
                first_outputs[path] = outputs[path].clone()
                first_stats[path] = stats[path].clone()
            else:
                _compare_outputs(
                    f"{path} repeat stability",
                    outputs[path],
                    first_outputs[path],
                    args.stability_atol,
                    group,
                )
                _assert_stats(
                    f"{path} repeat stability",
                    stats[path],
                    first_stats[path],
                    group,
                )

        candidate_max_abs, candidate_mean_abs = _compare_outputs(
            "grouped candidate versus standard decoder",
            outputs["grouped"],
            outputs["standard"],
            args.candidate_atol,
            group,
        )

    reference = _reference_output(
        gathered,
        weights.reference,
        rank_idx,
        args.activation_clamp,
        group,
    )
    ref_metrics = _check_reference_accuracy(
        first_outputs["standard"], reference, args, group
    )
    if rank_idx == 0:
        print(
            f"PASS seed={seed} route={pattern:7s} M={num_tokens:3d} "
            f"remote_routes={gathered.remote_routes:4d} "
            f"touched_experts={gathered.touched_experts:3d} "
            f"max_expert_load={gathered.max_expert_load:3d} "
            f"candidate_max_abs={candidate_max_abs:.3e} "
            f"candidate_mean_abs={candidate_mean_abs:.3e} "
            f"ref_cos_min={ref_metrics['cosine_min']:.5f} "
            f"ref_cos_mean={ref_metrics['cosine_mean']:.5f} "
            f"ref_norm={ref_metrics['norm_ratio']:.5f} "
            f"ref_max_abs={ref_metrics['max_abs']:.3e}",
            flush=True,
        )


def _validate_runtime(args: argparse.Namespace, rank_idx: int, group: dist.ProcessGroup) -> None:
    world_size = dist.get_world_size(group)
    if world_size != NUM_RANKS:
        raise RuntimeError(f"MiMo gate requires exactly {NUM_RANKS} ranks, got {world_size}")
    if get_arch_major() != 9:
        raise RuntimeError(f"MiMo gate requires SM90, got SM{get_arch_major()}0")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if not args.allow_non_h200 and "H200" not in device_name.upper():
        raise RuntimeError(
            f"release gate requires H200, rank {rank_idx} has {device_name!r}; "
            "use --allow-non-h200 only for development"
        )
    if max(args.batches) > 4096:
        raise ValueError("num_max_tokens_per_rank must not exceed 4096")
    if os.environ.get("DG_SM90_MOE_PHASE_PROFILE", "0") != "0":
        raise RuntimeError(
            "unset DG_SM90_MOE_PHASE_PROFILE for this gate; the exact 48-entry "
            "receive-statistics check is not a profiling buffer"
        )
    if rank_idx == 0:
        print(
            "MiMo2.5-Pro exact-shape gate: "
            f"H={HIDDEN}, I={INTERMEDIATE_HIDDEN}, E={NUM_EXPERTS}, "
            f"topk={NUM_TOPK}, ranks={NUM_RANKS}, local_experts={NUM_LOCAL_EXPERTS}, "
            f"block_n={BLOCK_N}, device={device_name}",
            flush=True,
        )


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    buffer = None
    try:
        _validate_runtime(args, rank_idx, group)
        num_max_tokens = max(32, max(args.batches))
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            NUM_EXPERTS,
            num_max_tokens,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE_HIDDEN,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        total_cases = len(args.seeds) * len(args.route_patterns) * len(args.batches)
        completed = 0
        for seed in args.seeds:
            if rank_idx == 0:
                print(f"Building exact-shape NVFP4 weights for seed={seed} ...", flush=True)
            weights = _build_weights(
                seed,
                rank_idx,
                args.weight_scale,
                args.reference_mode,
            )
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            global_peak_gib = _all_reduce_scalar(peak_gib, dist.ReduceOp.MAX, group)
            if rank_idx == 0:
                print(
                    f"Weight preparation complete; max rank peak allocation={global_peak_gib:.2f} GiB",
                    flush=True,
                )
            for pattern in args.route_patterns:
                for num_tokens in args.batches:
                    _run_case(
                        args,
                        buffer,
                        weights,
                        seed,
                        pattern,
                        num_tokens,
                        rank_idx,
                        group,
                    )
                    completed += 1
            dist.barrier(group=group)
            del weights
            _empty_cuda_cache()
        if rank_idx == 0:
            print(
                f"MiMo2.5-Pro 8-rank NVFP4 gate: PASS ({completed}/{total_cases} cases)",
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-shape eight-H200 MiMo2.5-Pro NVFP4 MegaMoE gate"
    )
    parser.add_argument("--num-processes", type=int, default=NUM_RANKS)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 24])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260715, 20260716])
    parser.add_argument(
        "--route-patterns",
        nargs="+",
        choices=("uniform", "hot", "zipf"),
        default=["uniform", "hot", "zipf"],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reference-mode", choices=("touched", "full"), default="touched")
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--candidate-atol", type=float, default=0.0)
    parser.add_argument("--stability-atol", type=float, default=0.0)
    parser.add_argument("--cosine-mean-threshold", type=float, default=0.90)
    parser.add_argument("--cosine-min-threshold", type=float, default=0.90)
    parser.add_argument("--norm-ratio-min", type=float, default=0.50)
    parser.add_argument("--norm-ratio-max", type=float, default=2.00)
    parser.add_argument(
        "--allow-non-h200",
        action="store_true",
        help="Development only: allow another SM90 GPU; release evidence must omit this flag.",
    )
    args = parser.parse_args()
    if args.num_processes != NUM_RANKS:
        parser.error(f"--num-processes must be exactly {NUM_RANKS}")
    if not args.batches or any(value <= 0 for value in args.batches):
        parser.error("--batches must contain positive token counts")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.candidate_atol < 0 or args.stability_atol < 0:
        parser.error("output tolerances must be non-negative")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(parsed_args.num_processes, parsed_args),
        nprocs=parsed_args.num_processes,
        join=True,
    )
