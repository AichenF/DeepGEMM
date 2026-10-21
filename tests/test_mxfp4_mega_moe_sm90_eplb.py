"""Dynamic EPLB gate for the SM90 MXFP4 MegaMoE kernel.

Three checks per batch, all under deliberately skewed routing:
  1. the device planner agrees with the Python reference planner;
  2. the remapped routing table matches the plan (counts per physical slot);
  3. the kernel output on physical slots matches the logical-expert reference;
and the same plan and remap through the single-launch kernel (warm and cold
slot tables). Then a paired benchmark of the same routing with EPLB on and
off. ``DG_EPLB_PROFILE=1`` adds the planner's in-kernel phase breakdown.

Usage: python tests/test_mxfp4_mega_moe_sm90_eplb.py --num-processes 4
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm import _C
from deep_gemm.mega.eplb_planner import plan_reference
from deep_gemm.quantization_mxfp4 import dequantize_mxfp4_to_fp32, quantize_to_mxfp4
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


def _silu(x):
    return x * torch.sigmoid(x)


# Planner phase stamps and loop counters, in the order of `eplb::PlanProfile`
# (`deep_gemm/include/deep_gemm/layout/mega_moe_eplb.cuh`). A phase is named
# after the stamp that ends it.
PLAN_STAMPS = ('entry', 'histogram', 'arrived', 'published', 'wait', 'setup',
               'greedy', 'prune', 'slots', 'tables', 'remap', 'end')
PLAN_COUNTERS = ('visits', 'picks', 'moves', 'prune_steps', 'extreme_moves', 'grid')


def plan_profile_enabled():
    return os.environ.get('DG_EPLB_PROFILE', '0') == '1'


def read_plan_profile(eplb, generation=None):
    """Decode the planner's timing record of ``generation`` (default: the
    last plan) from the workspace: phase durations in microseconds (SM clock
    cycles converted with the globaltimer span of the same launch), the
    globaltimer stamps in ns and the loop counters. Needs ``DG_EPLB_PROFILE=1``
    in the environment when the kernel was built, and a ``_C`` that reports
    the ``profile`` region; returns None otherwise."""
    layout = eplb.ws_layout
    if 'profile' not in layout:
        return None
    gen = eplb.generation if generation is None else generation
    slots, words = int(layout['num_profile_slots']), int(layout['num_profile_words'])
    base = int(layout['profile']) + (gen % slots) * words
    rec = eplb.ws.view(torch.int32)[base:base + words].cpu().contiguous()
    u64 = rec.view(torch.int64)
    n = len(PLAN_STAMPS)
    clocks = u64[:n].tolist()
    t_entry, t_peers, t_end = u64[n].item(), u64[n + 1].item(), u64[n + 2].item()
    counters = rec[2 * n + 6:2 * n + 6 + len(PLAN_COUNTERS)].tolist()
    span_cycles, span_ns = clocks[-1] - clocks[0], t_end - t_entry
    if span_cycles <= 0 or span_ns <= 0:
        return None
    ghz = span_cycles / span_ns
    phases = {PLAN_STAMPS[i]: (clocks[i] - clocks[i - 1]) / ghz / 1e3 for i in range(1, n)}
    return dict(phases=phases, total_us=span_cycles / ghz / 1e3, ghz=ghz,
                timer_entry_ns=t_entry, timer_peers_ns=t_peers, timer_end_ns=t_end,
                counters=dict(zip(PLAN_COUNTERS, counters)))


def format_plan_profile(prof):
    ph = prof['phases']
    body = ' + '.join(f'{k} {ph[k]:.1f}' for k in PLAN_STAMPS[1:] if ph[k] >= 0.05)
    c = prof['counters']
    return (f'total {prof["total_us"]:.1f} us = {body} | compute (total - wait) '
            f'{prof["total_us"] - ph["wait"]:.1f} us | picks={c["picks"]} moves={c["moves"]} '
            f'prune={c["prune_steps"]} extreme={c["extreme_moves"]} grid={c["grid"]} @ {prof["ghz"]:.2f} GHz')


def _single_launch(eplb, num_tokens, block_m):
    """Plan + remap in one grid, driven through ``_C`` directly (what
    ``DynamicEPLB.prepare`` launches)."""
    eplb.generation += 1
    _C.mega_moe_eplb_plan_and_remap(
        eplb.ws, eplb.ws_handle.buffer_ptrs, eplb.rank,
        eplb.symm.topk_idx[:num_tokens],
        eplb.num_experts, eplb.num_helper_slots,
        block_m, eplb.open_gate_blocks, eplb.generation)


def _skewed_topk(m_tokens, num_experts, num_topk, hot_experts, hot_share, generator):
    """Top-k routing where a share of all slots lands on a few hot experts."""
    scores = torch.rand((m_tokens, num_experts), generator=generator, device='cuda')
    hot = torch.tensor(hot_experts, device='cuda')
    mask = torch.rand((m_tokens, len(hot_experts)), generator=generator, device='cuda') < hot_share
    boost = torch.zeros_like(scores)
    boost[:, hot] = mask.float() * 10.0
    topk_weights, topk_idx = torch.topk(scores + boost, num_topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    return topk_idx, topk_weights


def _load_trace(path, num_experts, num_topk):
    """Routing trace captured from a real model: a dict with ``layers``, a list
    of int tensors ``[tokens, topk]`` in logical expert ids (one global batch
    per entry), plus ``num_experts`` and ``num_topk`` for a shape check."""
    trace = torch.load(path, map_location='cpu')
    assert int(trace['num_experts']) == num_experts and int(trace['num_topk']) == num_topk, \
        f"trace is for {trace['num_experts']} experts / top-{trace['num_topk']}"
    layers = [t.to(torch.int64) for t in trace['layers']]
    return layers


def _trace_topk(layers, layer_idx, m_tokens, rank_idx, num_ranks, generator):
    """Rank ``rank_idx``'s slice of one traced layer: ``m_tokens`` consecutive
    tokens out of the global batch, wrapping around if the capture is short.
    Routing weights are not in the trace; softmax over random logits stands in."""
    rows = layers[layer_idx % len(layers)].cuda()
    total = rows.shape[0]
    start = (rank_idx * m_tokens) % total
    idx = (torch.arange(m_tokens, device=rows.device) + start) % total
    topk_idx = rows[idx]
    w = torch.rand((m_tokens, topk_idx.shape[1]), generator=generator, device='cuda')
    return topk_idx, torch.softmax(w, dim=-1)


def _all_gather_counts(topk_idx, num_experts, group):
    counts = torch.bincount(topk_idx.flatten(), minlength=num_experts).to(torch.int64)
    gathered = [torch.empty_like(counts) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, counts, group=group)
    return [g.cpu().tolist() for g in gathered]


def _reference_output(x_fp8, x_sf, topk_idx, topk_weights, l1_dequant, l2_dequant,
                      l1_gs, l2_gs, rank_idx, num_local_experts, clamp, group):
    """Logical-expert reference for every rank's tokens.

    Each rank applies its own experts to all ranks' tokens, then the partial
    sums are reduced so that rank r receives the full result for its tokens.
    Ranks may hold different tokens (real routing traces do)."""
    num_ranks = dist.get_world_size(group)
    m_tokens, hidden = x_fp8.shape
    x_ref = (x_fp8.float().view(m_tokens, hidden // 128, 128) * x_sf.float().unsqueeze(2)).view(m_tokens, hidden)
    all_x = [torch.empty_like(x_ref) for _ in range(num_ranks)]
    all_idx = [torch.empty_like(topk_idx) for _ in range(num_ranks)]
    all_w = [torch.empty_like(topk_weights) for _ in range(num_ranks)]
    dist.all_gather(all_x, x_ref.contiguous(), group=group)
    dist.all_gather(all_idx, topk_idx.contiguous(), group=group)
    dist.all_gather(all_w, topk_weights.contiguous().float(), group=group)
    y_all = torch.zeros((num_ranks, m_tokens, hidden), device='cuda', dtype=torch.float32)
    for src in range(num_ranks):
        idx_cpu = all_idx[src].cpu()
        w_cpu = all_w[src].cpu()
        for local in range(num_local_experts):
            expert = rank_idx * num_local_experts + local
            hits = (idx_cpu == expert).nonzero()
            if hits.numel() == 0:
                continue
            tok = hits[:, 0].cuda()
            w = w_cpu[hits[:, 0], hits[:, 1]].cuda().float()
            l1_out = all_x[src][tok] @ l1_dequant[local].float().t() * l1_gs[local]
            inter = l1_out.shape[1] // 2
            gate = l1_out[:, :inter].clamp(max=clamp)
            up = l1_out[:, inter:].clamp(min=-clamp, max=clamp)
            h = _silu(gate) * up * w[:, None]
            y_all[src].index_add_(0, tok, (h @ l2_dequant[local].float().t()) * l2_gs[local])
    dist.all_reduce(y_all, op=dist.ReduceOp.SUM, group=group)
    return y_all[rank_idx]


def _check_plan(eplb, counts, block_m, prev_slot_expert, rank_idx):
    ref = plan_reference(counts, eplb.num_helper_slots, block_m,
                         prev_slot_expert=prev_slot_expert, open_gate_blocks=eplb.open_gate_blocks)
    dev_slots = eplb.slot_expert.cpu().tolist()
    assert dev_slots == ref.slot_expert, f'rank {rank_idx}: slot table differs\n dev={dev_slots}\n ref={ref.slot_expert}'
    dev_copies = eplb.copies.cpu().tolist()
    for e in range(eplb.num_experts):
        got = []
        for rank, cum in dev_copies[e]:
            if rank < 0:
                break
            got.append((rank, cum))
        assert got == ref.copies[e], f'rank {rank_idx}: copies for expert {e} differ: dev={got} ref={ref.copies[e]}'
    return ref


def _check_remap(eplb, ref, counts, logical_rows, physical_rows, rank_idx):
    """Every physical slot refers to a slot that holds the expert, and the
    per-slot counts equal the reference remap exactly."""
    slots = eplb.slots_per_rank
    H = eplb.home_experts
    seen = {}
    for row_l, row_p in zip(logical_rows, physical_rows):
        for e, p in zip(row_l, row_p):
            dest, local = divmod(p, slots)
            if local < H:
                assert dest * H + local == e, f'rank {rank_idx}: home slot mismatch {p} for expert {e}'
            else:
                assert ref.slot_expert[dest][local - H] == e, \
                    f'rank {rank_idx}: helper slot {p} does not hold expert {e}'
            seen[(dest, e)] = seen.get((dest, e), 0) + 1
    return seen


def _check_slot_counts(eplb, ref, seen, block_m, rank_idx, group):
    """Per-(rank, expert) token counts across all ranks must fill exactly the
    plan's blocks."""
    num_ranks, E = dist.get_world_size(group), eplb.num_experts
    local_counts = torch.zeros((num_ranks, E), dtype=torch.int64, device='cuda')
    for (dest, e), n in seen.items():
        local_counts[dest, e] += n
    dist.all_reduce(local_counts, group=group)
    local_counts = local_counts.cpu().tolist()
    for e in range(E):
        for r in range(num_ranks):
            n, blocks = local_counts[r][e], ref.blocks[e][r]
            assert n <= blocks * block_m and (blocks == 0) == (n == 0) and n > (blocks - 1) * block_m, \
                f'rank {rank_idx}: expert {e} on rank {r} got {n} tokens for {blocks} blocks of {block_m}'


def _check_single_launch(m_tokens, rank_idx, group, eplb, counts, topk_idx, block_m, cold):
    """Plan + remap in one launch on the same routing: the tables must match
    the reference (warm: from the current slot table; cold: from empty
    slots), the rewritten routing must be a valid remap of that plan, the
    ordinal counters must be left at zero, and the separate copy kernel that
    follows must leave every helper slot byte-exact."""
    b = eplb.symm
    if cold:
        eplb.slot_expert.fill_(-1)
    prev = eplb.slot_expert.cpu().tolist()
    b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
    _single_launch(eplb, m_tokens, block_m)
    torch.cuda.synchronize()
    ref = _check_plan(eplb, counts, block_m, prev, rank_idx)
    physical = b.topk_idx[:m_tokens].cpu().tolist()
    seen = _check_remap(eplb, ref, counts, topk_idx.cpu().tolist(), physical, rank_idx)
    _check_slot_counts(eplb, ref, seen, block_m, rank_idx, group)
    assert not eplb.local_counter.cpu().any().item(), f'rank {rank_idx}: ordinal counters not cleared'
    copies = int(eplb.copy_count.item())
    eplb.copy_weights()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    _verify_copies(eplb, ref, rank_idx, group)
    # The list is applied; a fused-copy kernel launch later must not redo it.
    eplb.copy_count.zero_()
    prof = read_plan_profile(eplb) if plan_profile_enabled() else None
    if rank_idx == 0:
        replicas = sum(1 for row in ref.slot_expert for e in row if e >= 0)
        print(f'  [single-launch {"cold" if cold else "warm"}] M={m_tokens} plan+remap match the reference: '
              f'replicas={replicas} copies(rank0)={copies} max_load={max(ref.loads)}'
              + (f' | {format_plan_profile(prof)}' if prof else ''), flush=True)


def _verify_copies(eplb, ref, rank_idx, group):
    """Every helper slot must hold a byte-exact copy of its expert's home slot."""
    num_ranks = dist.get_world_size(group)
    H, S = eplb.home_experts, eplb.num_helper_slots
    sums = []
    for name in eplb._plane_names:
        v = eplb.arena.views[name]
        flat = v.reshape(v.shape[0], -1)
        if flat.dtype == torch.float32:
            flat = flat.view(torch.int32)
        sums.append(flat.to(torch.int64).sum(dim=1))
    local = torch.stack(sums)  # [planes, slots]
    gathered = [torch.empty_like(local) for _ in range(num_ranks)]
    dist.all_gather(gathered, local, group=group)
    bad = []
    for r in range(num_ranks):
        for s_idx, e in enumerate(ref.slot_expert[r]):
            if e < 0:
                continue
            owner, local_e = divmod(e, H)
            if not torch.equal(gathered[r][:, H + s_idx], gathered[owner][:, local_e]):
                bad.append((r, s_idx, e, owner))
    assert not bad, f'rank {rank_idx}: helper slots differ from their owners (rank, slot, expert, owner): {bad[:8]}'


def _run_case(args, m_tokens, rank_idx, group, eplb_state, check=True):
    num_ranks = dist.get_world_size(group)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(args.seed + m_tokens)
    hidden, ih = args.hidden, args.intermediate_hidden
    E, K, S = args.num_experts, args.num_topk, args.num_helper_slots
    H = E // num_ranks

    eplb, l1_dequant, l2_dequant, l1_gs, l2_gs = eplb_state[:5]
    if args.trace_layers is not None:
        topk_idx, topk_weights = _trace_topk(args.trace_layers, args.trace_layer, m_tokens, rank_idx, num_ranks, gen)
    else:
        hot = [h % E for h in args.hot_experts]
        topk_idx, topk_weights = _skewed_topk(m_tokens, E, K, hot, args.hot_share, gen)
    x_bf = torch.randn((m_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)

    if not check:
        return topk_idx, topk_weights, x_fp8, x_sf
    counts = _all_gather_counts(topk_idx, E, group)
    prev_slot_expert = eplb.slot_expert.cpu().tolist()

    # Stage by stage so each one can be checked.
    b = eplb.symm
    b.x[:m_tokens].copy_(x_fp8)
    b.x_sf[:m_tokens].copy_(x_sf)
    b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
    b.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))
    block_m = eplb.plan(m_tokens)
    torch.cuda.synchronize()
    ref = _check_plan(eplb, counts, block_m, prev_slot_expert, rank_idx)
    copies = int(eplb.copy_count.item())
    if not eplb.fuse_copy:
        eplb.copy_weights()
    eplb.remap(m_tokens, block_m)
    torch.cuda.synchronize()
    physical = b.topk_idx[:m_tokens].cpu().tolist()
    seen = _check_remap(eplb, ref, counts, topk_idx.cpu().tolist(), physical, rank_idx)
    _check_slot_counts(eplb, ref, seen, block_m, rank_idx, group)

    if not eplb.fuse_copy:
        _verify_copies(eplb, ref, rank_idx, group)

    y = torch.empty((m_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.mxfp4_mega_moe(
        y, eplb.l1_weights, eplb.l2_weights, b,
        cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
        l1_global_scales=eplb.l1_global_scales, l2_global_scales=eplb.l2_global_scales,
        activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math),
        eplb_fused_copy=eplb.fused_copy_args() if eplb.fuse_copy else None)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if eplb.fuse_copy:
        # The kernel did the copy: every listed slot must now be byte-exact
        # and every counter must have seen the whole grid.
        _verify_copies(eplb, ref, rank_idx, group)
        done = eplb.copy_done.cpu()
        listed = eplb.copy_list.cpu()[:copies, 0].tolist()
        for s_idx in listed:
            assert done[s_idx, 0].item() == done[s_idx, 1].item() > 0, \
                f'rank {rank_idx}: slot {s_idx} readiness counters {done[s_idx].tolist()}'

    y_ref = _reference_output(x_fp8, x_sf, topk_idx, topk_weights, l1_dequant, l2_dequant,
                              l1_gs, l2_gs, rank_idx, H, args.activation_clamp, group)
    finite = torch.isfinite(y).all().item()
    cosine = torch.nn.functional.cosine_similarity(y.float(), y_ref, dim=-1)
    if args.check_fixed:
        # The same tokens through the fixed-placement kernel, as a control.
        fb, fw = eplb_state[5], eplb_state[6]
        fb.x[:m_tokens].copy_(x_fp8)
        fb.x_sf[:m_tokens].copy_(x_sf)
        fb.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
        fb.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))
        y_fixed = torch.empty((m_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.mxfp4_mega_moe(y_fixed, fw[0], fw[1], fb,
                                 l1_global_scales=torch.tensor(l1_gs, device='cuda'),
                                 l2_global_scales=torch.tensor(l2_gs, device='cuda'),
                                 activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
        torch.cuda.synchronize()
        cos_fixed = torch.nn.functional.cosine_similarity(y_fixed.float(), y_ref, dim=-1)
        slots = eplb.slots_per_rank
        phys = b.topk_idx[:m_tokens]
        on_helper = ((phys % slots) >= H).any(dim=1)
        if rank_idx == 0:
            print(f'  [control] fixed-placement cos mean={cos_fixed.mean().item():.4f} min={cos_fixed.min().item():.4f}; '
                  f'eplb cos<0.9 on {(cosine < 0.9).float().mean().item():.3f} of tokens '
                  f'(tokens touching a helper slot: {on_helper.float().mean().item():.3f}, '
                  f'their cos mean={cosine[on_helper].mean().item() if on_helper.any() else float("nan"):.4f}, '
                  f'others cos mean={cosine[~on_helper].mean().item() if (~on_helper).any() else float("nan"):.4f})', flush=True)
    cos_min, cos_mean = cosine.min().item(), cosine.mean().item()
    norm_ratio = (torch.linalg.vector_norm(y.float()) / torch.linalg.vector_norm(y_ref).clamp_min(1e-30)).item()
    if os.environ.get('DG_EPLB_GATE_DIAG', '0') == '1':
        # Which tokens deviate, through which slots: every rank prints its
        # own worst tokens (logical experts, physical slots, weights) and the
        # slots shared by the deviating tokens, so a wrong slot or a wrong
        # token can be told apart from a wrong scale.
        tok_ratio = (torch.linalg.vector_norm(y.float(), dim=-1) /
                     torch.linalg.vector_norm(y_ref, dim=-1).clamp_min(1e-30))
        bad = ((tok_ratio - 1).abs() > 0.02) | (cosine < 0.995)
        n_bad = int(bad.sum().item())
        phys = b.topk_idx[:m_tokens].cpu()
        logical = topk_idx.cpu()
        weights = topk_weights.float().cpu()
        slots = eplb.slots_per_rank
        lines = [f'  [diag rank {rank_idx}] M={m_tokens} bad tokens {n_bad}/{m_tokens} '
                 f'norm ratio {norm_ratio:.4f} cos min {cos_min:.4f}']
        from collections import Counter
        slot_hits = Counter()
        for t in bad.nonzero().flatten().tolist():
            for pid in phys[t].tolist():
                slot_hits[(pid // slots, pid % slots)] += 1
        for t in bad.nonzero().flatten().tolist()[:6]:
            ents = [f'e{int(e)}->r{int(p) // slots}s{int(p) % slots}:{w:.2f}'
                    for e, p, w in zip(logical[t].tolist(), phys[t].tolist(), weights[t].tolist())]
            lines.append(f'    token {t}: ratio {tok_ratio[t].item():.4f} cos {cosine[t].item():.4f} ' + ' '.join(ents))
        helper_hits = {k: v for k, v in slot_hits.items() if k[1] >= H}
        lines.append(f'    helper slots used by bad tokens (rank, slot): hits {sorted(helper_hits.items())}')
        lines.append(f'    slot table: {eplb.slot_expert.cpu().tolist()}')
        lines.append(f'    this rank refilled (slot, expert) this step: {eplb.copy_list.cpu().tolist()[:copies]}')
        print('\n'.join(lines), flush=True)
    baseline = max(sum(-(-sum(counts[r][e] for r in range(num_ranks)) // block_m)
                       for e in range(rr * H, (rr + 1) * H)) for rr in range(num_ranks))
    if rank_idx == 0:
        replicas = sum(1 for row in ref.slot_expert for e in row if e >= 0)
        print(f'M={m_tokens} block_m={block_m} replicas={replicas} copies(rank0)={copies} '
              f'max_load {baseline} -> {max(ref.loads)} blocks | '
              f'cos min={cos_min:.4f} mean={cos_mean:.4f} norm={norm_ratio:.4f} finite={finite}', flush=True)
    assert finite, 'non-finite output'
    assert cos_mean >= args.cosine_mean_threshold, f'cosine mean {cos_mean}'
    assert cos_min >= args.cosine_min_threshold, f'cosine min {cos_min}'
    assert args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max, f'norm ratio {norm_ratio}'
    if args.single_launch:
        # The same routing through the single-launch prepare, warm (the
        # placement just planned) and cold (every slot re-planned).
        for cold in (False, True):
            _check_single_launch(m_tokens, rank_idx, group, eplb, counts, topk_idx, block_m, cold)
    return topk_idx, topk_weights, x_fp8, x_sf


def _bench(args, m_tokens, rank_idx, group, eplb, routing, baseline_buffer, baseline_weights):
    topk_idx, topk_weights, x_fp8, x_sf = routing
    hidden = args.hidden
    cold_each_step = bool(args.cold_each_step)

    def run_eplb():
        if cold_each_step:
            # Forget the slot cache so the planner lists every assigned helper
            # slot again and the full weight transfer is paid in this step.
            eplb.slot_expert.fill_(-1)
        return eplb.forward(x_fp8, x_sf, topk_idx, topk_weights,
                            activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))

    def run_fixed():
        b = baseline_buffer
        n = m_tokens
        b.x[:n].copy_(x_fp8)
        b.x_sf[:n].copy_(x_sf)
        b.topk_idx[:n].copy_(topk_idx.to(torch.int64))
        b.topk_weights[:n].copy_(topk_weights.to(torch.float32))
        y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.mxfp4_mega_moe(y, baseline_weights[0], baseline_weights[1], b,
                                 activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
        return y

    torch.cuda.synchronize()
    dist.barrier(group=group)
    # Cold copy: every helper slot changes, so this is the full weight transfer
    # the steady-state loop below never pays again. The first pass only builds
    # the kernel; the second one is timed.
    b = eplb.symm
    for _ in range(2):
        eplb.slot_expert.fill_(-1)
        b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
        eplb.plan(m_tokens)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        copies = int(eplb.copy_count.item())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        eplb.copy_weights()
        end.record()
        torch.cuda.synchronize()
    cold_us = torch.tensor([start.elapsed_time(end) * 1e3], device='cuda')
    dist.all_reduce(cold_us, op=dist.ReduceOp.MAX, group=group)
    bytes_per_slot = sum(spec[1] for spec in eplb.arena.plane_specs(eplb._plane_names))
    if rank_idx == 0:
        print(f'[cold copy M={m_tokens}] {copies} slots x {bytes_per_slot / 2**20:.1f} MiB on rank 0, '
              f'rank-max {cold_us.item():.1f}us', flush=True)

    kernels = ('eplb_plan_kernel', 'eplb_copy_weights_kernel', 'eplb_remap_kernel', 'sm90_mxfp4_mega_moe')
    arms = {'both': (False, True), 'fused': (True,), 'separate': (False,)}[args.fuse_copy]
    # The fixed arm is a separate kernel instantiation. Build it on every rank
    # before the paired loop: the planner waits on its peers in-kernel and
    # would otherwise time out behind another rank's JIT compile.
    run_fixed()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for fused in arms:
        eplb.fuse_copy = fused
        for _ in range(3):
            run_eplb()
            run_fixed()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    # Every rank must issue the same number of plans, so the debug profile runs
    # everywhere and only rank 0 prints it.
    debug = bool(os.environ.get('DG_EPLB_DEBUG'))
    quiet = not (debug and rank_idx == 0)
    if debug:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            run_eplb()
            torch.cuda.synchronize()
        if rank_idx == 0:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=120), flush=True)
    # The planner and the kernel both wait on their peers in-kernel with a
    # 30 s budget; a profiler session can leave the hosts skewed by more than
    # that, so resynchronize before every segment that launches them.
    def resync():
        torch.cuda.synchronize()
        dist.barrier(group=group)

    resync()
    t_fixed = bench_kineto(run_fixed, 'sm90_mxfp4_mega_moe', barrier=dist.barrier, num_tests=args.bench_tests,
                           suppress_kineto_output=quiet)
    resync()
    t_fixed_us = t_fixed * 1e6
    state = 'cold-each-step' if cold_each_step else 'steady'
    # Interleave the arms `bench_rounds` times: the kernel time of one arm
    # can move between sub-runs by more than the arm difference.
    for fused in [f for _ in range(max(1, args.bench_rounds)) for f in arms]:
        eplb.fuse_copy = fused
        # bench_kineto sums every kernel whose name matches, so an arm that
        # never launches the copy kernel asks only for the other three.
        names = kernels if not fused else tuple(k for k in kernels if k != 'eplb_copy_weights_kernel')
        # How many slots a step of this arm copies: plan the same routing from
        # the state the timed steps see (the fused kernel clears the count
        # once it has consumed the list, so read it after a standalone plan).
        if cold_each_step:
            eplb.slot_expert.fill_(-1)
        b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
        eplb.plan(m_tokens)
        torch.cuda.synchronize()
        copies = int(eplb.copy_count.item())
        run_eplb()
        resync()
        t = bench_kineto(run_eplb, names, barrier=dist.barrier, num_tests=args.bench_tests,
                         suppress_kineto_output=quiet)
        resync()
        t = [v * 1e6 for v in t]
        if fused:
            plan_us, remap_us, kernel_only = t
            copy_us = 0.0
        else:
            plan_us, copy_us, remap_us, kernel_only = t
        total = plan_us + copy_us + remap_us + kernel_only
        # Rank-maximum latency is what the layer waits for.
        local = torch.tensor([total, kernel_only, t_fixed_us, plan_us, copy_us, remap_us, float(copies)], device='cuda')
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
        total, kernel_only, fixed, plan_us, copy_us, remap_us, max_copies = local.tolist()
        if rank_idx == 0:
            ratio = lambda a, b: f'{a / b:.3f}x' if b > 0 else 'n/a'
            arm = 'fused-copy' if fused else 'separate-copy'
            print(f'[bench M={m_tokens} {state} {arm} slots-copied-max={int(max_copies)}] fixed={fixed:.1f}us  '
                  f'eplb total={total:.1f}us (kernel {kernel_only:.1f} + plan {plan_us:.1f} + copy {copy_us:.1f} '
                  f'+ remap {remap_us:.1f})  speedup total={ratio(fixed, total)} kernel-only={ratio(fixed, kernel_only)}',
                  flush=True)
        if plan_profile_enabled():
            # In-kernel phase breakdown of the last plan of this arm.
            run_eplb()
            torch.cuda.synchronize()
            prof = read_plan_profile(eplb)
            if rank_idx == 0 and prof:
                print(f'[plan profile M={m_tokens} {arm}] {format_plan_profile(prof)}', flush=True)


def _check_remap_only(args, m_tokens, rank_idx, group, eplb_state, tag):
    """Remap onto the current placement without planning (what decode steps
    do): every token must land on a slot that holds its expert and the output
    must match the reference."""
    num_ranks = dist.get_world_size(group)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(args.seed + 7919 * (m_tokens + 1) + rank_idx)
    E, K, S = args.num_experts, args.num_topk, args.num_helper_slots
    H = E // num_ranks
    eplb, l1_dequant, l2_dequant, l1_gs, l2_gs = eplb_state[:5]
    hot = [h % E for h in args.hot_experts]
    topk_idx, topk_weights = _skewed_topk(m_tokens, E, K, hot, args.hot_share, gen)
    x_bf = torch.randn((m_tokens, args.hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)
    b = eplb.symm
    b.x[:m_tokens].copy_(x_fp8)
    b.x_sf[:m_tokens].copy_(x_sf)
    b.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))
    slot_expert = eplb.slot_expert.cpu().tolist()
    # Twice: the ordinal counters must be reset each time. Remap rewrites the
    # routing in place (logical -> physical), so reload the logical ids first.
    for _ in range(2):
        b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
        eplb.remap(m_tokens, eplb.block_m(m_tokens))
    torch.cuda.synchronize()
    physical = b.topk_idx[:m_tokens].cpu().tolist()
    logical = topk_idx.cpu().tolist()
    for row_l, row_p in zip(logical, physical):
        for e, slot in zip(row_l, row_p):
            dest, local = divmod(slot, H + S)
            held = (dest * H + local) if local < H else slot_expert[dest][local - H]
            assert held == e, f'rank {rank_idx} ({tag}): expert {e} mapped to slot {slot} which holds {held}'
    y = torch.empty((m_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.mxfp4_mega_moe(
        y, eplb.l1_weights, eplb.l2_weights, b,
        cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
        l1_global_scales=eplb.l1_global_scales, l2_global_scales=eplb.l2_global_scales,
        activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    y_ref = _reference_output(x_fp8, x_sf, topk_idx, topk_weights, l1_dequant, l2_dequant,
                              l1_gs, l2_gs, rank_idx, H, args.activation_clamp, group)
    cosine = torch.nn.functional.cosine_similarity(y.float(), y_ref, dim=-1)
    assert torch.isfinite(y).all().item() and cosine.mean().item() > args.cosine_mean_threshold \
        and cosine.min().item() > args.cosine_min_threshold, \
        f'rank {rank_idx} ({tag}): remap-only output mismatch, cosine mean {cosine.mean().item():.4f} min {cosine.min().item():.4f}'
    if rank_idx == 0:
        print(f'remap-only ({tag}, M={m_tokens}): cosine mean {cosine.mean().item():.4f} min {cosine.min().item():.4f}', flush=True)


def _worker(local_rank, num_local_ranks, args):
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', flush=True)
            return
        num_ranks = dist.get_world_size(group)
        E, H = args.num_experts, args.num_experts // num_ranks
        torch.manual_seed(args.seed + rank_idx)
        l1_bf = torch.randn((H, 2 * args.intermediate_hidden, args.hidden), dtype=torch.bfloat16, device='cuda') * args.weight_scale
        l2_bf = torch.randn((H, args.hidden, args.intermediate_hidden), dtype=torch.bfloat16, device='cuda') * args.weight_scale
        l1_packed, l1_scale = quantize_to_mxfp4(l1_bf, group_size=32)
        l2_packed, l2_scale = quantize_to_mxfp4(l2_bf, group_size=32)
        l1_dequant = dequantize_mxfp4_to_fp32(l1_packed, l1_scale, group_size=32)
        l2_dequant = dequantize_mxfp4_to_fp32(l2_packed, l2_scale, group_size=32)
        t1, t2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90((l1_packed, l1_scale), (l2_packed, l2_scale))
        l1_gs = torch.linspace(0.73, 1.37, H, dtype=torch.float32, device='cuda')
        l2_gs = torch.linspace(1.31, 0.67, H, dtype=torch.float32, device='cuda')

        eplb = deep_gemm.DynamicEPLB(group, E, args.num_helper_slots, args.num_max_tokens_per_rank,
                                     args.num_topk, args.hidden, args.intermediate_hidden,
                                     open_gate_blocks=args.open_gate_blocks,
                                     fuse_copy=args.fuse_copy != 'separate')
        eplb.set_home_weights(t1, t2, l1_gs, l2_gs)
        state = (eplb, l1_dequant, l2_dequant, l1_gs.cpu().tolist(), l2_gs.cpu().tolist())
        if args.check_fixed:
            fixed_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group, E, args.num_max_tokens_per_rank, args.num_topk, args.hidden, args.intermediate_hidden,
                use_fp8_dispatch=True, activation='swiglu')
            state = state + (fixed_buffer, (t1, t2))

        args.trace_layers = _load_trace(args.trace, E, args.num_topk) if args.trace else None
        trace_layer_ids = list(range(0, len(args.trace_layers), max(1, len(args.trace_layers) // args.trace_num_layers))) \
            [:args.trace_num_layers] if args.trace_layers is not None else [0]
        if args.correctness:
            # Before any plan: everything must go home.
            _check_remap_only(args, 64, rank_idx, group, state, 'no plan yet')
        routings = {}
        for m_tokens in args.batches:
            for layer in trace_layer_ids:
                args.trace_layer = layer
                routings[(m_tokens, layer)] = _run_case(args, m_tokens, rank_idx, group, state,
                                                        check=bool(args.correctness))
                if args.correctness:
                    # Second pass on the same routing exercises the slot cache.
                    _run_case(args, m_tokens, rank_idx, group, state)
                    if args.fuse_copy == 'both':
                        # And the other copy arm on a fresh placement.
                        eplb.fuse_copy = False
                        eplb.slot_expert.fill_(-1)
                        _run_case(args, m_tokens, rank_idx, group, state)
                        eplb.fuse_copy = True
        if args.correctness:
            # Decode-like steps reuse the last plan without re-planning.
            for m_small in (8, 64, 256):
                _check_remap_only(args, m_small, rank_idx, group, state, 'last plan')
        if rank_idx == 0 and args.correctness:
            print('correctness: PASS', flush=True)

        if args.bench:
            # The first kineto session in a process drops its events, so open
            # and close one before anything is measured.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
                torch.cuda._sleep(1000)
                torch.cuda.synchronize()
            baseline_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group, E, args.num_max_tokens_per_rank, args.num_topk, args.hidden, args.intermediate_hidden,
                use_fp8_dispatch=True, activation='swiglu')
            for m_tokens in args.batches:
                for layer in trace_layer_ids:
                    if args.trace_layers is not None and rank_idx == 0:
                        print(f'[trace layer {layer}]', flush=True)
                    _bench(args, m_tokens, rank_idx, group, eplb, routings[(m_tokens, layer)], baseline_buffer, (t1, t2))
            baseline_buffer.destroy()
        eplb.destroy()
    finally:
        dist.destroy_process_group()


def _parse_args():
    p = argparse.ArgumentParser(description='SM90 MXFP4 MegaMoE dynamic EPLB gate')
    p.add_argument('--batches', nargs='+', type=int, default=[64, 256, 1024])
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=384)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-helper-slots', type=int, default=4)
    p.add_argument('--open-gate-blocks', type=int, default=0)
    p.add_argument('--hot-experts', nargs='+', type=int, default=[3, 7, 11])
    p.add_argument('--hot-share', type=float, default=0.5)
    p.add_argument('--num-max-tokens-per-rank', type=int, default=4096)
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weight-scale', type=float, default=0.05)
    p.add_argument('--cosine-mean-threshold', type=float, default=0.995)
    p.add_argument('--cosine-min-threshold', type=float, default=0.99)
    p.add_argument('--norm-ratio-min', type=float, default=0.99)
    p.add_argument('--norm-ratio-max', type=float, default=1.01)
    p.add_argument('--trace', type=str, default=None,
                   help='routing trace (.pt) captured from a model; replaces the synthetic skew')
    p.add_argument('--trace-num-layers', type=int, default=4, help='how many traced layers to replay')
    p.add_argument('--check-fixed', type=int, default=0,
                   help='also run the fixed-placement kernel on the same tokens as a control')
    p.add_argument('--fuse-copy', choices=['separate', 'fused', 'both'],
                   default='separate' if os.environ.get('DG_EPLB_FUSE_COPY', '1') == '0' else 'fused',
                   help='weight copy by the separate kernel, fused into the MegaMoE kernel, or both arms')
    p.add_argument('--bench-rounds', type=int, default=1,
                   help='repeat the separate/fused arm pair this many times per cell')
    p.add_argument('--cold-each-step', type=int, default=0,
                   help='benchmark with every helper slot re-copied on every step')
    p.add_argument('--single-launch', type=int, default=1,
                   help='also check plan + remap in one launch (mega_moe_eplb_plan_and_remap) per case')
    p.add_argument('--bench', type=int, default=1)
    p.add_argument('--correctness', type=int, default=1)
    p.add_argument('--bench-tests', type=int, default=20)
    return p.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    torch.multiprocessing.spawn(_worker, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes, join=True)
