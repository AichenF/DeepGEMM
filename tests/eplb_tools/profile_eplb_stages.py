"""Per-stage cost of the dynamic EPLB balancer on the critical path.

The in-server trace splits a balanced MoE forward into "prepare" (the
balancer's own work) and "kernel", with CUDA events; the prepare part of a
remap-only forward came out at ~85 us although the remap kernel itself runs
in 2-8 us. This script separates the three places that time can hide:

  (a) GPU time of every stage kernel (kineto, ``bench_kineto``);
  (b) host wall time of every Python call that issues it, measured with
      ``time.perf_counter`` and no synchronization in between;
  (c) the CUDA-event span from before ``block_m`` to after the MoE kernel,
      split at the kernel launch exactly like the server trace does.

Variants per (trace, layer, M):
  fresh        plan + copy + remap + kernel (replanning forward)
  single       plan + remap in one launch (+ copy) + kernel
  remap        remap onto the last plan + kernel (decode, REPLAN_EVERY > 1)
  remap_cached as ``remap`` with ``block_m`` computed once outside the loop
  kernel       kernel only on already physical routing (the floor)
  fresh_filler ``fresh`` preceded by a filler on the main stream: two BF16
               GEMMs shaped like DeepSeek V4 Flash's shared experts at this M
               (the server's ``--filler-n`` x hidden gate/up and down), i.e.
               the stock order gate -> ... -> shared experts -> plan -> kernel

``--wait-study`` adds, per (trace, layer, M), the plan kernel alone (the
standalone planner and the single-launch kernel) under three host alignments:
``aligned`` (barrier + synchronize immediately before the launch, so the
in-kernel peer wait sees the histograms already there), ``loop`` (barrier
before the forward, as the table above) and ``free`` (back-to-back launches,
no barrier, as a server runs layers). With ``DG_EPLB_PROFILE=1`` the kernels
are built with phase stamps and the in-kernel breakdown (histogram, publish,
peer wait, setup, greedy, pruning, slots, tables, remap) is reported next to
the event span, so wait and compute are separated.

Rank-max numbers are reported (the layer waits for the slowest rank); rank 0
prints. With ``--profile-m`` a torch.profiler session (CPU + CUDA) wraps the
``remap`` variant at that M and the top-25 ops by CPU and by CUDA time are
printed, so host-side costs are attributed to the op that pays them.

Usage (8 GPUs, one trace per invocation):
  python tests/eplb_tools/profile_eplb_stages.py --trace traces/mimo25_gsm8k.pt \
      --num-experts 384 --num-topk 8 --hidden 6144 --intermediate-hidden 2048
"""
import argparse
import contextlib
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS_DIR = os.path.join(REPO_ROOT, 'tests')
for p in (REPO_ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import deep_gemm
from deep_gemm import _C
from deep_gemm.quantization_mxfp4 import quantize_to_mxfp4
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from test_mxfp4_mega_moe_sm90_eplb import (PLAN_STAMPS, _load_trace, _single_launch, _trace_topk,
                                            plan_profile_enabled, read_plan_profile)

KERNELS = ('eplb_plan_kernel', 'eplb_copy_weights_kernel', 'eplb_remap_kernel', 'eplb_plan_remap_kernel',
           'sm90_mxfp4_mega_moe')
VARIANTS = ('fresh', 'single', 'remap', 'remap_cached', 'kernel', 'fresh_filler')
# ``zero`` stays as a column for comparison with older logs; ``DynamicEPLB.remap``
# no longer issues a fill launch (the remap kernel clears its own counters), so
# it is 0 now. ``filler`` is the host time of the filler GEMM calls.
HOST_FIELDS = ('block_m', 'plan', 'copy', 'zero', 'remap', 'launch', 'total', 'filler')
# Profiler scopes cost a few microseconds even with no profiler active, so
# they are only emitted inside the torch.profiler session.
_PROFILING = False


def _scope(name):
    return torch.profiler.record_function(name) if _PROFILING else contextlib.nullcontext()


def _resync(group):
    """Both the planner and the kernel wait on their peers in-kernel; keep
    the hosts aligned before anything that launches them."""
    torch.cuda.synchronize()
    dist.barrier(group=group)


def _rank_max(values, group):
    t = torch.tensor(values, dtype=torch.float64, device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.tolist()


class StageRunner:
    """Issues one forward of a given variant and records host and event
    timings. Every call reloads the logical routing first (remap rewrites it
    in place); that copy is issued before the first event and is not timed."""

    def __init__(self, eplb, m_tokens, topk_logical, args):
        self.eplb = eplb
        self.n = m_tokens
        self.topk_logical = topk_logical
        self.topk_physical = None
        self.args = args
        self.y = torch.empty((m_tokens, eplb.hidden), dtype=torch.bfloat16, device='cuda')
        self.cached_block_m = eplb.block_m(m_tokens)
        self.host = {}
        self.events = None
        # Filler standing in for the shared experts (BF16; the model's own may
        # be narrower-typed and faster, the server trace gives the real span):
        # gate/up [M, hidden] x [hidden, filler_n] and down [M, filler_n / 2] x
        # [filler_n / 2, hidden].
        fn = args.filler_n
        self.filler_x = torch.randn((m_tokens, eplb.hidden), dtype=torch.bfloat16, device='cuda')
        self.filler_w1 = torch.randn((eplb.hidden, fn), dtype=torch.bfloat16, device='cuda') * 0.02
        self.filler_w2 = torch.randn((fn // 2, eplb.hidden), dtype=torch.bfloat16, device='cuda') * 0.02

    def _kernel(self):
        b = self.eplb.symm
        deep_gemm.mxfp4_mega_moe(
            self.y, self.eplb.l1_weights, self.eplb.l2_weights, b,
            cumulative_local_expert_recv_stats=self.eplb.cumulative_slot_recv_stats,
            l1_global_scales=self.eplb.l1_global_scales, l2_global_scales=self.eplb.l2_global_scales,
            activation_clamp=self.args.activation_clamp, fast_math=True)

    def _filler(self):
        with _scope('filler'):
            t0 = time.perf_counter()
            h = self.filler_x @ self.filler_w1
            h = h[:, : self.filler_w2.shape[0]] * torch.sigmoid(h[:, self.filler_w2.shape[0]:])
            _ = h @ self.filler_w2
            return time.perf_counter() - t0

    def _remap_split(self, block_m):
        """`DynamicEPLB.remap`: one launch (its kernel clears the ordinal
        counters, the planner too), timed on its own; the ``zero`` slot that
        older logs had is kept at 0."""
        e = self.eplb
        t1 = time.perf_counter()
        with _scope('eplb.remap'):
            e.remap(self.n, block_m)
        t2 = time.perf_counter()
        return 0.0, t2 - t1

    def run(self, variant, record=True):
        e, n = self.eplb, self.n
        b = e.symm
        src = self.topk_physical if variant == 'kernel' else self.topk_logical
        b.topk_idx[:n].copy_(src)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if record else None
        h = dict.fromkeys(HOST_FIELDS, 0.0)
        if record:
            ev[0].record()
        t_start = time.perf_counter()
        if variant == 'fresh' or variant == 'fresh_filler':
            if variant == 'fresh_filler':
                h['filler'] = self._filler()
            if record:
                ev[3].record()
            with _scope('eplb.block_m'):
                t0 = time.perf_counter(); block_m = e.block_m(n); t1 = time.perf_counter()
            with _scope('eplb.plan'):
                e.plan(n, block_m); t2 = time.perf_counter()
            with _scope('eplb.copy_weights'):
                e.copy_weights(); t3 = time.perf_counter()
            h['block_m'], h['plan'], h['copy'] = t1 - t0, t2 - t1, t3 - t2
            h['zero'], h['remap'] = self._remap_split(block_m)
        elif variant == 'single':
            # One launch for plan + remap; the copy follows unless fused.
            with _scope('eplb.block_m'):
                t0 = time.perf_counter(); block_m = e.block_m(n); t1 = time.perf_counter()
            with _scope('eplb.plan_and_remap(_C)'):
                _single_launch(e, n, block_m); t2 = time.perf_counter()
            t3 = t2
            if not e.fuse_copy:
                with _scope('eplb.copy_weights'):
                    e.copy_weights(); t3 = time.perf_counter()
            h['block_m'], h['plan'], h['copy'] = t1 - t0, t2 - t1, t3 - t2
        elif variant == 'remap':
            with _scope('eplb.block_m'):
                t0 = time.perf_counter(); block_m = e.block_m(n); t1 = time.perf_counter()
            h['block_m'] = t1 - t0
            h['zero'], h['remap'] = self._remap_split(block_m)
        elif variant == 'remap_cached':
            h['zero'], h['remap'] = self._remap_split(self.cached_block_m)
        if record:
            ev[1].record()
        t4 = time.perf_counter()
        with _scope('mxfp4_mega_moe'):
            self._kernel()
        t5 = time.perf_counter()
        h['launch'] = t5 - t4
        h['total'] = t5 - t_start
        if record:
            ev[2].record()
        self.host = h
        self.events = ev

    def capture_physical(self):
        """Remap once on the current placement and keep the physical ids for
        the kernel-only floor."""
        self.run('remap', record=False)
        torch.cuda.synchronize()
        self.topk_physical = self.eplb.symm.topk_idx[:self.n].clone()


def _measure_variant(runner, variant, group, iters):
    """Per-iteration host and event timings with the ranks realigned before
    each one; returns the per-rank medians (seconds / microseconds)."""
    hosts = {k: [] for k in HOST_FIELDS}
    prep, kern, tot, fill = [], [], [], []
    filler = variant.endswith('_filler')
    for _ in range(3):
        _resync(group)
        runner.run(variant)
    for _ in range(iters):
        _resync(group)
        runner.run(variant)
        torch.cuda.synchronize()
        for k in HOST_FIELDS:
            hosts[k].append(runner.host[k] * 1e6)
        e0, e1, e2, e3 = runner.events
        # With a filler, "prepare" is what follows it on the critical path.
        prep.append((e3 if filler else e0).elapsed_time(e1) * 1e3)
        kern.append(e1.elapsed_time(e2) * 1e3)
        tot.append(e0.elapsed_time(e2) * 1e3)
        fill.append(e0.elapsed_time(e3) * 1e3 if filler else 0.0)
    med = statistics.median
    return {k: med(v) for k, v in hosts.items()}, med(prep), med(kern), med(tot), med(fill)


def _kineto_variant(runner, variant, group, num_tests, quiet):
    _resync(group)
    times = bench_kineto(lambda: runner.run(variant, record=False), KERNELS, barrier=dist.barrier,
                         num_tests=num_tests, suppress_kineto_output=quiet)
    _resync(group)
    return [t * 1e6 for t in times]


def _cold_copy(eplb, runner, group):
    """Every helper slot changes: the full weight transfer of a first plan."""
    n = runner.n
    b = eplb.symm
    for _ in range(2):
        eplb.slot_expert.fill_(-1)
        b.topk_idx[:n].copy_(runner.topk_logical)
        _resync(group)
        eplb.plan(n)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        copies = int(eplb.copy_count.item())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        eplb.copy_weights()
        end.record()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
    gpu_us, host_us, copies = _rank_max([start.elapsed_time(end) * 1e3, (t1 - t0) * 1e6, copies], group)
    # Leave a steady state behind: a second plan on the same routing changes nothing.
    b.topk_idx[:n].copy_(runner.topk_logical)
    _resync(group)
    eplb.prepare(n)
    torch.cuda.synchronize()
    return gpu_us, host_us, int(copies)


# The single-launch entry point needs a `_C` built with it; older builds run
# the rest of the script unchanged.
_HAS_SINGLE_LAUNCH = hasattr(_C, 'mega_moe_eplb_plan_and_remap')
WAIT_MODES = ('aligned', 'loop', 'free')
# Per-iteration fields of the wait study: event span, in-kernel total, the
# phases, then the three globaltimer stamps and the loop counters.
_WS_PHASES = PLAN_STAMPS[1:]
_WS_FIELDS = ('ev_span', 'total') + _WS_PHASES + ('t_entry', 't_peers', 't_end', 'picks', 'moves', 'prune_steps',
                                                  'extreme_moves', 'grid')


def _ws_row(ev_span_us, prof):
    if prof is None:
        return [ev_span_us] + [0.0] * (len(_WS_FIELDS) - 1)
    ph, c = prof['phases'], prof['counters']
    return ([ev_span_us, prof['total_us']] + [ph[k] for k in _WS_PHASES]
            + [float(prof['timer_entry_ns']), float(prof['timer_peers_ns']), float(prof['timer_end_ns']),
               float(c['picks']), float(c['moves']), float(c['prune_steps']), float(c['extreme_moves']), float(c['grid'])])


def _plan_wait_study(runner, group, rank_idx, tag, layer, iters):
    """The plan kernel alone (standalone planner and single-launch kernel)
    under the three host alignments of the module docstring; prints the event
    span and, when the kernels carry stamps (``DG_EPLB_PROFILE=1``), the
    in-kernel phase breakdown with the peer wait separated from compute, plus
    the launch skew between ranks (globaltimer at entry, rank max - min)."""
    e, n = runner.eplb, runner.n
    block_m = runner.cached_block_m
    slots = int(e.ws_layout.get('num_profile_slots', 1))
    profiled = plan_profile_enabled()

    def reload():
        e.symm.topk_idx[:n].copy_(runner.topk_logical)

    kernels = ('plan', 'single') if _HAS_SINGLE_LAUNCH else ('plan',)
    for kernel in kernels:
        launch = (lambda: e.plan(n, block_m)) if kernel == 'plan' else (lambda: _single_launch(e, n, block_m))
        for mode in WAIT_MODES:
            rows = []
            for _ in range(3):
                _resync(group)
                reload()
                launch()
            torch.cuda.synchronize()
            if mode == 'free':
                # Back-to-back launches with nothing in between but the
                # routing reload; one profile record per launch, read after
                # the batch so no synchronisation interrupts the stream.
                _resync(group)
                for _ in range(max(1, iters // slots)):
                    evs = []
                    gen0 = e.generation
                    for _ in range(slots):
                        reload()
                        ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        ev[0].record()
                        launch()
                        ev[1].record()
                        evs.append(ev)
                    torch.cuda.synchronize()
                    for i, (a, b) in enumerate(evs):
                        prof = read_plan_profile(e, gen0 + 1 + i) if profiled else None
                        rows.append(_ws_row(a.elapsed_time(b) * 1e3, prof))
            else:
                for _ in range(iters):
                    if mode == 'loop':
                        _resync(group)
                    reload()
                    if mode == 'aligned':
                        torch.cuda.synchronize()
                        dist.barrier(group=group)
                    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    ev[0].record()
                    launch()
                    ev[1].record()
                    torch.cuda.synchronize()
                    prof = read_plan_profile(e) if profiled else None
                    rows.append(_ws_row(ev[0].elapsed_time(ev[1]) * 1e3, prof))
            t = torch.tensor(rows, dtype=torch.float64, device='cuda')  # [iters, fields]
            t_max, t_min = t.clone(), t.clone()
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(t_min, op=dist.ReduceOp.MIN, group=group)
            if rank_idx != 0:
                continue
            med = lambda x: x.median(dim=0).values.tolist()
            mx, mn = dict(zip(_WS_FIELDS, med(t_max))), dict(zip(_WS_FIELDS, med(t_min)))
            fe, fp = _WS_FIELDS.index('t_entry'), _WS_FIELDS.index('t_peers')
            skew_entry = ((t_max[:, fe] - t_min[:, fe]) / 1e3).median().item()
            skew_peers = ((t_max[:, fp] - t_min[:, fp]) / 1e3).median().item()
            head = (f'[wait study {tag} L{layer} M={n} {kernel:<6} {mode:<7}] ev_span max {mx["ev_span"]:.1f} '
                    f'min {mn["ev_span"]:.1f} us')
            if not profiled:
                print(head, flush=True)
                continue
            phases = ' + '.join(f'{k} {mx[k]:.1f}' for k in _WS_PHASES if mx[k] >= 0.05)
            print(f'{head} | in-kernel total {mx["total"]:.1f} = {phases} | wait min {mn["wait"]:.1f} '
                  f'| compute (total - wait) {mx["total"] - mx["wait"]:.1f} | entry skew {skew_entry:.1f} us, '
                  f'peers-ready skew {skew_peers:.1f} us | picks={int(mx["picks"])} moves={int(mx["moves"])} '
                  f'prune={int(mx["prune_steps"])} extreme={int(mx["extreme_moves"])} grid={int(mx["grid"])}', flush=True)
    _resync(group)
    # Leave the routing logical and the placement settled for the next cell.
    reload()
    e.prepare(n)
    torch.cuda.synchronize()


def _profile_session(runner, group, rank_idx, iters, out_dir, tag):
    """torch.profiler (CPU + CUDA) around the remap variant; rank 0 prints the
    top-25 ops by CPU and by CUDA time and writes the chrome trace."""
    for _ in range(3):
        _resync(group)
        runner.run('remap')
    global _PROFILING
    _PROFILING = True
    _resync(group)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            with _scope('resync'):
                _resync(group)
            with _scope('forward.remap_variant'):
                runner.run('remap')
        torch.cuda.synchronize()
    _PROFILING = False
    _resync(group)
    if rank_idx != 0:
        return
    ka = prof.key_averages()
    print(f'\n[profiler {tag} M={runner.n}] {iters} remap-variant forwards, top-25 by CPU total', flush=True)
    print(ka.table(sort_by='cpu_time_total', row_limit=25, max_name_column_width=70), flush=True)
    print(f'\n[profiler {tag} M={runner.n}] top-25 by CUDA total', flush=True)
    print(ka.table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70), flush=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'profile_stages_{tag}_m{runner.n}_rank0.json')
        prof.export_chrome_trace(path)
        print(f'[profiler] chrome trace: {path}', flush=True)


def _worker(local_rank, num_local_ranks, args):
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', flush=True)
            return
        num_ranks = dist.get_world_size(group)
        E, H = args.num_experts, args.num_experts // num_ranks
        tag = os.path.splitext(os.path.basename(args.trace))[0]
        torch.manual_seed(args.seed + rank_idx)
        l1_bf = torch.randn((H, 2 * args.intermediate_hidden, args.hidden), dtype=torch.bfloat16, device='cuda') * 0.05
        l2_bf = torch.randn((H, args.hidden, args.intermediate_hidden), dtype=torch.bfloat16, device='cuda') * 0.05
        l1_packed, l1_scale = quantize_to_mxfp4(l1_bf, group_size=32)
        l2_packed, l2_scale = quantize_to_mxfp4(l2_bf, group_size=32)
        del l1_bf, l2_bf
        t1, t2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90((l1_packed, l1_scale), (l2_packed, l2_scale))
        del l1_packed, l1_scale, l2_packed, l2_scale
        l1_gs = torch.linspace(0.73, 1.37, H, dtype=torch.float32, device='cuda')
        l2_gs = torch.linspace(1.31, 0.67, H, dtype=torch.float32, device='cuda')
        eplb = deep_gemm.DynamicEPLB(group, E, args.num_helper_slots, args.num_max_tokens_per_rank,
                                     args.num_topk, args.hidden, args.intermediate_hidden)
        eplb.set_home_weights(t1, t2, l1_gs, l2_gs)

        layers = _load_trace(args.trace, E, args.num_topk)
        layer_ids = list(range(0, len(layers), max(1, len(layers) // args.trace_num_layers)))[:args.trace_num_layers]
        gen = torch.Generator(device='cuda')
        gen.manual_seed(args.seed)

        # The first kineto session in a process drops its events.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
            torch.cuda._sleep(1000)
            torch.cuda.synchronize()
        quiet = True
        if rank_idx == 0:
            print(f'[setup {tag}] E={E} top-{args.num_topk} hidden={args.hidden} inter={args.intermediate_hidden} '
                  f'ranks={num_ranks} helper_slots={args.num_helper_slots} layers={layer_ids} '
                  f'trace_layers={len(layers)} rows/layer={layers[0].shape[0]}', flush=True)
            print('columns: gpu_us = kineto per-kernel time (plan/copy/remap/single = plan+remap in one launch/kernel); '
                  'host_us = perf_counter per Python call (block_m/plan/copy/zero/remap/launch/total/filler; for the '
                  'single variant "plan" is the one plan+remap call); ev_us = CUDA-event spans '
                  '(prepare = before block_m -> before kernel launch, or after the filler -> before kernel launch for '
                  'the *_filler variants; kernel; total; filler); '
                  'gap_us = ev_total - sum(gpu_us) - ev_filler. All rank-max.', flush=True)
            print('trace          layer     M variant      | gpu_us plan   copy  remap single kernel | host_us block_m  plan  copy  zero remap launch  total filler '
                  '| ev_us prepare kernel  total filler |  gap_us', flush=True)

        profiled = False
        for layer in layer_ids:
            for m_tokens in args.batches:
                topk_idx, topk_weights = _trace_topk(layers, layer, m_tokens, rank_idx, num_ranks, gen)
                x_bf = torch.randn((m_tokens, args.hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
                x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)
                b = eplb.symm
                b.x[:m_tokens].copy_(x_fp8)
                b.x_sf[:m_tokens].copy_(x_sf)
                b.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))
                runner = StageRunner(eplb, m_tokens, topk_idx.to(torch.int64).contiguous(), args)
                del x_bf, x_fp8, x_sf

                cold_gpu, cold_host, copies = _cold_copy(eplb, runner, group)
                if rank_idx == 0:
                    print(f'[cold copy {tag} layer {layer} M={m_tokens}] {copies} slots changed on the slowest rank: '
                          f'gpu {cold_gpu:.1f}us host {cold_host:.1f}us', flush=True)
                # Build every kernel instantiation and settle the placement before timing.
                _resync(group)
                runner.run('fresh', record=False)
                torch.cuda.synchronize()
                runner.capture_physical()

                for variant in (VARIANTS if args.table else ()):
                    if variant.endswith('_filler') and not args.filler:
                        continue
                    if variant == 'single' and not _HAS_SINGLE_LAUNCH:
                        continue
                    gpu = _kineto_variant(runner, variant, group, args.bench_tests, quiet)
                    host, prep, kern, tot, fill = _measure_variant(runner, variant, group, args.iters)
                    vals = _rank_max(gpu + [host[k] for k in HOST_FIELDS] + [prep, kern, tot, fill], group)
                    nk = len(KERNELS)
                    gpu = vals[:nk]
                    hv = vals[nk:nk + len(HOST_FIELDS)]
                    prep, kern, tot, fill = vals[nk + len(HOST_FIELDS):]
                    gap = tot - sum(gpu) - fill
                    if rank_idx == 0:
                        print(f'{tag:<14} {layer:>5} {m_tokens:>5} {variant:<12} | '
                              f'{gpu[0]:>12.1f} {gpu[1]:>6.1f} {gpu[2]:>6.1f} {gpu[3]:>6.1f} {gpu[4]:>6.1f} | '
                              f'{hv[0]:>15.1f} {hv[1]:>5.1f} {hv[2]:>5.1f} {hv[3]:>5.1f} {hv[4]:>5.1f} {hv[5]:>6.1f} {hv[6]:>6.1f} {hv[7]:>6.1f} | '
                              f'{prep:>13.1f} {kern:>6.1f} {tot:>6.1f} {fill:>6.1f} | {gap:>7.1f}', flush=True)
                if args.wait_study:
                    _plan_wait_study(runner, group, rank_idx, tag, layer, args.wait_iters)
                if args.profile_m == m_tokens and not profiled:
                    _profile_session(runner, group, rank_idx, args.profile_iters, args.profile_dir, tag)
                    profiled = True
                del runner
        if rank_idx == 0:
            print(f'[done {tag}]', flush=True)
        eplb.destroy()
    finally:
        dist.destroy_process_group()


def _parse_args():
    p = argparse.ArgumentParser(description='Per-stage profile of the SM90 MXFP4 MegaMoE dynamic EPLB')
    p.add_argument('--trace', type=str, required=True, help='routing trace (.pt) captured from a model')
    p.add_argument('--trace-num-layers', type=int, default=2)
    p.add_argument('--batches', nargs='+', type=int, default=[1024, 2048, 4096])
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=384)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-helper-slots', type=int, default=4)
    p.add_argument('--num-max-tokens-per-rank', type=int, default=4096)
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bench-tests', type=int, default=10, help='kineto iterations per variant')
    p.add_argument('--iters', type=int, default=20, help='host/event timing iterations per variant')
    p.add_argument('--profile-m', type=int, default=2048, help='M for the torch.profiler session (0 = off)')
    p.add_argument('--profile-iters', type=int, default=10)
    p.add_argument('--profile-dir', type=str, default=None, help='where to write the chrome trace')
    p.add_argument('--filler', type=int, default=1, help='also run the fresh_filler variant')
    p.add_argument('--table', type=int, default=1, help='run the per-variant table (0: only the studies below)')
    p.add_argument('--wait-study', type=int, default=1,
                   help='plan kernel alone under aligned / loop / free host alignment (set DG_EPLB_PROFILE=1 '
                        'for the in-kernel breakdown)')
    p.add_argument('--wait-iters', type=int, default=24)
    p.add_argument('--filler-n', type=int, default=4096,
                   help='gate/up width of the shared-expert stand-in (2 x intermediate x shared experts)')
    return p.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    torch.multiprocessing.spawn(_worker, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes, join=True)
