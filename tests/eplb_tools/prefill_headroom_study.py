"""Prefill headroom study for the MegaMoE dynamic EPLB on real routing traces.

For the compute-bound prefill kernel the MoE cost of a rank is the number of
token blocks (``block_m`` rows each) of the experts it executes. This script
measures, on captured prefill routing, how far the busiest rank is from the
ideal ``total_blocks / EP`` under

* ``home``: the contiguous placement the serving baseline uses (expert ``e``
  on rank ``e // H``), no replicas; also the no-algo column;
* ``planner``: the MegaFlux-style device planner, host reference
  ``plan_reference`` in ``deep_gemm/mega/eplb_planner.py``, with ``S`` helper
  slots per rank (``fresh``: no slot cache; ``chained``: the slot cache of the
  previous chunk of the same layer, which is what the serving path does);
* ``opt``: the exact optimum with ``S`` helper slots per rank, whole-block
  splitting of a replicated expert between its holders (mixed-integer program
  solved with HiGHS through ``scipy.optimize.milp``), plus its LP relaxation
  as a lower bound;
* ``ideal``: ``total_blocks / EP``.

Subcommands::

    # On a machine with torch: per-layer expert histograms over consecutive
    # 512-token windows of the captured call stream (a few MB per trace).
    python prefill_headroom_study.py cache <trace.pt> <out.npz> [--window 512]

    # Anywhere with numpy + scipy: the study itself.
    python prefill_headroom_study.py study --cache dsv4_gsm8k=<npz> ... \
        --out results.json --md tables.md [--ep 4 8] [--chunks 4096 16384 32768] \
        [--slots 0 4 8] [--block-m 64] [--max-chunks-per-layer N] [--workers P]

    # Owner permutation (which rank owns which expert, chosen once per layer
    # from the first quarter of the stream; zero per-call cost) alone, under
    # the planner, and with an EPLB-style static set + per-chunk split.
    python prefill_headroom_study.py permute --cache dsv4_gsm8k=<npz> --cache dsv4_sharegpt=<npz> \
        --cross dsv4_gsm8k:dsv4_sharegpt --cross dsv4_sharegpt:dsv4_gsm8k --out permute.json

    # Unit checks on synthetic inputs (no trace needed).
    python prefill_headroom_study.py selftest

Chunks are consecutive windows of the captured call stream of a layer (the
trace concatenates the captured calls in order, so a chunk of ``C`` tokens
is the routing of ``C`` consecutive prompt tokens as the server would see it
with ``chunked_prefill_size = C``); a trailing partial chunk is kept when it
holds at least half of ``C``. For ``C = 32768`` a stream shorter than ``C`` is
taken whole. The chunk is sequence-sharded across the ``EP`` ranks, but every
rank's tokens go to the experts' home ranks, so the per-rank load depends
only on the chunk's per-expert totals; the planner reference likewise uses
only the totals.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLANNER_PATH = os.path.normpath(os.path.join(_HERE, '..', '..', 'deep_gemm', 'mega', 'eplb_planner.py'))


def _load_planner_module():
    """``deep_gemm.mega.eplb_planner`` without importing the ``deep_gemm``
    package (which needs torch and the compiled extension)."""
    spec = importlib.util.spec_from_file_location('eplb_planner_ref', _PLANNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# cache: trace .pt -> per-window histograms (.npz)
# --------------------------------------------------------------------------

def cmd_cache(args):
    import numpy as np
    import torch
    trace = torch.load(args.trace, map_location='cpu')
    E, K = int(trace['num_experts']), int(trace['num_topk'])
    layers = trace['layers']
    W = args.window
    num_tokens = [int(t.shape[0]) for t in layers]
    n_win = max(-(-n // W) for n in num_tokens)
    hist = np.zeros((len(layers), n_win, E), dtype=np.int32)
    for li, t in enumerate(layers):
        ids = t.to(torch.int64).numpy()
        n = ids.shape[0]
        for w in range(-(-n // W)):
            seg = ids[w * W:(w + 1) * W].reshape(-1)
            hist[li, w] = np.bincount(seg, minlength=E)
    np.savez_compressed(args.out, hist=hist, layer_ids=np.asarray(trace['layer_ids']),
                        num_tokens=np.asarray(num_tokens), window=W, num_experts=E, num_topk=K)
    print(f'{args.out}: {len(layers)} layers, tokens per layer {min(num_tokens)}..{max(num_tokens)}, '
          f'{n_win} windows of {W}, E={E} K={K}')


# --------------------------------------------------------------------------
# loads
# --------------------------------------------------------------------------

def blocks_of(totals: Sequence[int], block_m: int) -> List[int]:
    return [-(-int(t) // block_m) for t in totals]


def home_loads(b: Sequence[int], R: int) -> List[int]:
    E = len(b)
    H = E // R
    return [sum(b[r * H:(r + 1) * H]) for r in range(R)]


def planner_max_load(planner_mod, totals: Sequence[int], R: int, S: int, block_m: int,
                     prev_slot_expert=None, open_gate_blocks: int = 0):
    """Busiest-rank blocks after the device planner's host reference. Returns
    (max_load, slot_expert) so the caller can chain the slot cache."""
    E = len(totals)
    counts = [[0] * E for _ in range(R)]
    counts[0] = [int(t) for t in totals]
    plan = planner_mod.plan_reference(counts, S, block_m, prev_slot_expert=prev_slot_expert,
                                      open_gate_blocks=open_gate_blocks)
    return max(plan.loads), plan.slot_expert, plan


def stale_loads(prev_plan, b: Sequence[int], R: int) -> List[int]:
    """Per-rank blocks when chunk ``i`` is remapped onto the plan of chunk
    ``i - 1`` (deferred mode): expert ``e``'s blocks fill its copies in order
    by the previous cumulative block ranges, and everything past the last
    range goes to the last copy, exactly as ``remap_reference`` does."""
    load = [0] * R
    for e, blocks in enumerate(b):
        prev_cum = 0
        copies = prev_plan.copies[e]
        for k, (rank, cum) in enumerate(copies):
            if k == len(copies) - 1:
                take = blocks - min(blocks, prev_cum)
            else:
                take = min(blocks, cum) - min(blocks, prev_cum)
            load[rank] += take
            prev_cum = cum
    return load


# --------------------------------------------------------------------------
# exact optimum (MILP) and LP bound
# --------------------------------------------------------------------------

def opt_min_max(b: Sequence[int], R: int, S: int, ub: Optional[int] = None,
                time_limit: float = 60.0, relax: bool = False) -> Tuple[float, bool]:
    """Minimum busiest-rank blocks when every rank may hold up to ``S``
    replicas of experts it does not own and a replicated expert's blocks are
    split in whole blocks between its holders.

    Only ranks whose home load exceeds ``ceil(total / R)`` can need to export
    (any export from a rank already at or below the optimum can be undone
    without raising the maximum), so replica variables exist only for their
    experts. Given the replica choice the block split is a transportation
    problem with integer data, so the block variables may stay continuous.
    Returns ``(value, proven)``; with ``relax=True`` the LP relaxation (a
    lower bound, fractional slots and blocks) is returned instead.
    """
    import numpy as np
    from scipy.optimize import LinearConstraint, Bounds, milp
    from scipy.sparse import coo_matrix

    E = len(b)
    H = E // R
    base = home_loads(b, R)
    lb0 = -(-sum(b) // R)
    if S == 0 or max(base) == lb0:
        return float(max(base)), True
    donors = [r for r in range(R) if base[r] > lb0]
    ex = [e for d in donors for e in range(d * H, (d + 1) * H) if b[e] > 0]
    owner = {e: e // H for e in ex}
    nx = len(ex) * R
    pairs = [(e, r) for e in ex for r in range(R) if r != owner[e]]
    ny = len(pairs)
    # variable layout: [T] [x_{e,r} for e in ex, r in R] [y_{e,r} for pairs]
    nv = 1 + nx + ny
    xi = {(e, r): 1 + i * R + r for i, e in enumerate(ex) for r in range(R)}
    yi = {p: 1 + nx + j for j, p in enumerate(pairs)}
    rows, cols, vals, lo, hi = [], [], [], [], []
    row = 0
    # (1) each donor expert's blocks are fully assigned
    for e in ex:
        for r in range(R):
            rows.append(row); cols.append(xi[(e, r)]); vals.append(1.0)
        lo.append(float(b[e])); hi.append(float(b[e])); row += 1
    # (2) rank load <= T
    fixed = [base[r] - sum(b[e] for e in ex if owner[e] == r) for r in range(R)]
    for r in range(R):
        for e in ex:
            rows.append(row); cols.append(xi[(e, r)]); vals.append(1.0)
        rows.append(row); cols.append(0); vals.append(-1.0)
        lo.append(-np.inf); hi.append(float(-fixed[r])); row += 1
    # (3) x_{e,r} <= b_e y_{e,r}
    for (e, r) in pairs:
        rows.append(row); cols.append(xi[(e, r)]); vals.append(1.0)
        rows.append(row); cols.append(yi[(e, r)]); vals.append(-float(b[e]))
        lo.append(-np.inf); hi.append(0.0); row += 1
    # (4) at most S replicas per rank
    for r in range(R):
        idx = [yi[p] for p in pairs if p[1] == r]
        if not idx:
            continue
        for j in idx:
            rows.append(row); cols.append(j); vals.append(1.0)
        lo.append(-np.inf); hi.append(float(S)); row += 1
    A = coo_matrix((vals, (rows, cols)), shape=(row, nv)).tocsr()
    c = np.zeros(nv); c[0] = 1.0
    lower = np.zeros(nv); upper = np.full(nv, np.inf)
    lower[0] = float(lb0)
    upper[0] = float(ub if ub is not None else max(base))
    for (e, r), j in xi.items():
        upper[j] = float(b[e])
    for j in yi.values():
        upper[j] = 1.0
    integrality = np.zeros(nv)
    if not relax:
        integrality[0] = 1
        for j in yi.values():
            integrality[j] = 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lower, upper),
               integrality=integrality, options={'time_limit': time_limit, 'disp': False})
    if res.x is None:
        # Infeasible under the upper bound (should not happen) or time out
        # without an incumbent: fall back to the best bound known.
        val = float(getattr(res, 'mip_dual_bound', lb0) or lb0)
        return val, False
    val = float(res.x[0])
    if relax:
        return val, True
    proven = (res.status == 0) or (getattr(res, 'mip_gap', 1.0) is not None and res.mip_gap < 1e-6)
    return round(val), bool(proven)


# --------------------------------------------------------------------------
# study
# --------------------------------------------------------------------------

def chunk_groups(n_tokens: int, window: int, chunk: int) -> List[Tuple[int, int]]:
    """(first_window, last_window_exclusive) pairs covering consecutive
    chunks of ``chunk`` tokens; a trailing partial chunk is kept if it holds
    at least half a chunk; a stream shorter than the chunk is one chunk."""
    n_win = -(-n_tokens // window)
    per = chunk // window
    if n_tokens <= chunk:
        return [(0, n_win)]
    out = []
    w = 0
    while w < n_win:
        end = min(w + per, n_win)
        tokens = min(n_tokens, end * window) - w * window
        if tokens >= chunk // 2:
            out.append((w, end))
        w = end
    return out


def _case_worker(payload):
    """One (layer, chunk) instance for all (EP, S) settings. Runs in a worker
    process; returns a list of result rows."""
    import numpy as np
    (name, li, ci, totals, eps, slots, block_m, opt_mode, time_limit, prev_slots) = payload
    planner = _load_planner_module()
    b = blocks_of(totals, block_m)
    rows = []
    new_prev = {}
    for R in eps:
        base = home_loads(b, R)
        ideal = sum(b) / R
        for S in slots:
            rec = dict(trace=name, layer=li, chunk=ci, ep=R, slots=S,
                       total_blocks=int(sum(b)), ideal=ideal, home=max(base))
            if S == 0:
                rec.update(planner=max(base), planner_chained=max(base), opt=max(base), opt_proven=True,
                           opt_lp=float(max(base)), replicas=0, stale=max(base), resplit=max(base), union=max(base), first=None)
            else:
                t0 = time.perf_counter()
                p_fresh, slot_fresh, plan_fresh = planner_max_load(planner, totals, R, S, block_m)
                prev = prev_slots.get((R, S))
                prev_table, prev_plan = prev if prev is not None else (None, None)
                p_chain, slot_chain, plan_chain = planner_max_load(planner, totals, R, S, block_m, prev_slot_expert=prev_table)
                new_prev[(R, S)] = (slot_chain, plan_chain)
                rec['planner'] = p_fresh
                rec['planner_chained'] = p_chain
                if prev_plan is None:
                    # First chunk of the stream: nothing to be stale against.
                    rec['stale'] = max(base)
                    rec['resplit'] = max(base)
                    rec['union'] = p_fresh
                    rec['first'] = True
                else:
                    # Deferred mode: chunk i remapped onto the plan of chunk i-1.
                    rec['stale'] = max(stale_loads(prev_plan, b, R))
                    # Cheap incremental refresh, upper bound: the previous
                    # replicas stay (cached, no copy) and up to S new ones may
                    # be added from the fresh counts; implemented as a fresh
                    # plan with 2S slots whose cache is the previous table.
                    # Previous replicas kept, block split recomputed from the
                    # fresh counts, no new copy (open gate too high to open one).
                    rec['resplit'] = planner_max_load(planner, totals, R, S, block_m, prev_slot_expert=prev_table,
                                                      open_gate_blocks=1 << 30)[0]
                    padded = [list(row) + [-1] * S for row in prev_table]
                    rec['union'] = planner_max_load(planner, totals, R, 2 * S, block_m, prev_slot_expert=padded)[0]
                    rec['first'] = False
                rec['replicas'] = sum(1 for row in slot_fresh for e in row if e >= 0)
                rec['planner_s'] = time.perf_counter() - t0
                ub = min(p_fresh, p_chain)
                lb0 = -(-sum(b) // R)
                if opt_mode == 'none':
                    rec.update(opt=None, opt_proven=False, opt_lp=None)
                elif ub == lb0:
                    rec.update(opt=ub, opt_proven=True, opt_lp=float(lb0))
                else:
                    lp, _ = opt_min_max(b, R, S, ub=ub, time_limit=time_limit, relax=True)
                    rec['opt_lp'] = lp
                    if math.ceil(lp - 1e-9) >= ub or opt_mode == 'lp':
                        rec.update(opt=ub if math.ceil(lp - 1e-9) >= ub else None,
                                   opt_proven=math.ceil(lp - 1e-9) >= ub)
                    else:
                        t1 = time.perf_counter()
                        v, proven = opt_min_max(b, R, S, ub=ub, time_limit=time_limit)
                        rec.update(opt=v, opt_proven=proven, opt_s=time.perf_counter() - t1)
            rows.append(rec)
    return rows, new_prev


def run_study(args):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    caches = {}
    for spec in args.cache:
        name, path = spec.split('=', 1)
        caches[name] = np.load(path)
    all_rows = []
    meta = {}
    for name, z in caches.items():
        hist = z['hist']
        num_tokens = z['num_tokens'].tolist()
        W = int(z['window'])
        E = int(z['num_experts'])
        meta[name] = dict(layers=int(hist.shape[0]), num_experts=E, num_topk=int(z['num_topk']),
                          tokens_per_layer=[int(x) for x in num_tokens], window=W,
                          layer_ids=[int(x) for x in z['layer_ids']])
        for chunk in args.chunks:
            layer_payloads = []
            for li in range(hist.shape[0]):
                groups = chunk_groups(num_tokens[li], W, chunk)
                if args.max_chunks_per_layer and len(groups) > args.max_chunks_per_layer:
                    step = len(groups) / args.max_chunks_per_layer
                    groups = [groups[int(i * step)] for i in range(args.max_chunks_per_layer)]
                layer_payloads.append([(li, ci, hist[li, a:bnd].sum(0).tolist()) for ci, (a, bnd) in enumerate(groups)])
            t0 = time.perf_counter()
            # Chunks of one layer are chained (slot cache), so a layer is a
            # sequential unit; layers run in parallel.
            jobs = [(name, chunk, p, args.ep, args.slots, args.block_m, args.opt, args.time_limit,
                     meta[name]['num_topk']) for p in layer_payloads]
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    for res in ex.map(_layer_job_entry, jobs):
                        all_rows.extend(res)
            else:
                for j in jobs:
                    all_rows.extend(_layer_job_entry(j))
            n_cases = sum(len(p) for p in layer_payloads)
            print(f'{name} chunk {chunk}: {n_cases} (layer, chunk) cases x {len(args.ep)} EP x {len(args.slots)} S '
                  f'in {time.perf_counter() - t0:.1f}s', flush=True)
    with open(args.out, 'w') as f:
        json.dump(dict(meta=meta, block_m=args.block_m, rows=all_rows), f)
    print(f'wrote {args.out} ({len(all_rows)} rows)')
    if args.md:
        with open(args.md, 'w') as f:
            f.write(render_tables(all_rows, meta, args.block_m))
        print(f'wrote {args.md}')


def _layer_job_entry(payload):
    name, chunk, payloads, eps, slots, block_m, opt_mode, time_limit, topk = payload
    prev = {}
    out = []
    for (li, ci, totals) in payloads:
        rows, prev = _case_worker((name, li, ci, totals, eps, slots, block_m, opt_mode, time_limit, prev))
        for r in rows:
            r['chunk_tokens'] = chunk
            r['tokens'] = int(sum(totals)) // topk
        out.extend(rows)
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


_NAMES = {'dsv4_gsm8k': 'DeepSeek V4 Flash GSM8K', 'dsv4_sharegpt': 'DeepSeek V4 Flash ShareGPT',
          'dsv4_gsm8k1k': 'DeepSeek V4 Flash GSM8K 8-shot (1k)', 'dsv4_mmlu1k': 'DeepSeek V4 Flash MMLU 8-shot (1k)',
          'mimo25_gsm8k': 'MiMo 2.5 GSM8K', 'mimo25_sharegpt': 'MiMo 2.5 ShareGPT'}
_SHORT = {'dsv4_gsm8k': 'GSM8K', 'dsv4_sharegpt': 'ShareGPT', 'dsv4_gsm8k1k': 'GSM8K 8-shot', 'dsv4_mmlu1k': 'MMLU 8-shot',
          'mimo25_gsm8k': 'GSM8K', 'mimo25_sharegpt': 'ShareGPT'}


def render_tables(rows, meta, block_m, compact: bool = False) -> str:
    """Markdown tables, one row per (trace, EP, chunk, S). ``compact`` drops
    the chained-planner and LP-bound columns (they agree with the fresh
    planner and OPT to the shown precision) and splits one table per EP."""
    if compact:
        return _render_compact(rows, block_m)
    out = []
    out.append(f'Busiest-rank load in token blocks of {block_m}, mean over layers and chunks; '
               f'gap = (column - ideal) / ideal.\n')
    traces = sorted({r['trace'] for r in rows})
    eps = sorted({r['ep'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    slots = sorted({r['slots'] for r in rows})
    hdr = ('| trace | EP | chunk | tokens/rank | cases | S | ideal | home (no-algo) | gap | planner | gap | '
           'planner (chained) | gap | OPT | gap | OPT proven | LP bound | home/planner | home/OPT | '
           'replicas used | planner at ideal |')
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    out.append(hdr)
    out.append(sep)
    for tr in traces:
        for R in eps:
            for C in chunks:
                for S in slots:
                    sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C and r['slots'] == S]
                    if not sel:
                        continue
                    ideal = _mean([r['ideal'] for r in sel])
                    home = _mean([r['home'] for r in sel])
                    pl = _mean([r['planner'] for r in sel])
                    pc = _mean([r['planner_chained'] for r in sel])
                    opt_vals = [r['opt'] if r['opt'] is not None else r.get('opt_lp') for r in sel]
                    opt = _mean(opt_vals)
                    lp = _mean([r.get('opt_lp') for r in sel])
                    proven = sum(1 for r in sel if r.get('opt_proven'))
                    reps = _mean([r.get('replicas', 0) for r in sel])
                    at_ideal = sum(1 for r in sel if r['planner'] <= math.ceil(r['ideal'] - 1e-9))
                    tok = _mean([r['tokens'] for r in sel]) / R
                    g = lambda v: f'{100 * (v - ideal) / ideal:.1f}%'
                    out.append(f'| {tr} | {R} | {C} | {tok:.0f} | {len(sel)} | {S} | {ideal:.1f} | {home:.1f} | {g(home)} | '
                               f'{pl:.1f} | {g(pl)} | {pc:.1f} | {g(pc)} | {opt:.1f} | {g(opt)} | {proven}/{len(sel)} | '
                               f'{lp:.1f} | {home / pl:.3f} | {home / opt:.3f} | {reps:.1f} | {100 * at_ideal / len(sel):.0f}% |')
    return '\n'.join(out) + '\n'


def _render_compact(rows, block_m) -> str:
    out = []
    traces = sorted({r['trace'] for r in rows})
    eps = sorted({r['ep'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    slots = sorted({r['slots'] for r in rows})
    for R in eps:
        out.append(f'EP{R}: busiest-rank load in token blocks of {block_m}, mean over layers and chunks; '
                   f'gap = (column - ideal) / ideal; "no-algo" and "home" are the same contiguous placement.\n')
        out.append('| trace | chunk | tokens/rank | cases | S | ideal | no-algo = home | gap | planner | gap | OPT | gap | '
                   'OPT proven | home/planner | home/OPT | replicas used (of S x EP) | chunks where planner = ceil(ideal) |')
        out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for tr in traces:
            for C in chunks:
                for S in slots:
                    sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C and r['slots'] == S]
                    if not sel:
                        continue
                    ideal = _mean([r['ideal'] for r in sel])
                    home = _mean([r['home'] for r in sel])
                    pl = _mean([r['planner'] for r in sel])
                    opt = _mean([r['opt'] if r['opt'] is not None else r.get('opt_lp') for r in sel])
                    proven = sum(1 for r in sel if r.get('opt_proven'))
                    reps = _mean([r.get('replicas', 0) for r in sel])
                    at_ideal = sum(1 for r in sel if r['planner'] <= math.ceil(r['ideal'] - 1e-9))
                    tok = _mean([r['tokens'] for r in sel]) / R
                    g = lambda v: f'{100 * (v - ideal) / ideal:.1f}%'
                    name = _NAMES.get(tr, tr)
                    out.append(f'| {name} | {C} | {tok:.0f} | {len(sel)} | {S} | {ideal:.1f} | {home:.1f} | {g(home)} | '
                               f'{pl:.1f} | {g(pl)} | {opt:.1f} | {g(opt)} | {proven}/{len(sel)} | '
                               f'{home / pl:.3f} | {home / opt:.3f} | {reps:.1f} | {100 * at_ideal / len(sel):.0f}% |')
        out.append('')
    return '\n'.join(out) + '\n'


def render_stale(rows, block_m) -> str:
    """Deferred-plan quality: chunk i remapped onto chunk i-1's plan (stale),
    and the incremental-refresh upper bound (union). First chunks of a
    stream are excluded (stale = home there by definition)."""
    out = []
    traces = sorted({r['trace'] for r in rows})
    eps = sorted({r['ep'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    slots = sorted(s for s in {r['slots'] for r in rows} if s > 0)
    for R in eps:
        out.append(f'EP{R}: busiest-rank blocks of {block_m}, mean over layers and chunks (first chunk of every '
                   f'layer stream excluded); gap vs ideal; "within 2%" = chunks where stale <= 1.02 x fresh planner.\n')
        out.append('| trace | chunk | tokens/rank | cases | S | ideal | home | fresh planner | gap | stale plan | gap | '
                   'home/stale | stale within 2% of fresh | stale worse than home | previous replicas, fresh split, no copy | gap | '
                   'home/resplit | stale + up to S new (upper bound) | gap | home/union |')
        out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for tr in traces:
            for C in chunks:
                for S in slots:
                    sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C
                           and r['slots'] == S and r.get('first') is False]
                    if not sel:
                        continue
                    ideal = _mean([r['ideal'] for r in sel])
                    home = _mean([r['home'] for r in sel])
                    pl = _mean([r['planner'] for r in sel])
                    st = _mean([r['stale'] for r in sel])
                    un = _mean([r['union'] for r in sel])
                    rs = _mean([r['resplit'] for r in sel])
                    within = sum(1 for r in sel if r['stale'] <= 1.02 * r['planner'])
                    worse = sum(1 for r in sel if r['stale'] > r['home'])
                    tok = _mean([r['tokens'] for r in sel]) / R
                    g = lambda v: f'{100 * (v - ideal) / ideal:.1f}%'
                    out.append(f'| {_NAMES.get(tr, tr)} | {C} | {tok:.0f} | {len(sel)} | {S} | {ideal:.1f} | {home:.1f} | '
                               f'{pl:.1f} | {g(pl)} | {st:.1f} | {g(st)} | {home / st:.3f} | {100 * within / len(sel):.0f}% | '
                               f'{100 * worse / len(sel):.0f}% | {rs:.1f} | {g(rs)} | {home / rs:.3f} | {un:.1f} | {g(un)} | {home / un:.3f} |')
        out.append('')
    return '\n'.join(out) + '\n'


def cmd_stale(args):
    with open(args.results) as f:
        d = json.load(f)
    sys.stdout.write(render_stale(d['rows'], d['block_m']))


def cmd_tables(args):
    with open(args.results) as f:
        d = json.load(f)
    sys.stdout.write(render_tables(d['rows'], d['meta'], d['block_m'], compact=args.compact))


# --------------------------------------------------------------------------
# calibration against the kernel replay (first window of M tokens per rank)
# --------------------------------------------------------------------------

def cmd_calibrate(args):
    """Home / planner block ratio for the windows the kernel benchmark replays
    (rank r takes tokens [r*M, (r+1)*M) of the layer stream, wrapping) at
    the layer indices the benchmark used, for comparison with its measured
    kernel-only speedups."""
    import numpy as np
    planner = _load_planner_module()
    z = np.load(args.cache)
    hist, num_tokens, W = z['hist'], z['num_tokens'].tolist(), int(z['window'])
    idx = [int(x) for x in args.layers]
    R, S = args.ep, args.slots
    print(f'layers {idx}, EP {R}, S {S}, block_m {args.block_m}')
    for M in args.m:
        need = R * M // W
        vals = []
        for li in idx:
            n_win = -(-num_tokens[li] // W)
            sel = [w % n_win for w in range(need)]
            totals = hist[li, sel].sum(0).tolist()
            b = blocks_of(totals, args.block_m)
            home = max(home_loads(b, R))
            pl = planner_max_load(planner, totals, R, S, args.block_m)[0]
            opt, proven = opt_min_max(b, R, S, ub=pl, time_limit=args.time_limit)
            vals.append((home, pl, opt, sum(b) / R))
        print(f'M={M}: ' + ' / '.join(f'{h / p:.2f}' for h, p, o, i in vals) + '  (home/planner)   ' +
              ' / '.join(f'{h / o:.2f}' for h, p, o, i in vals) + '  (home/OPT)   ' +
              ' / '.join(f'{h / i:.2f}' for h, p, o, i in vals) + '  (home/ideal)')


# --------------------------------------------------------------------------
# static redundancy + per-chunk split (no planner, no copies on the path)
# --------------------------------------------------------------------------

def opt_split_fixed(b: Sequence[int], holders: Sequence[Sequence[int]], R: int) -> int:
    """Exact min-max busiest-rank blocks when the replica set is fixed:
    ``holders[e]`` are the ranks holding expert ``e`` (owner included) and
    its blocks may be split in whole blocks between them. With the set
    fixed this is a transportation problem with integer data, so the LP
    optimum rounded up is the integer optimum."""
    import numpy as np
    from scipy.optimize import LinearConstraint, Bounds, milp
    from scipy.sparse import coo_matrix
    fixed = [0] * R
    var = []  # (e, r) with a choice
    for e, h in enumerate(holders):
        if len(h) == 1:
            fixed[h[0]] += b[e]
        elif b[e] > 0:
            var.extend((e, r) for r in h)
    if not var:
        return max(fixed)
    nv = 1 + len(var)
    rows, cols, vals, lo, hi = [], [], [], [], []
    row = 0
    by_e: Dict[int, List[int]] = {}
    for j, (e, r) in enumerate(var):
        by_e.setdefault(e, []).append(1 + j)
    for e, js in by_e.items():
        for j in js:
            rows.append(row); cols.append(j); vals.append(1.0)
        lo.append(float(b[e])); hi.append(float(b[e])); row += 1
    for r in range(R):
        for j, (e, rr) in enumerate(var):
            if rr == r:
                rows.append(row); cols.append(1 + j); vals.append(1.0)
        rows.append(row); cols.append(0); vals.append(-1.0)
        lo.append(-np.inf); hi.append(float(-fixed[r])); row += 1
    A = coo_matrix((vals, (rows, cols)), shape=(row, nv)).tocsr()
    c = np.zeros(nv); c[0] = 1.0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(np.zeros(nv), np.full(nv, np.inf)),
               integrality=np.zeros(nv), options={'disp': False})
    assert res.x is not None, res.message
    return int(math.ceil(res.x[0] - 1e-6))


def _holders_from_table(table, E: int, R: int) -> List[List[int]]:
    H = E // R
    holders = [[e // H] for e in range(E)]
    for r, row in enumerate(table):
        for e in row:
            if e >= 0 and r not in holders[e]:
                holders[e].append(r)
    return holders


def _static_layer_job(payload):
    """One layer of one trace: replica tables from the first ``frac`` of the
    stream, then every later chunk under every (EP, S, rule, source)."""
    (tr, li, hist_rows, n_tokens, W, eps, slots, chunks, block_m, frac, cross_hist_rows) = payload
    import numpy as np
    planner = _load_planner_module()
    hist = np.asarray(hist_rows)
    n_win = hist.shape[0]
    E = hist.shape[1]
    n25 = int(math.ceil(frac * n_win))
    hist25 = hist[:n25].sum(0).tolist()
    cross25 = np.asarray(cross_hist_rows)[:int(math.ceil(frac * len(cross_hist_rows)))].sum(0).tolist() \
        if cross_hist_rows is not None else None
    rem_tokens = n_tokens - n25 * W
    out = []
    for R in eps:
        tables = {}
        for S in slots:
            counts = [[0] * E for _ in range(R)]
            counts[0] = hist25
            tables[('planner', S, 'same')] = planner.plan_reference(counts, S, block_m).slot_expert
            tables[('eplb', S, 'same')] = planner.static_replica_placement(hist25, R, S)
            if cross25 is not None and S == 8:
                counts[0] = cross25
                tables[('planner', S, 'cross')] = planner.plan_reference(counts, S, block_m).slot_expert
                tables[('eplb', S, 'cross')] = planner.static_replica_placement(cross25, R, S)
        for C in chunks:
            groups = chunk_groups(rem_tokens, W, C)
            for ci, (a, bnd) in enumerate(groups):
                totals = hist[n25 + a:n25 + bnd].sum(0).tolist()
                b = blocks_of(totals, block_m)
                base = home_loads(b, R)
                ideal = sum(b) / R
                fresh = {S: planner_max_load(planner, totals, R, S, block_m)[0] for S in slots}
                for (rule, S, src), table in tables.items():
                    split = planner_max_load(planner, totals, R, S, block_m, prev_slot_expert=table,
                                             open_gate_blocks=1 << 30)[0]
                    exact = opt_split_fixed(b, _holders_from_table(table, E, R), R)
                    out.append(dict(trace=tr, layer=li, chunk=ci, chunk_tokens=C, ep=R, slots=S, rule=rule, src=src,
                                    tokens=int(sum(totals)), ideal=ideal, home=max(base), fresh=fresh[S],
                                    split=split, exact=exact,
                                    replicas=sum(1 for row in table for e in row if e >= 0)))
    return out


def cmd_static(args):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    caches = {}
    for spec in args.cache:
        name, path = spec.split('=', 1)
        caches[name] = np.load(path)
    cross = {}
    for spec in args.cross or []:
        a, bname = spec.split(':', 1)
        cross[a] = bname
    jobs = []
    topk = {}
    for name, z in caches.items():
        hist, num_tokens, W = z['hist'], z['num_tokens'].tolist(), int(z['window'])
        topk[name] = int(z['num_topk'])
        other = caches.get(cross.get(name)) if name in cross else None
        for li in range(hist.shape[0]):
            cr = other['hist'][li].tolist() if other is not None else None
            jobs.append((name, li, hist[li].tolist(), num_tokens[li], W, args.ep, args.slots, args.chunks,
                         args.block_m, args.frac, cr))
    rows = []
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_static_layer_job, jobs):
                rows.extend(res)
    else:
        for j in jobs:
            rows.extend(_static_layer_job(j))
    for r in rows:
        r['tokens'] //= topk[r['trace']]
    print(f'{len(rows)} rows in {time.perf_counter() - t0:.1f}s', file=sys.stderr)
    with open(args.out, 'w') as f:
        json.dump(dict(rows=rows, frac=args.frac, block_m=args.block_m), f)
    sys.stdout.write(render_static(rows, args.block_m, args.frac))


def render_static(rows, block_m, frac) -> str:
    out = []
    traces = sorted({r['trace'] for r in rows})
    eps = sorted({r['ep'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    for rule, title in (('planner', "replica set = the planner's replica choice on the first-quarter histogram"),
                        ('eplb', "replica set = static_replica_placement (EPLB-style, load per copy) on the first-quarter histogram")):
        for R in eps:
            out.append(f'EP{R}, {title}; busiest-rank blocks of {block_m}, mean over layers and the chunks of the last '
                       f'{100 * (1 - frac):.0f} % of each layer stream; gap vs ideal; "within 2%" = chunks where the '
                       f'static split <= 1.02 x the fresh planner with the same S.\n')
            out.append('| trace | replicas from | chunk | tokens/rank | cases | S | replicas used (of S x EP) | ideal | home | '
                       'fresh planner | gap | static set + per-chunk split | gap | home/static-split | within 2% of fresh | '
                       'exact split, same set | gap | home/exact |')
            out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for tr in traces:
                for src in ('same', 'cross'):
                    for C in chunks:
                        for S in sorted({r['slots'] for r in rows}):
                            sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C
                                   and r['slots'] == S and r['rule'] == rule and r['src'] == src]
                            if not sel:
                                continue
                            ideal = _mean([r['ideal'] for r in sel])
                            home = _mean([r['home'] for r in sel])
                            fr = _mean([r['fresh'] for r in sel])
                            sp = _mean([r['split'] for r in sel])
                            ex = _mean([r['exact'] for r in sel])
                            within = sum(1 for r in sel if r['split'] <= 1.02 * r['fresh'])
                            reps = _mean([r['replicas'] for r in sel])
                            tok = _mean([r['tokens'] for r in sel]) / R
                            g = lambda v: f'{100 * (v - ideal) / ideal:.1f}%'
                            other = {'dsv4_gsm8k': 'ShareGPT', 'dsv4_sharegpt': 'GSM8K'}.get(tr, 'other')
                            label = 'same stream, first 25 %' if src == 'same' else f'{other} stream, first 25 %'
                            out.append(f'| {_NAMES.get(tr, tr)} | {label} | {C} | {tok:.0f} | {len(sel)} | {S} | {reps:.1f} | '
                                       f'{ideal:.1f} | {home:.1f} | {fr:.1f} | {g(fr)} | {sp:.1f} | {g(sp)} | {home / sp:.3f} | '
                                       f'{100 * within / len(sel):.0f}% | {ex:.1f} | {g(ex)} | {home / ex:.3f} |')
            out.append('')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------
# permute: owner permutation (which rank owns which expert), zero per-call cost
# --------------------------------------------------------------------------

def _perm_max_load(planner, totals, perm, R: int, block_m: int) -> int:
    """Busiest-rank blocks of the contiguous placement after relabeling the
    experts by ``perm`` (no replicas)."""
    return max(home_loads(blocks_of(planner.apply_owner_permutation(totals, perm), block_m), R))


def _count_replicas(table) -> int:
    return sum(1 for row in table for e in row if e >= 0)


def _permute_layer_job(payload):
    """One layer of one trace: owner permutation from the first ``frac`` of
    the stream (and from the other dataset's first ``frac``), then every
    later chunk under home / permutation / planner / permutation + planner /
    permutation + EPLB-style static set + per-chunk split, plus a
    permutation recomputed from the previous chunk of the stream."""
    (tr, li, hist_rows, n_tokens, W, eps, slots, chunks, block_m, frac, cross_hist_rows) = payload
    import numpy as np
    planner = _load_planner_module()
    hist = np.asarray(hist_rows)
    n_win, E = hist.shape
    n25 = int(math.ceil(frac * n_win))
    hist25 = hist[:n25].sum(0).tolist()
    # cross_hist_rows: {other trace name: that trace's windows of this layer}.
    cross25 = {o: np.asarray(rows_)[:int(math.ceil(frac * len(rows_)))].sum(0).tolist()
               for o, rows_ in (cross_hist_rows or {}).items()}
    rem_tokens = n_tokens - n25 * W
    out = []
    for R in eps:
        perm_same = planner.owner_permutation(hist25, R, block_m)
        perm_cross = {o: planner.owner_permutation(c, R, block_m) for o, c in cross25.items()}
        hist25_p = planner.apply_owner_permutation(hist25, perm_same)
        # Two EPLB-style static sets on the permuted layout: the gated rule
        # (opens a copy only while it lowers the holders below the receiver)
        # and the ungated fill of every slot by load per copy.
        tables = {S: planner.static_replica_placement(hist25_p, R, S) for S in slots}
        tables_full = {S: planner._fill_by_frequency(hist25_p, R, S) for S in slots}
        for C in chunks:
            per = C // W
            groups = chunk_groups(rem_tokens, W, C)
            for ci, (a, bnd) in enumerate(groups):
                totals = hist[n25 + a:n25 + bnd].sum(0).tolist()
                b = blocks_of(totals, block_m)
                base = home_loads(b, R)
                ideal = sum(b) / R
                # The chunk of C tokens preceding this one in the stream (the
                # tail of the calibration part for the first evaluation chunk).
                prev = hist[max(0, n25 + a - per):n25 + a].sum(0).tolist()
                perm_prev = planner.owner_permutation(prev, R, block_m)
                totals_p = planner.apply_owner_permutation(totals, perm_same)
                common = dict(trace=tr, layer=li, chunk=ci, chunk_tokens=C, ep=R, tokens=int(sum(totals)),
                              ideal=ideal, home=max(base),
                              perm=max(home_loads(blocks_of(totals_p, block_m), R)),
                              perm_prev=_perm_max_load(planner, totals, perm_prev, R, block_m),
                              perm_moved=sum(1 for p_, e in enumerate(perm_same) if p_ // (E // R) != e // (E // R)))
                for o, pc in perm_cross.items():
                    common['perm_cross:' + o] = _perm_max_load(planner, totals, pc, R, block_m)
                for S in slots:
                    rec = dict(common, slots=S)
                    p_home, slot_home, _ = planner_max_load(planner, totals, R, S, block_m)
                    p_perm, slot_perm, _ = planner_max_load(planner, totals_p, R, S, block_m)
                    rec.update(planner=p_home, replicas=_count_replicas(slot_home),
                               perm_planner=p_perm, perm_replicas=_count_replicas(slot_perm))
                    rec['perm_prev_planner'] = planner_max_load(
                        planner, planner.apply_owner_permutation(totals, perm_prev), R, S, block_m)[0]
                    for o, pc in perm_cross.items():
                        rec['perm_cross_planner:' + o] = planner_max_load(
                            planner, planner.apply_owner_permutation(totals, pc), R, S, block_m)[0]
                    table = tables[S]
                    rec['perm_static_split'] = planner_max_load(planner, totals_p, R, S, block_m,
                                                                prev_slot_expert=table, open_gate_blocks=1 << 30)[0]
                    rec['perm_static_exact'] = opt_split_fixed(blocks_of(totals_p, block_m),
                                                               _holders_from_table(table, E, R), R)
                    rec['static_replicas'] = _count_replicas(table)
                    full = tables_full[S]
                    rec['perm_full_split'] = planner_max_load(planner, totals_p, R, S, block_m,
                                                              prev_slot_expert=full, open_gate_blocks=1 << 30)[0]
                    rec['perm_full_exact'] = opt_split_fixed(blocks_of(totals_p, block_m),
                                                             _holders_from_table(full, E, R), R)
                    rec['full_replicas'] = _count_replicas(full)
                    out.append(rec)
    return out


def cmd_permute(args):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    caches = {}
    for spec in args.cache:
        name, path = spec.split('=', 1)
        caches[name] = np.load(path)
    cross: Dict[str, List[str]] = {}
    for spec in args.cross or []:
        a, bname = spec.split(':', 1)
        assert bname in caches, f'--cross {spec}: {bname} is not a loaded cache'
        cross.setdefault(a, []).append(bname)
    jobs = []
    topk = {}
    for name, z in caches.items():
        hist, num_tokens, W = z['hist'], z['num_tokens'].tolist(), int(z['window'])
        topk[name] = int(z['num_topk'])
        for li in range(hist.shape[0]):
            cr = {o: caches[o]['hist'][li].tolist() for o in cross.get(name, [])}
            jobs.append((name, li, hist[li].tolist(), num_tokens[li], W, args.ep, args.slots, args.chunks,
                         args.block_m, args.frac, cr))
    rows = []
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_permute_layer_job, jobs):
                rows.extend(res)
    else:
        for j in jobs:
            rows.extend(_permute_layer_job(j))
    for r in rows:
        r['tokens'] //= topk[r['trace']]
    print(f'{len(rows)} rows in {time.perf_counter() - t0:.1f}s', file=sys.stderr)
    with open(args.out, 'w') as f:
        json.dump(dict(rows=rows, frac=args.frac, block_m=args.block_m), f)
    sys.stdout.write(render_permute(rows, args.block_m, args.frac))


def _percentiles(xs, qs=(10, 50, 90)):
    xs = sorted(xs)
    out = []
    for q in qs:
        k = (len(xs) - 1) * q / 100.0
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        out.append(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))
    return out


def render_permute(rows, block_m, frac) -> str:
    out = []
    traces = sorted({r['trace'] for r in rows})
    eps = sorted({r['ep'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    slots = sorted({r['slots'] for r in rows})
    for R in eps:
        out.append(f'EP{R}: busiest-rank blocks of {block_m}, mean over layers and the chunks of the last '
                   f'{100 * (1 - frac):.0f} % of each layer stream; gap vs ideal; "closed" = share of the '
                   f'home-to-ideal gap the arm removes, (home - arm) / (home - ideal); replicas = helper slots in '
                   f'use, summed over ranks.\n')
        out.append('| trace | permutation from | chunk | tokens/rank | cases | S | ideal | home | gap | '
                   'permutation only | gap | closed | home/perm | home + planner | gap | replicas | '
                   'permutation + planner | gap | replicas | permutation + EPLB set (gated) + per-chunk split | gap | replicas | '
                   'exact split, same set | gap | permutation + EPLB set (every slot) + per-chunk split | gap | replicas | '
                   'exact split, same set | gap |')
        out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for tr in traces:
            cross_names = sorted({k.split(':', 1)[1] for r in rows if r['trace'] == tr
                                  for k in r if k.startswith('perm_cross:')})
            srcs = [('same', 'perm', 'perm_planner', 'same stream, first %.0f %%' % (100 * frac))]
            srcs += [('cross', 'perm_cross:' + o, 'perm_cross_planner:' + o,
                      f'{_SHORT.get(o, o)} stream, first {100 * frac:.0f} %') for o in cross_names]
            srcs += [('prev', 'perm_prev', 'perm_prev_planner', 'previous chunk of the stream')]
            for src, key, key_pl, label in srcs:
                for C in chunks:
                    for S in slots:
                        sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C
                               and r['slots'] == S and key in r]
                        if not sel:
                            continue
                        ideal = _mean([r['ideal'] for r in sel])
                        home = _mean([r['home'] for r in sel])
                        pm = _mean([r[key] for r in sel])
                        pl = _mean([r['planner'] for r in sel])
                        pp = _mean([r[key_pl] for r in sel])
                        reps = _mean([r['replicas'] for r in sel])
                        preps = _mean([r['perm_replicas'] for r in sel]) if src == 'same' else None
                        tok = _mean([r['tokens'] for r in sel]) / R
                        g = lambda v: f'{100 * (v - ideal) / ideal:.1f}%'
                        closed = f'{100 * (home - pm) / (home - ideal):.0f}%'
                        if src == 'same':
                            ss = _mean([r['perm_static_split'] for r in sel])
                            se = _mean([r['perm_static_exact'] for r in sel])
                            sreps = _mean([r['static_replicas'] for r in sel])
                            fs = _mean([r['perm_full_split'] for r in sel])
                            fe = _mean([r['perm_full_exact'] for r in sel])
                            freps = _mean([r['full_replicas'] for r in sel])
                            tail = (f'{pp:.1f} | {g(pp)} | {preps:.1f} | {ss:.1f} | {g(ss)} | {sreps:.1f} | {se:.1f} | {g(se)} | '
                                    f'{fs:.1f} | {g(fs)} | {freps:.1f} | {fe:.1f} | {g(fe)} |')
                        else:
                            tail = f'{pp:.1f} | {g(pp)} | - | - | - | - | - | - | - | - | - | - | - |'
                        out.append(f'| {_NAMES.get(tr, tr)} | {label} | {C} | {tok:.0f} | {len(sel)} | {S} | {ideal:.1f} | '
                                   f'{home:.1f} | {g(home)} | {pm:.1f} | {g(pm)} | {closed} | {home / pm:.3f} | '
                                   f'{pl:.1f} | {g(pl)} | {reps:.1f} | {tail}')
        out.append('')
        out.append(f'EP{R}: per-chunk spread of busiest rank / ideal (p10 / median / p90 over layers and chunks), '
                   f'and the share of chunks where the permutation alone is worse than home. S = {slots[0]} for the '
                   f'planner arms.\n')
        out.append('| trace | chunk | cases | home | permutation (same stream) | permutation (other stream) | '
                   'permutation (previous chunk) | home + planner | permutation + planner | '
                   'permutation + EPLB set (gated) + split | permutation + EPLB set (every slot) + split | '
                   'permutation worse than home | experts moved by the permutation (of E) |')
        out.append('|---|---:|---:|---|---|---|---|---|---|---|---|---:|---:|')
        for tr in traces:
            for C in chunks:
                sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C
                       and r['slots'] == slots[0]]
                if not sel:
                    continue

                def cell(k):
                    vals = [r[k] / r['ideal'] for r in sel if k in r]
                    if not vals:
                        return '-'
                    return ' / '.join(f'{v:.2f}' for v in _percentiles(vals))
                cross_keys = sorted({k for r in sel for k in r if k.startswith('perm_cross:')})
                cross_cell = '; '.join(f'{_SHORT.get(k.split(":", 1)[1], k)}: {cell(k)}' for k in cross_keys) or '-'
                worse = sum(1 for r in sel if r['perm'] > r['home'])
                moved = _mean([r['perm_moved'] for r in sel])
                out.append(f'| {_NAMES.get(tr, tr)} | {C} | {len(sel)} | {cell("home")} | {cell("perm")} | '
                           f'{cross_cell} | {cell("perm_prev")} | {cell("planner")} | {cell("perm_planner")} | '
                           f'{cell("perm_static_split")} | {cell("perm_full_split")} | {100 * worse / len(sel):.0f}% | {moved:.0f} |')
        out.append('')
    return '\n'.join(out) + '\n'


def cmd_permute_tables(args):
    with open(args.results) as f:
        d = json.load(f)
    sys.stdout.write(render_permute(d['rows'], d['block_m'], d['frac']))


# --------------------------------------------------------------------------
# budget: load ratios -> time ceiling per layer and chunk
# --------------------------------------------------------------------------

# Fixed-placement kernel time per MoE layer at EP8 (us, rank maximum, real
# GSM8K routing), from PROFILE_eplb_stages.md (EP8 table, "kernel only" /
# fresh rows, GPU MoE column). Entries marked interpolated are a linear fit
# through the measured M = 1024 and 4096 points.
_KERNEL_US_EP8 = {
    # model: {tokens per rank: (us, source)}
    'dsv4': {512: (460, 'interpolated'), 1024: (726, 'measured'), 2048: (1258, 'measured'), 4096: (2343, 'measured')},
    'mimo25': {512: (820, 'interpolated'), 1024: (1375, 'measured'), 2048: (2490, 'interpolated; in-server fixed arm 2856-2890'),
               4096: (4719, 'measured')},
}
# Planner kernel on the critical path at EP8, real routing (GPU plan column of
# the same table): DeepSeek shape 76-110 us, MiMo shape 120-168 us.
_PLANNER_US_EP8 = {'dsv4': (76, 110), 'mimo25': (120, 168)}
# Measured synchronous prepare (plan + copy + remap) inside the server at
# 16k chunks, MiMo: 351 us (design note, 2026-10-15).
_PREPARE_IN_SERVER_US = 351


def cmd_budget(args):
    with open(args.results) as f:
        d = json.load(f)
    rows = d['rows']
    traces = sorted({r['trace'] for r in rows})
    chunks = sorted({r['chunk_tokens'] for r in rows})
    slots = sorted(s for s in {r['slots'] for r in rows} if s > 0)
    R = args.ep
    out = []
    out.append(f'Time ceiling per MoE layer and chunk at EP{R}. Ratios are means over layers and chunks of the '
               f'busiest-rank blocks (home / planner, home / OPT, ratio of the means). The saving is the '
               f'fixed-placement kernel time times (1 - 1/ratio); the kernel realizes about 92-98% of the block '
               f'ratio (calibration), so these are upper bounds. Net = saving minus the planner kernel '
               f'(76-110 us DeepSeek shape, 120-168 us MiMo shape at EP8); break-even = the planner cost at which '
               f'the net gain is zero.\n')
    out.append('| trace | chunk | tokens/rank | kernel us (fixed) | S | home/planner | home/OPT | '
               'max saving us (planner) | max saving us (OPT) | net us after planner kernel | net % of layer | '
               'break-even planner cost us (planner / OPT) | net after in-server prepare 351 us |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for tr in traces:
        model = 'mimo25' if tr.startswith('mimo') else 'dsv4'
        for C in chunks:
            tpr = C // R
            k = _KERNEL_US_EP8[model].get(tpr) if R == 8 else None
            for S in slots:
                sel = [r for r in rows if r['trace'] == tr and r['ep'] == R and r['chunk_tokens'] == C and r['slots'] == S]
                if not sel:
                    continue
                home = _mean([r['home'] for r in sel])
                pl = _mean([r['planner'] for r in sel])
                opt = _mean([r['opt'] if r['opt'] is not None else r.get('opt_lp') for r in sel])
                rp, ro = home / pl, home / opt
                if k is None:
                    out.append(f'| {tr} | {C} | {tpr} | n/a | {S} | {rp:.3f} | {ro:.3f} | '
                               f'{100 * (1 - 1 / rp):.1f}% of kernel | {100 * (1 - 1 / ro):.1f}% of kernel | | | | |')
                    continue
                kus, src = k
                sp, so = kus * (1 - 1 / rp), kus * (1 - 1 / ro)
                lo, hi = _PLANNER_US_EP8[model]
                net_lo, net_hi = sp - hi, sp - lo
                out.append(f'| {tr} | {C} | {tpr} | {kus} ({src}) | {S} | {rp:.3f} | {ro:.3f} | {sp:.0f} | {so:.0f} | '
                           f'{net_lo:.0f} to {net_hi:.0f} | {100 * net_lo / kus:.1f}% to {100 * net_hi / kus:.1f}% | '
                           f'{sp:.0f} / {so:.0f} | {sp - _PREPARE_IN_SERVER_US:.0f} |')
    sys.stdout.write('\n'.join(out) + '\n')


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _brute_force_opt(b, R, S):
    """Exhaustive optimum for tiny instances: enumerate replica sets, then the
    block split by max-flow (scipy)."""
    import itertools
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    E = len(b)
    H = E // R

    def feasible(T, holders):
        # nodes: source 0, experts 1..E, ranks E+1..E+R, sink E+R+1
        n = E + R + 2
        rows, cols, caps = [], [], []
        for e in range(E):
            rows.append(0); cols.append(1 + e); caps.append(b[e])
            for r in holders[e]:
                rows.append(1 + e); cols.append(1 + E + r); caps.append(b[e])
        for r in range(R):
            rows.append(1 + E + r); cols.append(n - 1); caps.append(T)
        g = csr_matrix((caps, (rows, cols)), shape=(n, n), dtype=np.int32)
        return maximum_flow(g, 0, n - 1).flow_value == sum(b)

    best = None
    cand = [(e, r) for e in range(E) for r in range(R) if r != e // H]
    for k in range(0, R * S + 1):
        for combo in itertools.combinations(cand, k):
            per_rank = [0] * R
            ok = True
            for (e, r) in combo:
                per_rank[r] += 1
                if per_rank[r] > S:
                    ok = False
                    break
            if not ok:
                continue
            holders = [{e // H} for e in range(E)]
            for (e, r) in combo:
                holders[e].add(r)
            lo, hi = -(-sum(b) // R), max(home_loads(b, R))
            while lo < hi:
                mid = (lo + hi) // 2
                if feasible(mid, holders):
                    hi = mid
                else:
                    lo = mid + 1
            best = lo if best is None else min(best, lo)
    return best


def cmd_selftest(args):
    import random
    planner = _load_planner_module()
    random.seed(0)
    # 1. OPT (MILP) equals brute force on tiny instances.
    for trial in range(40):
        R = random.choice([2, 3])
        H = random.choice([2, 3])
        S = random.choice([1, 2])
        b = [random.choice([0, 1, 2, 3, 5, 8, 13]) for _ in range(R * H)]
        if sum(b) == 0:
            continue
        v, proven = opt_min_max(b, R, S)
        bf = _brute_force_opt(b, R, S)
        assert proven and v == bf, f'MILP {v} (proven {proven}) != brute force {bf} for b={b} R={R} S={S}'
    print('selftest 1: MILP optimum matches brute force on 40 tiny instances')
    # 2. Ordering: ideal <= LP <= OPT <= planner <= home on random larger instances.
    for trial in range(20):
        R, H, S = 4, 16, random.choice([2, 4])
        b = [max(0, int(random.paretovariate(1.2) * 2)) for _ in range(R * H)]
        totals = [x * 64 for x in b]
        home = max(home_loads(b, R))
        pl = planner_max_load(planner, totals, R, S, 64)[0]
        lp, _ = opt_min_max(b, R, S, ub=pl, relax=True)
        v, proven = opt_min_max(b, R, S, ub=pl)
        ideal = sum(b) / R
        assert ideal - 1e-9 <= lp <= v + 1e-9 <= pl + 1e-9 <= home + 1e-9, (ideal, lp, v, pl, home)
        assert proven
    print('selftest 2: ideal <= LP <= OPT <= planner <= home on 20 random instances')
    # 3. Chunk grouping.
    assert chunk_groups(31471, 512, 32768) == [(0, 62)]
    g = chunk_groups(31471, 512, 4096)
    assert g[0] == (0, 8) and g[-1] == (56, 62) and len(g) == 8, g
    g = chunk_groups(321539, 512, 4096)
    assert len(g) == 79 and g[-1] == (624, 629), (len(g), g[-1])  # 2051-token tail is kept (>= half)
    g = chunk_groups(31471, 512, 16384)
    assert g == [(0, 32), (32, 62)], g
    print('selftest 3: chunk grouping')
    # 4. Planner with S=0 equals home.
    b = [random.randint(0, 20) for _ in range(64)]
    pl = planner_max_load(planner, [x * 64 for x in b], 8, 0, 64)[0]
    assert pl == max(home_loads(b, 8))
    print('selftest 4: planner with S=0 equals home placement')
    # 5. Owner permutation: a permutation with H experts per rank, never worse
    #    than home on its own histogram, deterministic, identity on a flat
    #    histogram; apply_owner_permutation + plan_reference(S=0) reproduces it.
    for trial in range(30):
        R, H = random.choice([2, 4, 8]), random.choice([4, 16, 32])
        E = R * H
        totals = [max(0, int(random.paretovariate(1.2) * 64)) for _ in range(E)]
        perm = planner.owner_permutation(totals, R, 64)
        assert sorted(perm) == list(range(E)), 'not a permutation'
        assert perm == planner.owner_permutation(totals, R, 64), 'not deterministic'
        assert all(perm[r * H:(r + 1) * H] == sorted(perm[r * H:(r + 1) * H]) for r in range(R))
        b = blocks_of(totals, 64)
        home = max(home_loads(b, R))
        pm = max(home_loads(blocks_of(planner.apply_owner_permutation(totals, perm), 64), R))
        assert pm <= home, (pm, home)
        counts = [[0] * E for _ in range(R)]
        counts[0] = totals
        cp = planner.apply_owner_permutation(counts, perm)
        assert cp[0] == [totals[e] for e in perm] and all(row == [0] * E for row in cp[1:])
        assert max(planner.plan_reference(cp, 0, 64).loads) == pm
        # Nested input sums over ranks.
        assert planner.owner_permutation(counts, R, 64) == perm
    assert planner.owner_permutation([7] * 32, 4, 64) == list(range(32))
    assert planner.owner_permutation([0] * 32, 8, 64) == list(range(32))
    # One hot expert per rank pair: greedy must spread them, one per rank.
    hot = [0] * 16
    hot[0] = hot[1] = hot[2] = hot[3] = 64 * 10
    perm = planner.owner_permutation(hot, 4, 64)
    assert max(home_loads(blocks_of(planner.apply_owner_permutation(hot, perm), 64), 4)) == 10
    print('selftest 5: owner permutation on 30 random instances')
    print('selftest passed')


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache'); c.add_argument('trace'); c.add_argument('out')
    c.add_argument('--window', type=int, default=512); c.set_defaults(fn=cmd_cache)
    s = sub.add_parser('study')
    s.add_argument('--cache', action='append', required=True, help='name=path.npz (repeatable)')
    s.add_argument('--out', default='prefill_headroom_results.json')
    s.add_argument('--md', default=None)
    s.add_argument('--ep', type=int, nargs='+', default=[4, 8])
    s.add_argument('--chunks', type=int, nargs='+', default=[4096, 16384, 32768])
    s.add_argument('--slots', type=int, nargs='+', default=[0, 4, 8])
    s.add_argument('--block-m', type=int, default=64)
    s.add_argument('--max-chunks-per-layer', type=int, default=0,
                   help='subsample long streams to this many chunks per layer (evenly spaced); 0 = all')
    s.add_argument('--opt', choices=['exact', 'lp', 'none'], default='exact')
    s.add_argument('--time-limit', type=float, default=60.0, help='seconds per MILP')
    s.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.set_defaults(fn=run_study)
    t = sub.add_parser('tables'); t.add_argument('results'); t.add_argument('--compact', action='store_true')
    t.set_defaults(fn=cmd_tables)
    k = sub.add_parser('calibrate'); k.add_argument('cache'); k.add_argument('--layers', nargs='+', required=True)
    k.add_argument('--ep', type=int, default=8); k.add_argument('--slots', type=int, default=4)
    k.add_argument('--m', type=int, nargs='+', default=[1024, 4096]); k.add_argument('--block-m', type=int, default=64)
    k.add_argument('--time-limit', type=float, default=60.0); k.set_defaults(fn=cmd_calibrate)
    sl = sub.add_parser('stale'); sl.add_argument('results'); sl.set_defaults(fn=cmd_stale)
    sc = sub.add_parser('static')
    sc.add_argument('--cache', action='append', required=True, help='name=path.npz (repeatable)')
    sc.add_argument('--cross', action='append', default=None, help='name:othername, replicas also taken from the other trace (S=8)')
    sc.add_argument('--ep', type=int, nargs='+', default=[4, 8])
    sc.add_argument('--chunks', type=int, nargs='+', default=[4096, 16384, 32768])
    sc.add_argument('--slots', type=int, nargs='+', default=[4, 8, 16])
    sc.add_argument('--block-m', type=int, default=64)
    sc.add_argument('--frac', type=float, default=0.25, help='share of each layer stream used to choose the replicas')
    sc.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sc.add_argument('--out', default='prefill_static_results.json')
    sc.set_defaults(fn=cmd_static)
    pm = sub.add_parser('permute')
    pm.add_argument('--cache', action='append', required=True, help='name=path.npz (repeatable)')
    pm.add_argument('--cross', action='append', default=None, help='name:othername, permutation also taken from the other trace')
    pm.add_argument('--ep', type=int, nargs='+', default=[4, 8])
    pm.add_argument('--chunks', type=int, nargs='+', default=[4096, 16384, 32768])
    pm.add_argument('--slots', type=int, nargs='+', default=[4])
    pm.add_argument('--block-m', type=int, default=64)
    pm.add_argument('--frac', type=float, default=0.25, help='share of each layer stream used to choose the permutation')
    pm.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pm.add_argument('--out', default='prefill_permute_results.json')
    pm.set_defaults(fn=cmd_permute)
    pt = sub.add_parser('permute-tables'); pt.add_argument('results'); pt.set_defaults(fn=cmd_permute_tables)
    bg = sub.add_parser('budget'); bg.add_argument('results'); bg.add_argument('--ep', type=int, default=8)
    bg.set_defaults(fn=cmd_budget)
    st = sub.add_parser('selftest'); st.set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
