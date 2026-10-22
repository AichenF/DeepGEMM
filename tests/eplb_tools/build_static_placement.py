"""Build the resident rank-permutation placement file for SGLang from routing histograms.

The MegaMoE kernel owns experts contiguously: physical expert ``p`` lives on
rank ``p // H`` (``H = E / EP``). ``static_placement`` in
``deep_gemm/mega/eplb_planner.py`` chooses, once per layer, which logical
expert each physical resident slot holds so that the per-rank block cost of a
calibration histogram is level (PREFILL_headroom_study.md section 10). This
script turns that choice into the two files the serving path needs:

* the expert-location file SGLang's ``--init-expert-location`` loader accepts
  (``compute_initial_expert_location_metadata`` -> ``init_by_mapping``): a JSON
  object with the single key ``physical_to_logical_map``, one row per model
  layer (``num_hidden_layers`` rows, not only the MoE layers), each row the
  permutation ``perm[p] = logical expert in physical slot p``; layers without
  calibration data (the dense or hash-routed first layers) get the identity.
  With no redundant experts ``num_physical == num_logical``, SGLang's weight
  loader then puts logical expert ``perm[p]`` into physical slot ``p`` on rank
  ``p // H`` and, with ``--ep-dispatch-algorithm static`` (the default once a
  location is given), top-k returns physical ids, so the MegaMoE kernel and
  the balancer see the permuted layout without any change;
* a companion JSON with the per-layer permutation and its calibration summary
  (``SGLANG_MEGAMOE_STATIC_PLACEMENT``), which the adapter uses to cross-check the
  location SGLang actually loaded and which documents the choice.

Calibration follows the study's protocol: per layer, the first ``--frac``
(default 25 %) of the 512-token windows of each histogram cache
(``prefill_headroom_study.py cache``). Several caches are combined by
normalising each cache's calibration histogram to the same total weight (the
mean of their totals, so the counts stay in token-slot units for the block
rounding) and summing, which is what a server that must carry both traffics
should be calibrated on. ``--predict`` evaluates the chosen permutation on the
remaining 75 % of every cache's stream, chunk by chunk, with the study's
busiest-rank block cost (resident versus permutation versus ideal).

Usage::

    python build_static_placement.py --cache gsm8k1k=<npz> --cache mmlu1k=<npz> \
        --ep 8 --num-layers 43 --out-location dsv4_static_placement_ep8.json \
        --out-check dsv4_static_placement_ep8_layers.json \
        [--predict 16384 32768] [--predict-ep 8 4] [--out-predict predict.json]

Pure Python on the histograms (numpy only), deterministic, so the file built
here equals the one every rank would derive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLANNER_PATH = os.path.join(_HERE, '..', '..', 'deep_gemm', 'mega', 'eplb_planner.py')


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_planner():
    """``deep_gemm.mega.eplb_planner`` without importing ``deep_gemm`` (which
    needs torch and the compiled extension)."""
    if os.path.exists(_PLANNER_PATH):
        return _load_module('eplb_planner_ref', _PLANNER_PATH)
    from deep_gemm.mega import eplb_planner  # type: ignore
    return eplb_planner


def chunk_groups(n_tokens: int, window: int, chunk: int):
    """Same chunk rule as ``prefill_headroom_study.chunk_groups``: consecutive
    chunks of ``chunk`` tokens over the windows, a trailing partial chunk kept
    when it holds at least half a chunk, a stream shorter than the chunk taken
    whole."""
    study = os.path.join(_HERE, 'prefill_headroom_study.py')
    if os.path.exists(study):
        return _load_module('prefill_headroom_study_ref', study).chunk_groups(n_tokens, window, chunk)
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


def blocks_of(totals: Sequence[int], block_m: int) -> List[int]:
    return [-(-int(t) // block_m) for t in totals]


def home_loads(b: Sequence[int], R: int) -> List[int]:
    H = len(b) // R
    return [sum(b[r * H:(r + 1) * H]) for r in range(R)]


def load_caches(specs: Sequence[str]):
    import numpy as np
    caches = {}
    for spec in specs:
        name, path = spec.split('=', 1)
        z = np.load(path)
        caches[name] = dict(hist=z['hist'], num_tokens=[int(x) for x in z['num_tokens']],
                            window=int(z['window']), num_experts=int(z['num_experts']),
                            num_topk=int(z['num_topk']), layer_ids=[int(x) for x in z['layer_ids']])
    return caches


def calibration_histograms(caches: Dict[str, dict], frac: float):
    """Per cache: {layer_id: (counts[E] of the first ``frac`` of the layer's
    windows, number of calibration windows)}."""
    out = {}
    for name, c in caches.items():
        W = c['window']
        per_layer = {}
        for li, layer_id in enumerate(c['layer_ids']):
            n_win = -(-c['num_tokens'][li] // W)
            n25 = int(math.ceil(frac * n_win))
            per_layer[layer_id] = (c['hist'][li, :n25].sum(0).astype('int64').tolist(), n25)
        out[name] = per_layer
    return out


def combine(cals: Dict[str, dict], layer_id: int) -> List[int]:
    """Equal-weight sum of the caches' calibration histograms of one layer:
    each normalised to the mean of their totals, rounded to integers."""
    rows = [cals[n][layer_id][0] for n in cals if layer_id in cals[n]]
    if len(rows) == 1:
        return rows[0]
    totals = [float(sum(r)) for r in rows]
    scale = sum(totals) / len(totals)
    E = len(rows[0])
    combined = [0.0] * E
    for r, t in zip(rows, totals):
        if t <= 0:
            continue
        f = scale / t
        for e in range(E):
            combined[e] += r[e] * f
    return [int(round(x)) for x in combined]


def predict(caches, cals, perms_by_ep, frac, block_m, chunks, planner):
    """Busiest-rank block cost on the last ``1 - frac`` of every cache's
    stream: resident (contiguous) versus the given permutation versus ideal, mean
    over layers and chunks, plus per-chunk spread. ``perms_by_ep`` maps
    EP -> {layer_id: perm}."""
    import numpy as np
    rows = []
    for name, c in caches.items():
        W = c['window']
        for R, perms in perms_by_ep.items():
            for C in chunks:
                home_all, perm_all, ideal_all, worse = [], [], [], 0
                ratios_home, ratios_perm = [], []
                for li, layer_id in enumerate(c['layer_ids']):
                    n25 = cals[name][layer_id][1]
                    rem_tokens = c['num_tokens'][li] - n25 * W
                    hist = c['hist'][li]
                    perm = perms[layer_id]
                    for a, bnd in chunk_groups(rem_tokens, W, C):
                        totals = hist[n25 + a:n25 + bnd].sum(0).tolist()
                        b = blocks_of(totals, block_m)
                        resident = max(home_loads(b, R))
                        pm = max(home_loads(planner.apply_static_placement(b, perm), R))
                        ideal = sum(b) / R
                        home_all.append(resident)
                        perm_all.append(pm)
                        ideal_all.append(ideal)
                        ratios_home.append(resident / ideal)
                        ratios_perm.append(pm / ideal)
                        worse += pm > resident
                n = len(home_all)
                mh, mp, mi = (sum(home_all) / n, sum(perm_all) / n, sum(ideal_all) / n)
                rows.append(dict(trace=name, ep=R, chunk=C, cases=n,
                                 tokens_per_rank=C // R,
                                 ideal=mi, resident=mh, perm=mp, home_over_perm=mh / mp,
                                 home_over_ideal=mh / mi, perm_over_ideal=mp / mi,
                                 closed=(mh - mp) / (mh - mi) if mh > mi else 0.0,
                                 worse_share=worse / n,
                                 home_pcts=[float(np.percentile(ratios_home, q)) for q in (10, 50, 90)],
                                 perm_pcts=[float(np.percentile(ratios_perm, q)) for q in (10, 50, 90)]))
    return rows


def render_predict(rows, block_m, frac, label) -> str:
    out = [f'Predicted busiest-rank block cost (blocks of {block_m}), permutation from {label}, '
           f'evaluated on the last {100 - round(frac * 100)} % of each layer stream; mean over layers and chunks; '
           f'resident / perm is the ratio of the means; p10 / median / p90 are per-chunk busiest rank / ideal.', '',
           '| trace | EP | chunk | cases | ideal | resident | permutation | resident / perm | gap resident | gap perm | closed | resident p10 / med / p90 | perm p10 / med / p90 | perm worse than resident |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|']
    for r in rows:
        out.append(f"| {r['trace']} | {r['ep']} | {r['chunk']} | {r['cases']} | {r['ideal']:.1f} | {r['resident']:.1f} | {r['perm']:.1f} | "
                   f"{r['home_over_perm']:.3f} | {100 * (r['home_over_ideal'] - 1):.1f}% | {100 * (r['perm_over_ideal'] - 1):.1f}% | "
                   f"{100 * r['closed']:.0f}% | {' / '.join(f'{x:.2f}' for x in r['home_pcts'])} | "
                   f"{' / '.join(f'{x:.2f}' for x in r['perm_pcts'])} | {100 * r['worse_share']:.0f}% |")
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', action='append', required=True, help='name=path.npz (repeatable; several are combined)')
    p.add_argument('--ep', type=int, required=True, help='EP size the placement file is built for')
    p.add_argument('--num-layers', type=int, required=True,
                   help='rows of physical_to_logical_map = num_hidden_layers of the model (43 for DeepSeek V4 Flash)')
    p.add_argument('--block-m', type=int, default=64)
    p.add_argument('--frac', type=float, default=0.25, help='share of each layer stream used for the calibration')
    p.add_argument('--out-location', required=True, help='JSON for --init-expert-location')
    p.add_argument('--out-check', required=True, help='companion JSON for SGLANG_MEGAMOE_STATIC_PLACEMENT: per-layer placement and calibration summary')
    p.add_argument('--predict', type=int, nargs='*', default=[], help='chunk sizes to evaluate the permutation on')
    p.add_argument('--predict-ep', type=int, nargs='*', default=None, help='EP sizes for --predict (default: --ep)')
    p.add_argument('--out-predict', default=None, help='JSON with the prediction rows')
    args = p.parse_args()

    planner = _load_planner()
    caches = load_caches(args.cache)
    names = list(caches)
    E = caches[names[0]]['num_experts']
    for n in names:
        assert caches[n]['num_experts'] == E, f'{n}: {caches[n]["num_experts"]} experts, expected {E}'
    assert E % args.ep == 0, f'{E} experts do not split over EP {args.ep}'
    cals = calibration_histograms(caches, args.frac)
    layer_ids = sorted(set().union(*[set(cals[n]) for n in names]))
    assert layer_ids and layer_ids[-1] < args.num_layers, f'trace layer ids {layer_ids[0]}..{layer_ids[-1]} do not fit {args.num_layers} layers'
    label = ' + '.join(names) + (' (equal weight)' if len(names) > 1 else '')
    predict_eps = sorted(set((args.predict_ep if args.predict_ep is not None else []) + [args.ep]))

    H = E // args.ep
    perms_by_ep: Dict[int, Dict[int, List[int]]] = {}
    summary = {}
    for R in predict_eps:
        perms_by_ep[R] = {}
        for layer_id in layer_ids:
            counts = combine(cals, layer_id)
            perm = planner.static_placement(counts, R, args.block_m)
            assert sorted(perm) == list(range(E))
            perms_by_ep[R][layer_id] = perm
            if R == args.ep:
                b = blocks_of(counts, args.block_m)
                resident = home_loads(b, R)
                permuted = home_loads(planner.apply_static_placement(b, perm), R)
                # Experts whose resident rank differs from the contiguous placement
                # (the study's definition; a slot change within a rank moves nothing).
                summary[layer_id] = dict(moved=sum(1 for q, e in enumerate(perm) if q // H != e // H),
                                         calibration_tokens=int(sum(counts)),
                                         home_max=max(resident), perm_max=max(permuted), ideal=sum(b) / R)
    perms = perms_by_ep[args.ep]
    rows = [perms[l] if l in perms else list(range(E)) for l in range(args.num_layers)]
    with open(args.out_location, 'w') as f:
        json.dump({'physical_to_logical_map': rows}, f, separators=(',', ':'))
    companion = dict(num_layers=args.num_layers, num_experts=E, ep=args.ep, resident_experts_per_rank=H,
                     block_m=args.block_m, frac=args.frac, sources=names, calibration=label,
                     layers={str(l): perms[l] for l in layer_ids},
                     summary={str(l): summary[l] for l in layer_ids})
    with open(args.out_check, 'w') as f:
        json.dump(companion, f, separators=(',', ':'))
    moved = [summary[l]['moved'] for l in layer_ids]
    ratio = [summary[l]['home_max'] / summary[l]['perm_max'] for l in layer_ids]
    print(f'{args.out_location}: {args.num_layers} rows x {E} experts, EP {args.ep} ({H} resident experts per rank), '
          f'{len(layer_ids)} MoE layers permuted ({layer_ids[0]}..{layer_ids[-1]}), identity elsewhere; '
          f'calibration = {label}, first {round(args.frac * 100)} % of each layer stream')
    print(f'experts whose resident rank changes, per layer: min {min(moved)} / mean {sum(moved) / len(moved):.1f} / max {max(moved)} of {E}; '
          f'calibration busiest rank resident / perm: min {min(ratio):.3f} / mean {sum(ratio) / len(ratio):.3f} / max {max(ratio):.3f}')
    if args.predict:
        pred = predict(caches, cals, perms_by_ep, args.frac, args.block_m, args.predict, planner)
        sys.stdout.write(render_predict(pred, args.block_m, args.frac, label))
        if args.out_predict:
            with open(args.out_predict, 'w') as f:
                json.dump(dict(rows=pred, calibration=label, frac=args.frac, block_m=args.block_m), f)


if __name__ == '__main__':
    main()
