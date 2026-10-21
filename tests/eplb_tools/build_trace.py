"""Fold per-(layer, call) topk dumps into one trace file for the benchmark.

Usage: python build_trace.py <dump_dir> <out.pt> --num-experts E --num-topk K [--min-tokens 1024]
Each layer's calls are concatenated along tokens; layers with fewer than
--min-tokens routed tokens are dropped.
With --steps-max-tokens T the calls with at most T tokens (decode steps) are
kept one by one instead: ``steps[i]`` is the list of per-step ``[tokens, topk]``
tensors of layer ``layer_ids[i]``, in call order.
"""
import argparse, glob, os, re
import torch

p = argparse.ArgumentParser()
p.add_argument('dump_dir'); p.add_argument('out')
p.add_argument('--num-experts', type=int, required=True)
p.add_argument('--num-topk', type=int, required=True)
p.add_argument('--min-tokens', type=int, default=1024)
p.add_argument('--steps-max-tokens', type=int, default=0)
p.add_argument('--min-steps', type=int, default=32)
p.add_argument('--cycle', type=int, default=0,
               help='dumps without a layer id (layer-01_*): assign layer = offset + call_index %% cycle')
p.add_argument('--layer-offset', type=int, default=0)
a = p.parse_args()

by_layer = {}
files = sorted(glob.glob(os.path.join(a.dump_dir, 'layer*_call*.pt')))
for f in files:
    m = re.search(r'layer(-?\d+)_call(\d+)', os.path.basename(f))
    layer, call = int(m.group(1)), int(m.group(2))
    if layer < 0:
        # The hook saw no layer id (MiMo's top-k path): the calls of one
        # forward come in layer order, so the call index gives the layer.
        assert a.cycle > 0, 'dumps carry no layer id; pass --cycle <MoE layers per forward>'
        layer = a.layer_offset + call % a.cycle
    t = torch.load(f)
    if t.shape[1] > a.num_topk:  # fused shared-expert columns, if any
        t = t[:, :a.num_topk]
    by_layer.setdefault(layer, []).append(t)
layers, ids = [], []
if a.steps_max_tokens > 0:
    steps = {}
    for layer in sorted(by_layer):
        kept = [t for t in by_layer[layer]
                if 0 < t.shape[0] <= a.steps_max_tokens and (t >= 0).all() and (t < a.num_experts).all()]
        if len(kept) >= a.min_steps:
            steps[layer] = kept
    sizes = sorted({int(t.shape[0]) for v in steps.values() for t in v})
    print(f'{len(steps)} layers with >= {a.min_steps} decode steps; steps per layer '
          f'{[len(v) for v in steps.values()][:8]}...; step sizes {sizes[:6]}..{sizes[-3:]}')
    counts = torch.stack([torch.bincount(torch.cat(v).flatten(), minlength=a.num_experts) for v in steps.values()]).float()
    skew = (a.num_experts * (counts / counts.sum(1, keepdim=True)).pow(2).sum(1))
    print(f'routing skew kappa min/median/max = {skew.min():.2f}/{skew.median():.2f}/{skew.max():.2f}')
    torch.save({'num_experts': a.num_experts, 'num_topk': a.num_topk, 'layer_ids': list(steps),
                'steps': list(steps.values())}, a.out)
    raise SystemExit
for layer in sorted(by_layer):
    t = torch.cat(by_layer[layer])
    t = t[(t >= 0).all(dim=1) & (t < a.num_experts).all(dim=1)]
    if t.shape[0] >= a.min_tokens:
        layers.append(t); ids.append(layer)
counts = torch.stack([torch.bincount(t.flatten(), minlength=a.num_experts) for t in layers]).float()
skew = (a.num_experts * (counts / counts.sum(1, keepdim=True)).pow(2).sum(1))
print(f'{len(layers)} layers, tokens per layer {[int(t.shape[0]) for t in layers][:8]}..., '
      f'routing skew kappa min/median/max = {skew.min():.2f}/{skew.median():.2f}/{skew.max():.2f}')
torch.save({'num_experts': a.num_experts, 'num_topk': a.num_topk, 'layer_ids': ids, 'layers': layers}, a.out)
