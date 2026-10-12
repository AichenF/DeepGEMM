"""Fold per-(layer, call) topk dumps into one trace file for the benchmark.

Usage: python build_trace.py <dump_dir> <out.pt> --num-experts E --num-topk K [--min-tokens 1024]
Each layer's calls are concatenated along tokens; layers with fewer than
--min-tokens routed tokens are dropped.
"""
import argparse, glob, os, re
import torch

p = argparse.ArgumentParser()
p.add_argument('dump_dir'); p.add_argument('out')
p.add_argument('--num-experts', type=int, required=True)
p.add_argument('--num-topk', type=int, required=True)
p.add_argument('--min-tokens', type=int, default=1024)
a = p.parse_args()

by_layer = {}
for f in sorted(glob.glob(os.path.join(a.dump_dir, 'layer*_call*.pt'))):
    layer = int(re.search(r'layer(\d+)_', f).group(1))
    t = torch.load(f)
    if t.shape[1] > a.num_topk:  # fused shared-expert columns, if any
        t = t[:, :a.num_topk]
    by_layer.setdefault(layer, []).append(t)
layers, ids = [], []
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
