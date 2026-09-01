"""Single-GPU compute bench for the TP MXFP4 MoE per-rank kernel (the thing we
optimize; the all-reduce is separate/multi-GPU). Run one candidate on one GPU:

    CUDA_VISIBLE_DEVICES=<i> python tp_compute_bench.py --solution <path> --m <M>

Prints COMPILED / CORRECT (cosine vs torch mxfp4-dequant golden) / RUNTIME (ms).
Designed to run inside the bench container; use it 8-wide (one GPU per candidate).
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

REPO = os.environ.get('TP_REPO', '/lustre/raplab/client/xutingz/fac/DeepGEMM_tp')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_qm = _load(os.path.join(REPO, 'deep_gemm', 'quantization_mxfp4.py'), 'qm')
quantize_to_mxfp4, dequant = _qm.quantize_to_mxfp4, _qm.dequantize_mxfp4_to_fp32


def gen_shard(seed, r, E, I, H, ws, tp, dev):
    Is = I // tp
    g = torch.Generator(device=dev).manual_seed(seed * 100003 + r)
    gate = (torch.randn(E, Is, H, generator=g, device=dev) * ws).to(torch.bfloat16)
    up = (torch.randn(E, Is, H, generator=g, device=dev) * ws).to(torch.bfloat16)
    W2 = (torch.randn(E, H, Is, generator=g, device=dev) * ws).to(torch.bfloat16)
    W1s = torch.cat([gate, up], dim=1).contiguous()
    del gate, up
    l1 = quantize_to_mxfp4(W1s, group_size=32)
    l2 = quantize_to_mxfp4(W2.contiguous(), group_size=32)
    return l1, l2, Is


def ref_partial(x, l1, l2, topk_idx, topk_w, Is):
    E = l1[0].shape[0]; M, H = x.shape; topk = topk_idx.shape[1]
    W1 = dequant(l1[0], l1[1], group_size=32).view(E, 2 * Is, H).float()
    W2 = dequant(l2[0], l2[1], group_size=32).view(E, H, Is).float()
    gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]
    fe = topk_idx.reshape(-1).long(); fw = topk_w.reshape(-1).float()
    xt = x.float().unsqueeze(1).expand(M, topk, H).reshape(-1, H)
    g = torch.einsum('nh,nih->ni', xt, gate_w[fe])
    u = torch.einsum('nh,nih->ni', xt, up_w[fe])
    a = (g * torch.sigmoid(g)) * u
    yo = torch.einsum('ni,nhi->nh', a, W2[fe]) * fw.unsqueeze(1)
    y = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    y.index_add_(0, torch.arange(M, device=x.device).repeat_interleave(topk), yo)
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solution', required=True)
    p.add_argument('--m', type=int, default=8)
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--inter', type=int, default=2048)
    p.add_argument('--experts', type=int, default=384)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--tp', type=int, default=8)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--wscale', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cos-thr', type=float, default=0.99)
    a = p.parse_args()
    dev = torch.device('cuda', 0)
    torch.cuda.set_device(0)

    try:
        sol = _load(a.solution, 'solution')
    except Exception as e:
        print(f"COMPILED=False  ({type(e).__name__}: {str(e)[:200]})", flush=True)
        return
    print("COMPILED=True", flush=True)

    M, H, I, E, topk, tp = a.m, a.hidden, a.inter, a.experts, a.topk, a.tp
    gx = torch.Generator(device=dev).manual_seed(a.seed)
    x = torch.randn(M, H, generator=gx, device=dev).to(torch.bfloat16)
    scores = torch.randn(M, E, generator=gx, device=dev)
    topk_w, topk_idx = torch.topk(scores, topk, dim=-1)
    l1, l2, Is = gen_shard(a.seed, a.rank, E, I, H, a.wscale, tp, dev)

    def compute():
        return sol.tp_moe_partial(x, l1[0], l1[1], l2[0], l2[1], topk_idx, topk_w, Is)

    try:
        y = compute()
        golden = ref_partial(x, l1, l2, topk_idx, topk_w, Is)
        cos = torch.nn.functional.cosine_similarity(y.flatten(), golden.flatten(), dim=0).item()
        finite = bool(torch.isfinite(y).all().item())
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"CORRECT=False  (RUNTIME_ERROR {type(e).__name__}: {str(e)[:200]})", flush=True)
        return

    for _ in range(a.warmup):
        compute()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(a.iters):
        compute()
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / a.iters

    correct = (cos >= a.cos_thr) and finite
    print(f"CORRECT={correct}  (cosine={cos:.5f} finite={finite})", flush=True)
    print(f"RUNTIME={ms:.4f} ms  (M={M} E={E} tp={tp} pairs={M*topk})", flush=True)


if __name__ == '__main__':
    main()
