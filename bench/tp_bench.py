"""Custom AKO evaluator for the TP MXFP4 MoE (distributed, 8 ranks).

Times the per-rank forward (solution.tp_moe_partial + NVLink all-reduce), checks
cosine vs a torch mxfp4-dequant golden, and prints KernelBench-style keys the AKO
loop reads: COMPILED / CORRECT / RUNTIME / REF_RUNTIME / SPEEDUP.

Memory: weights are generated per-shard (never the full [E,2I,H]) and quantized
immediately, so each GPU only holds its own MXFP4 shard (~1GB) — fits alongside
other users on shared GPUs.
"""
import argparse
import importlib.util
import os
import sys

import torch
import torch.distributed as dist


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPO = '/home/xutingz/fac/DeepGEMM_tp'
_qm = _load(os.path.join(REPO, 'deep_gemm', 'quantization_mxfp4.py'), 'qm')
quantize_to_mxfp4, dequant = _qm.quantize_to_mxfp4, _qm.dequantize_mxfp4_to_fp32


def init_dist(local_rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    port = os.environ.get('MASTER_PORT', '8361')
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            world_size=world, rank=local_rank,
                            device_id=torch.device(f'cuda:{local_rank}'))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.new_group(list(range(world)))


def gen_shard(seed, r, E, I, H, ws, dev):
    """Deterministically generate + MXFP4-quantize shard r (never the full weights)."""
    Is = I // (seed_tp := gen_shard.tp)
    gw = torch.Generator(device=dev).manual_seed(seed * 100003 + r)
    gate = (torch.randn(E, Is, H, generator=gw, device=dev, dtype=torch.float32) * ws).to(torch.bfloat16)
    up = (torch.randn(E, Is, H, generator=gw, device=dev, dtype=torch.float32) * ws).to(torch.bfloat16)
    W2 = (torch.randn(E, H, Is, generator=gw, device=dev, dtype=torch.float32) * ws).to(torch.bfloat16)
    W1s = torch.cat([gate, up], dim=1).contiguous()
    del gate, up
    l1 = quantize_to_mxfp4(W1s, group_size=32)
    l2 = quantize_to_mxfp4(W2.contiguous(), group_size=32)
    del W1s, W2
    return l1, l2, Is


def ref_partial(x, l1, l2, topk_idx, topk_w, Is):
    """Trusted torch mxfp4-dequant partial FFN (correctness golden)."""
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


def worker(local_rank, args):
    rank, group = init_dist(local_rank, args.tp)
    dev = torch.device('cuda', local_rank)
    sol = _load(args.solution, 'solution')
    gen_shard.tp = args.tp

    M, H, I, E, topk, tp = args.m, args.hidden, args.inter, args.experts, args.topk, args.tp
    gx = torch.Generator(device=dev).manual_seed(args.seed)
    x = (torch.randn(M, H, generator=gx, device=dev, dtype=torch.float32)).to(torch.bfloat16)
    scores = torch.randn(M, E, generator=gx, device=dev, dtype=torch.float32)
    topk_w, topk_idx = torch.topk(scores, topk, dim=-1)

    l1, l2, Is = gen_shard(args.seed, rank, E, I, H, args.wscale, dev)

    def compute():
        return sol.tp_moe_partial(x, l1[0], l1[1], l2[0], l2[1], topk_idx, topk_w, Is)

    def forward():
        yp = compute()
        dist.all_reduce(yp, op=dist.ReduceOp.SUM, group=group)
        return yp

    y = forward()
    ok = torch.tensor([1.0, 1.0], device=dev)
    if rank == 0:
        golden = torch.zeros(M, H, device=dev, dtype=torch.float32)
        for r in range(tp):
            lr1, lr2, _ = gen_shard(args.seed, r, E, I, H, args.wscale, dev)
            golden += ref_partial(x, lr1, lr2, topk_idx, topk_w, Is)
            del lr1, lr2
        cos = torch.nn.functional.cosine_similarity(y.flatten(), golden.flatten(), dim=0).item()
        finite = bool(torch.isfinite(y).all().item())
        ok = torch.tensor([cos, 1.0 if finite else 0.0], device=dev)
        del golden

    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    # Primary signal: rank-0 compute-only (the sharded-FFN kernel), on the freeer GPU 0,
    # so the optimization signal is not dominated by GPU 5-7 contention on the all-reduce barrier.
    for _ in range(args.warmup):
        compute()
    torch.cuda.synchronize()
    st.record()
    for _ in range(args.iters):
        compute()
    en.record(); torch.cuda.synchronize()
    compute_ms = st.elapsed_time(en) / args.iters

    # Secondary: full forward (compute + all-reduce), max across ranks (contention-inflated).
    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize(); dist.barrier(group)
    st.record()
    for _ in range(args.iters):
        forward()
    en.record(); torch.cuda.synchronize()
    tms = torch.tensor([st.elapsed_time(en) / args.iters], device=dev)
    dist.all_reduce(tms, op=dist.ReduceOp.MAX, group=group)

    ref_ms = torch.tensor([0.0], device=dev)
    if rank == 0:
        for _ in range(3):
            _ = ref_partial(x, l1, l2, topk_idx, topk_w, Is)
        torch.cuda.synchronize()
        st.record()
        for _ in range(args.iters):
            _ = ref_partial(x, l1, l2, topk_idx, topk_w, Is)
        en.record(); torch.cuda.synchronize()
        ref_ms = torch.tensor([st.elapsed_time(en) / args.iters], device=dev)

    if rank == 0:
        cos, finite = ok[0].item(), ok[1].item()
        correct = (cos >= args.cos_thr) and (finite > 0.5)
        rms = ref_ms.item()
        speedup = rms / compute_ms if compute_ms > 0 else 0.0
        print("COMPILED=True", flush=True)
        print(f"CORRECT={correct}  (cosine={cos:.5f} finite={finite>0.5})", flush=True)
        print(f"RUNTIME={compute_ms:.4f} ms  (rank0 compute; M={M} E={E} tp={tp})", flush=True)
        print(f"FULL_FORWARD={tms.item():.4f} ms  (compute+allreduce, max_rank, contention-inflated)", flush=True)
        print(f"REF_RUNTIME={rms:.4f} ms", flush=True)
        print(f"SPEEDUP={speedup:.3f}x", flush=True)
    sys.stdout.flush()
    dist.barrier(group)
    os._exit(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--solution', required=True)
    p.add_argument('--m', type=int, default=8)
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--inter', type=int, default=2048)
    p.add_argument('--experts', type=int, default=128)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--tp', type=int, default=8)
    p.add_argument('--wscale', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--cos-thr', type=float, default=0.99)
    a = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(a,), nprocs=a.tp, join=True)
    print("SPAWN_JOINED_OK", flush=True)
