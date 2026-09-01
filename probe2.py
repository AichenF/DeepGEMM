"""Probe the iter-2 (2ms) breakdown on GPU 0."""
import importlib.util, os, time
import torch
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_ext_tp')
REPO = '/home/xutingz/fac/DeepGEMM_tp'
def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sol = _load(os.path.join(REPO, 'solution', 'tp_moe_kernel.py'), 'sol')
qm = _load(os.path.join(REPO, 'deep_gemm', 'quantization_mxfp4.py'), 'qm')

dev = torch.device('cuda', 0)
M, H, I, E, topk, tp = 8, 6144, 2048, 128, 8, 8
Is = I // tp
g = torch.Generator(device=dev).manual_seed(0)
x = (torch.randn(M, H, generator=g, device=dev)).to(torch.bfloat16)
scores = torch.randn(M, E, generator=g, device=dev)
topk_w, topk_idx = torch.topk(scores, topk, dim=-1)
W1s = (torch.randn(E, 2 * Is, H, generator=g, device=dev) * 0.05).to(torch.bfloat16)
W2s = (torch.randn(E, H, Is, generator=g, device=dev) * 0.05).to(torch.bfloat16)
l1p, l1s = qm.quantize_to_mxfp4(W1s, 32); l2p, l2s = qm.quantize_to_mxfp4(W2s, 32)

def T(fn, n=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3

fe = topk_idx.reshape(-1).long()
uniq, inv = torch.unique(fe, return_inverse=True); U = uniq.numel()
print(f"U={U}")
print(f"full tp_moe_partial: {T(lambda: sol.tp_moe_partial(x, l1p, l1s, l2p, l2s, topk_idx, topk_w, Is)):.3f} ms")
print(f"CUDA dequant l1+l2 : {T(lambda: (sol._dq_bf16(l1p[uniq], l1s[uniq], U*2*Is, H), sol._dq_bf16(l2p[uniq], l2s[uniq], U*H, Is))):.3f} ms")
W1 = sol._dq_bf16(l1p[uniq], l1s[uniq], U*2*Is, H).view(U, 2*Is, H)
W2 = sol._dq_bf16(l2p[uniq], l2s[uniq], U*H, Is).view(U, H, Is)
gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]
xt = x.unsqueeze(1).expand(M, topk, H).reshape(-1, H)
print(f"FC1 einsum (gather):{T(lambda: torch.einsum('nh,nih->ni', xt, gate_w[inv])):.3f} ms")
a = torch.randn(M*topk, Is, device=dev, dtype=torch.bfloat16)
print(f"FC2 einsum (gather):{T(lambda: torch.einsum('ni,nhi->nh', a, W2[inv])):.3f} ms")
yo = torch.randn(M*topk, H, device=dev)
row = torch.arange(M, device=dev).repeat_interleave(topk)
print(f"index_add          :{T(lambda: torch.zeros(M,H,device=dev).index_add_(0,row,yo)):.3f} ms")
