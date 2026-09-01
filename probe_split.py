"""Single-GPU probe: where does the 9.5ms rank-0 compute go — dequant vs matmul?"""
import importlib.util, os, time
import torch

REPO = '/home/xutingz/fac/DeepGEMM_tp'
def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
qm = _load(os.path.join(REPO, 'deep_gemm', 'quantization_mxfp4.py'), 'qm')
dequant = qm.dequantize_mxfp4_to_fp32

dev = torch.device('cuda', 0)
M, H, I, E, topk, tp = 8, 6144, 2048, 128, 8, 8
Is = I // tp
g = torch.Generator(device=dev).manual_seed(0)
x = (torch.randn(M, H, generator=g, device=dev) * 1).to(torch.bfloat16)
scores = torch.randn(M, E, generator=g, device=dev)
topk_w, topk_idx = torch.topk(scores, topk, dim=-1)
W1s = (torch.randn(E, 2 * Is, H, generator=g, device=dev) * 0.05).to(torch.bfloat16)
W2s = (torch.randn(E, H, Is, generator=g, device=dev) * 0.05).to(torch.bfloat16)
l1p, l1s = qm.quantize_to_mxfp4(W1s, group_size=32)
l2p, l2s = qm.quantize_to_mxfp4(W2s, group_size=32)

flat_e = topk_idx.reshape(-1).long()
uniq, inv = torch.unique(flat_e, return_inverse=True)
U = uniq.numel()
dt = torch.bfloat16


def T(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3


def dq1(): return _dq1_hold.__setitem__(0, _dequant_pair())
def _dequant_pair():
    W1 = dequant(l1p[uniq], l1s[uniq], group_size=32).view(U, 2 * Is, H).to(dt)
    W2 = dequant(l2p[uniq], l2s[uniq], group_size=32).view(U, H, Is).to(dt)
    return W1, W2
_dq1_hold = [None]

print(f"U(touched experts)={U}")
print(f"dequant (l1+l2, touched): {T(lambda: _dequant_pair()):.3f} ms")
W1, W2 = _dequant_pair()
gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]
xt = x.to(dt).unsqueeze(1).expand(M, topk, H).reshape(-1, H)
print(f"FC1 einsum (gather+matmul): {T(lambda: torch.einsum('nh,nih->ni', xt, gate_w[inv])):.3f} ms")
a = torch.randn(M * topk, Is, device=dev, dtype=dt)
print(f"FC2 einsum (gather+matmul): {T(lambda: torch.einsum('ni,nhi->nh', a, W2[inv])):.3f} ms")
print(f"unique(): {T(lambda: torch.unique(flat_e, return_inverse=True)):.3f} ms")
