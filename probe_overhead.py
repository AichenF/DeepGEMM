"""Where is the flat ~1ms for small M? Kernel vs python-wrapper overhead."""
import importlib.util, os, time
import torch
REPO = os.environ.get('TP_REPO', '/lustre/raplab/client/xutingz/fac/DeepGEMM_tp')
def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sol = _load(os.path.join(REPO, 'solution', 'tp_moe_kernel.py'), 'sol')
qm = _load(os.path.join(REPO, 'deep_gemm', 'quantization_mxfp4.py'), 'qm')
dev = torch.device('cuda', 0)
M, H, I, E, topk, tp = 1, 6144, 2048, 384, 8, 8
Is = I // tp
g = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(M, H, generator=g, device=dev).to(torch.bfloat16)
scores = torch.randn(M, E, generator=g, device=dev)
topk_w, topk_idx = torch.topk(scores, topk, dim=-1)
gate = (torch.randn(E, Is, H, generator=g, device=dev)*0.05).to(torch.bfloat16)
up = (torch.randn(E, Is, H, generator=g, device=dev)*0.05).to(torch.bfloat16)
W2 = (torch.randn(E, H, Is, generator=g, device=dev)*0.05).to(torch.bfloat16)
l1p, l1s = qm.quantize_to_mxfp4(torch.cat([gate,up],1).contiguous(), 32)
l2p, l2s = qm.quantize_to_mxfp4(W2.contiguous(), 32)

def T(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t)/n*1e3

# prebuilt kernel args
tok = torch.arange(M, device=dev, dtype=torch.int32).repeat_interleave(topk)
exp = topk_idx.reshape(-1).to(torch.int32)
wt = topk_w.reshape(-1).to(torch.float32)
xb = x.to(torch.bfloat16).contiguous()
l1pc, l1sc, l2pc, l2sc = l1p.contiguous(), l1s.contiguous(), l2p.contiguous(), l2s.contiguous()

print(f"M={M} pairs={M*topk}")
print(f"full tp_moe_partial : {T(lambda: sol.tp_moe_partial(x,l1p,l1s,l2p,l2s,topk_idx,topk_w,Is)):.4f} ms")
print(f"raw kernel (prebuilt): {T(lambda: sol._ext.fused_tp_moe(xb,l1pc,l1sc,l2pc,l2sc,tok,exp,wt,M,H,Is)):.4f} ms")
print(f"python setup only   : {T(lambda: (torch.arange(M,device=dev,dtype=torch.int32).repeat_interleave(topk), topk_idx.reshape(-1).to(torch.int32), topk_w.reshape(-1).to(torch.float32), x.to(torch.bfloat16).contiguous())):.4f} ms")
print(f"contiguous() x4     : {T(lambda: (l1p.contiguous(),l1s.contiguous(),l2p.contiguous(),l2s.contiguous())):.4f} ms")
