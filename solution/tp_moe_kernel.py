"""TP MXFP4 MoE — per-rank partial FFN over the intermediate shard.

Interface (AKO-optimized target):
    tp_moe_partial(x, l1_packed, l1_scale, l2_packed, l2_scale, topk_idx, topk_w, Is)
      -> y_partial[M, H]  (this rank's partial; caller all-reduce-sums across TP ranks)

Weights are RAW per-rank MXFP4 shards (not the fused/tile-major layout):
    l1 = quantize_to_mxfp4(cat(gate_shard, up_shard) [E,2*Is,H])   -> packed [E,2Is,H/2], scale [E,2Is,H/32]
    l2 = quantize_to_mxfp4(W2_shard [E,H,Is])                       -> packed [E,H,Is/2], scale [E,H,Is/32]

Baseline: dequant MXFP4 -> compute in torch. AKO iterations replace this with a
fused CUDA MXFP4 kernel (in-kernel dequant + FC1/SwiGLU/FC2 grouped by expert).
"""
import importlib.util
import os

import torch

_REPO = '/home/xutingz/fac/DeepGEMM_tp'
_spec = importlib.util.spec_from_file_location(
    'quantization_mxfp4', os.path.join(_REPO, 'deep_gemm', 'quantization_mxfp4.py'))
_qm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qm)
_dequant = _qm.dequantize_mxfp4_to_fp32


def tp_moe_partial(x, l1_packed, l1_scale, l2_packed, l2_scale, topk_idx, topk_w, Is):
    M, H = x.shape
    topk = topk_idx.shape[1]
    dt = torch.bfloat16

    flat_e = topk_idx.reshape(-1).long()               # [M*topk]
    flat_w = topk_w.reshape(-1).to(torch.float32)      # [M*topk]
    # Only dequant the experts actually touched by this batch (<= M*topk).
    uniq, inv = torch.unique(flat_e, return_inverse=True)
    U = uniq.numel()
    W1 = _dequant(l1_packed[uniq], l1_scale[uniq], group_size=32).view(U, 2 * Is, H).to(dt)
    W2 = _dequant(l2_packed[uniq], l2_scale[uniq], group_size=32).view(U, H, Is).to(dt)
    gate_w, up_w = W1[:, :Is, :], W1[:, Is:, :]        # [U, Is, H]

    xt = x.to(dt).unsqueeze(1).expand(M, topk, H).reshape(-1, H)   # [M*topk, H]
    g = torch.einsum('nh,nih->ni', xt, gate_w[inv])    # bf16 tensor-core matmul
    u = torch.einsum('nh,nih->ni', xt, up_w[inv])
    gf = g.float()
    a = ((gf * torch.sigmoid(gf)) * u.float()).to(dt)  # SwiGLU (fp32 for the nonlinearity)
    yo = torch.einsum('ni,nhi->nh', a, W2[inv]).float() * flat_w.unsqueeze(1)

    y = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    row = torch.arange(M, device=x.device).repeat_interleave(topk)
    y.index_add_(0, row, yo)
    return y
