"""Run one layer's checkpoint experts through the SM90 MXFP4 MegaMoE kernel
and compare with a dequantized reference.

Isolates the weight path (checkpoint bytes -> adapter transform -> kernel)
from the serving path. EP1, the first N experts of one layer, random tokens
routed top-k among them. Tries the adapter's byte convention and its two
obvious alternatives so a layout mismatch names itself.

Usage:
  python tests/eplb_tools/check_ckpt_experts.py --model <dir> --layer 5 --num-experts 16
"""
import argparse
import glob
import json
import os
import re
import struct
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _headers(model_dir):
    """Tensor name -> (file, dtype, shape) over every safetensors shard."""
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (f, v["dtype"], v["shape"])
    return out


def _load(name, headers, device):
    from safetensors import safe_open
    f, dtype, shape = headers[name]
    with safe_open(f, "pt", device="cpu") as s:
        t = s.get_tensor(name)
    return t.to(device)


# HF naming (MiMo: mlp.experts.N.gate_proj.weight / .weight_scale) and the
# DeepSeek V4 export (ffn.experts.N.w1.weight / .scale); w1 = gate, w3 = up,
# w2 = down in both conventions.
_PROJ_ALIASES = {"gate_proj": ("gate_proj", "w1"), "up_proj": ("up_proj", "w3"), "down_proj": ("down_proj", "w2")}


def _find(headers, layer, expert, proj):
    names = "|".join(_PROJ_ALIASES[proj])
    pat = re.compile(rf"layers\.{layer}\.(mlp|ffn)\.experts\.{expert}\.({names})\.weight$")
    scale_pat = re.compile(rf"layers\.{layer}\.(mlp|ffn)\.experts\.{expert}\.({names})\.(weight_scale(_inv)?|scale)$")
    w = [k for k in headers if pat.search(k)]
    s = [k for k in headers if scale_pat.search(k)]
    assert len(w) == 1 and len(s) == 1, f"{proj} e{expert}: weights {w} scales {s}"
    return w[0], s[0]


def ocp_to_chunk8(packed):
    """OCP bytes (element 2j in the low nibble) -> DeepGEMM chunk-of-8 layout."""
    *outer, half_k = packed.shape
    k = half_k * 2
    nib = torch.empty((*outer, k), dtype=torch.uint8, device=packed.device)
    nib[..., 0::2] = packed & 0x0F
    nib[..., 1::2] = (packed >> 4) & 0x0F
    ch = nib.view(*outer, k // 8, 8)
    return (ch[..., 4:8] | (ch[..., 0:4] << 4)).to(torch.uint8).reshape(*outer, half_k).contiguous()


def ocp_swapped_to_chunk8(packed):
    """Same, but assuming element 2j sits in the HIGH nibble of byte j."""
    *outer, half_k = packed.shape
    k = half_k * 2
    nib = torch.empty((*outer, k), dtype=torch.uint8, device=packed.device)
    nib[..., 0::2] = (packed >> 4) & 0x0F
    nib[..., 1::2] = packed & 0x0F
    ch = nib.view(*outer, k // 8, 8)
    return (ch[..., 4:8] | (ch[..., 0:4] << 4)).to(torch.uint8).reshape(*outer, half_k).contiguous()


def dequant_ocp(packed, scale_u8, bias=127):
    """OCP MXFP4 -> fp32: low nibble first, 32-element groups, E8M0 scale."""
    *outer, half_k = packed.shape
    k = half_k * 2
    nib = torch.empty((*outer, k), dtype=torch.uint8, device=packed.device)
    nib[..., 0::2] = packed & 0x0F
    nib[..., 1::2] = (packed >> 4) & 0x0F
    sign = ((nib >> 3) & 1).to(torch.float32)
    mag = FP4_VALUES.to(packed.device)[(nib & 0x7).long()]
    val = mag * (1 - 2 * sign)
    g = k // 32
    sc = torch.ldexp(torch.ones_like(scale_u8, dtype=torch.float32),
                     (scale_u8.to(torch.int32) - bias)).view(*outer, g, 1)
    return (val.view(*outer, g, 32) * sc).view(*outer, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--num-experts", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--scale-key", default=None, help="weight_scale or weight_scale_inv (auto)")
    args = ap.parse_args()

    rank, _, group = init_dist(0, 1)
    dev = torch.device("cuda")
    H = _headers(args.model)
    E = args.num_experts
    g_w, u_w, d_w, g_s, u_s, d_s = [], [], [], [], [], []
    for e in range(E):
        gw, gs = _find(H, args.layer, e, "gate_proj")
        uw, us = _find(H, args.layer, e, "up_proj")
        dw, ds = _find(H, args.layer, e, "down_proj")
        if e == 0:
            print("names:", gw, gs, "|", dw, ds, "| dtypes", H[gw][1], H[gs][1], H[dw][1], H[ds][1],
                  "| shapes", H[gw][2], H[gs][2], H[dw][2], H[ds][2], flush=True)
        g_w.append(_load(gw, H, dev)); u_w.append(_load(uw, H, dev)); d_w.append(_load(dw, H, dev))
        g_s.append(_load(gs, H, dev)); u_s.append(_load(us, H, dev)); d_s.append(_load(ds, H, dev))

    def u8(t):
        return t.view(torch.uint8) if t.dtype != torch.uint8 else t

    def scale_u8(t):
        if t.dtype in (torch.uint8, torch.int8) or str(t.dtype).endswith("e8m0fnu"):
            return t.view(torch.uint8)
        f = t.float()
        return (torch.log2(f.clamp_min(2.0 ** -127)).round() + 127).clamp(0, 255).to(torch.uint8)

    w13 = torch.stack([torch.cat([u8(g), u8(u)], 0) for g, u in zip(g_w, u_w)]).contiguous()
    s13 = torch.stack([torch.cat([scale_u8(g), scale_u8(u)], 0) for g, u in zip(g_s, u_s)]).contiguous()
    w2 = torch.stack([u8(d) for d in d_w]).contiguous()
    s2 = torch.stack([scale_u8(d) for d in d_s]).contiguous()
    hidden = w13.shape[2] * 2
    inter = w13.shape[1] // 2
    print(f"w13 {tuple(w13.shape)} s13 {tuple(s13.shape)} w2 {tuple(w2.shape)} s2 {tuple(s2.shape)} hidden {hidden} inter {inter}")
    for nm, s in (("s13", s13), ("s2", s2)):
        si = s.to(torch.int32)
        print(f"scale bytes {nm}: min {si.min().item()} max {si.max().item()} mean {si.float().mean().item():.1f} "
              f"(E8M0 with bias 127: 2^{si.min().item() - 127} .. 2^{si.max().item() - 127})")
    for nm, s in (("s13", s13), ("s2", s2)):
        c = s.to(torch.int32).view(E, -1); span = (c.amax(1) - c.amin(1))
        print(f"per-expert code span {nm}: max {span.max().item()} mean {span.float().mean().item():.1f} (kernel window is 12 codes)")
    for nm, w in (("w13", w13), ("w2", w2)):
        lo = (w & 0x0F).to(torch.int64); hi = ((w >> 4) & 0x0F).to(torch.int64)
        print(f"nibble histogram {nm}: low {torch.bincount(lo.flatten(), minlength=16).tolist()}")
        print(f"                     high {torch.bincount(hi.flatten(), minlength=16).tolist()}")

    # Reference under the OCP convention (what SGLang's Marlin path assumes and
    # what produced correct text).
    w13_ref = dequant_ocp(w13, s13)
    w2_ref = dequant_ocp(w2, s2)
    print(f"dequant stats: w13 absmax {w13_ref.abs().max().item():.4g} rms {w13_ref.pow(2).mean().sqrt().item():.4g}; "
          f"w2 absmax {w2_ref.abs().max().item():.4g} rms {w2_ref.pow(2).mean().sqrt().item():.4g}")

    M, K = args.tokens, args.topk
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    x_bf = (torch.randn((M, hidden), dtype=torch.bfloat16, device=dev, generator=gen))
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)
    x_ref = (x_fp8.float().view(M, hidden // 128, 128) * x_sf.float().unsqueeze(2)).view(M, hidden)
    topk_idx = torch.stack([torch.randperm(E, device=dev, generator=gen)[:K] for _ in range(M)]).to(torch.int64)
    topk_w = torch.rand((M, K), device=dev, generator=gen)
    topk_w = topk_w / topk_w.sum(1, keepdim=True)

    def reference(w13d, w2d):
        y = torch.zeros((M, hidden), dtype=torch.float32, device=dev)
        for e in range(E):
            hits = (topk_idx == e).nonzero()
            if hits.numel() == 0:
                continue
            tok = hits[:, 0]; w = topk_w[hits[:, 0], hits[:, 1]]
            l1 = x_ref[tok] @ w13d[e].t()
            h = torch.nn.functional.silu(l1[:, :inter]) * l1[:, inter:] * w[:, None]
            y.index_add_(0, tok, h @ w2d[e].t())
        return y
    y_ref = reference(w13_ref, w2_ref)

    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, E, max(M, 256), K, hidden, inter,
                                                 use_fp8_dispatch=True, activation="swiglu")

    def run(tag, w13_p, w2_p, s13_p, s2_p, ref=y_ref, gs=(None, None)):
        l1, l2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90((w13_p, s13_p), (w2_p, s2_p))
        buf.x[:M].copy_(x_fp8); buf.x_sf[:M].copy_(x_sf)
        buf.topk_idx[:M].copy_(topk_idx); buf.topk_weights[:M].copy_(topk_w.float())
        y = torch.empty((M, hidden), dtype=torch.bfloat16, device=dev)
        deep_gemm.mxfp4_mega_moe(y, l1, l2, buf, l1_global_scales=gs[0], l2_global_scales=gs[1],
                                 activation_clamp=None, fast_math=True)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(y.float(), ref, dim=-1)
        rel = (y.float() - ref).norm() / ref.norm()
        print(f"[{tag}] cos mean {cos.mean().item():.4f} min {cos.min().item():.4f} rel-err {rel.item():.4f} "
              f"finite {torch.isfinite(y).all().item()} |y| {y.float().abs().mean().item():.4g} |ref| {ref.abs().mean().item():.4g}", flush=True)

    run("adapter before fix: OCP->chunk8, [gate;up], E8M0 raw", ocp_to_chunk8(w13), ocp_to_chunk8(w2), s13, s2)
    from deep_gemm.quantization_mxfp4 import fold_e8m0_into_kernel_window
    s13_w, gs1, lost1 = fold_e8m0_into_kernel_window(s13)
    s2_w, gs2, lost2 = fold_e8m0_into_kernel_window(s2)
    print(f"window fold: l1 shift {(133 - s13.view(E, -1).amax(1).int()).tolist()[:8]}... groups lost {lost1:.5f}; "
          f"l2 groups lost {lost2:.5f}")
    run("adapter after fix: codes folded into 122..133 + per-expert global scales",
        ocp_to_chunk8(w13), ocp_to_chunk8(w2), s13_w, s2_w, gs=(gs1, gs2))
    run("variant: bytes passed through (no nibble permutation)", w13, w2, s13, s2)
    run("variant: high nibble first", ocp_swapped_to_chunk8(w13), ocp_swapped_to_chunk8(w2), s13, s2)
    w13_swap = torch.cat([w13[:, inter:], w13[:, :inter]], 1).contiguous()
    s13_swap = torch.cat([s13[:, inter:], s13[:, :inter]], 1).contiguous()
    run("variant: [up;gate]", ocp_to_chunk8(w13_swap), ocp_to_chunk8(w2), s13_swap, s2)
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
