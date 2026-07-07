import torch

import deep_gemm


FP4 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda",
)


def launch(a_f32, codes, name):
    m, k = a_f32.shape
    n = codes.shape[0]
    a = a_f32.to(torch.float8_e4m3fn)
    a_scale = torch.ones((m, 1), dtype=torch.float32, device="cuda")
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).unsqueeze(0).contiguous()
    block_scale = torch.ones((1, n, k // 16), dtype=torch.float8_e4m3fn, device="cuda")
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    offsets = torch.tensor([0, m], dtype=torch.int32, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_nvfp4_gemm_nt_contiguous(
        a, a_scale, packed, block_scale, global_scale, out, offsets
    )
    expected = a.float() @ FP4[codes.long()].T
    error = (out.float() - expected).abs()
    print(name, "max", error.max().item(), "mean", error.mean().item(),
          "cos", torch.nn.functional.cosine_similarity(
              out.float().flatten(), expected.flatten(), dim=0).item())
    print("out row0", out[0, :16].float().cpu().tolist())
    print("ref row0", expected[0, :16].cpu().tolist())
    print("out col0", out[:16, 0].float().cpu().tolist())
    print("ref col0", expected[:16, 0].cpu().tolist())


m, n, k = 64, 64, 128
ones_a = torch.ones((m, k), dtype=torch.float32, device="cuda")
ones_w = torch.full((n, k), 2, dtype=torch.uint8, device="cuda")
launch(ones_a, ones_w, "all ones")

row_values = FP4[(torch.arange(m, device="cuda") % 7) + 1]
row_a = row_values[:, None].expand(m, k).contiguous()
launch(row_a, ones_w, "row pattern")

col_codes = ((torch.arange(n, device="cuda") % 7) + 1).to(torch.uint8)
col_w = col_codes[:, None].expand(n, k).contiguous()
launch(ones_a, col_w, "column pattern")

k_a = torch.zeros((m, k), dtype=torch.float32, device="cuda")
k_a[:, torch.arange(m, device="cuda") % k] = 1.0
k_codes = ((torch.arange(k, device="cuda") % 7) + 1).to(torch.uint8)
k_w = k_codes[None, :].expand(n, k).contiguous()
launch(k_a, k_w, "K pattern")
