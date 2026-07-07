"""Traceable reference driver for the SM90 W4A8 grouped GEMM generation."""

import torch


CUSTOM_MODEL_CPP_BINDINGS = r"""
torch::Tensor launch_gpu_implementation(
    torch::Tensor a,
    torch::Tensor a_scale,
    torch::Tensor w_packed,
    torch::Tensor block_scale,
    torch::Tensor global_scale,
    torch::Tensor offsets);
"""


FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class ReferenceModel:
    @staticmethod
    def forward(a, a_scale, w_packed, block_scale, global_scale, offsets):
        e, n, packed_k = w_packed.shape
        k = packed_k * 2
        table = FP4_E2M1.to(w_packed.device)
        lo = (w_packed & 0x0F).long()
        hi = (w_packed >> 4).long()
        codes = torch.stack((lo, hi), dim=-1).reshape(e, n, k)
        w = table[codes].reshape(e, n, k // 16, 16)
        scale = block_scale.float() * global_scale.reshape(e, 1, 1)
        w = (w * scale.unsqueeze(-1)).reshape(e, n, k)
        a_deq = a.float() * a_scale
        out = torch.empty(a.shape[0], n, dtype=torch.bfloat16, device=a.device)
        host_offsets = offsets.cpu().tolist()
        for expert in range(e):
            begin, end = host_offsets[expert:expert + 2]
            if begin < end:
                out[begin:end] = (
                    a_deq[begin:end] @ w[expert].transpose(0, 1)
                ).to(torch.bfloat16)
        return out


test_sizes = [
    (512, 2048, 2048, 8),
    (4096, 2048, 2048, 8),
    (8192, 4096, 2048, 16),
    (4096, 7168, 2048, 32),
    (8192, 4096, 7168, 8),
    (128, 4096, 7168, 8),
]


# The authoritative comparison is elementwise:
# abs(actual - reference) <= 2.0 + 0.05 * abs(reference), with no NaN/Inf.
