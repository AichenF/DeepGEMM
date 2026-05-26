"""Offline MXFP4 quantization for W4A8 fused MegaMoE (v2)."""
import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0


def fp32_to_fp4_nibble(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs()
    values = FP4_VALUES.to(x.device)
    diffs = (mag.unsqueeze(-1) - values).abs()
    nibble_idx = diffs.argmin(dim=-1).to(torch.uint8)
    return sign | nibble_idx


def quantize_to_mxfp4_w4a8(weight_fp8: torch.Tensor, group_size: int = 32):
    assert weight_fp8.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight_fp8.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight_fp8.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    e_unbiased = torch.ceil(torch.log2(desired_scale)).clamp(-127, 127)
    scale = torch.exp2(e_unbiased)
    e8m0 = (e_unbiased + 127).to(torch.uint8).squeeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # Marlin permutation: chunk of 8 K nibbles → 4 bytes with
    #   byte b: low = K[b+4], high = K[b].
    # Marlin's bit shift produces frag_b[0]=[K0..K3], frag_b[1]=[K4..K7].
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    return packed, e8m0.contiguous()


def dequantize_mxfp4_to_fp32(packed: torch.Tensor, e8m0: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    G = K // group_size
    # Inverse Marlin permutation: each 4-byte chunk represents 8 K elements;
    # low nibbles → K[4..7], high nibbles → K[0..3].
    pck = packed.view(*outer_shape, K // 8, 4)
    low = pck & 0x0F                # K[b+4]
    high = (pck >> 4) & 0x0F        # K[b]
    nibbles = torch.cat([high, low], dim=-1).view(*outer_shape, K)
    sign_bit = (nibbles >> 3) & 0x1
    mag_idx = (nibbles & 0x7).to(torch.long)
    fp4_values = FP4_VALUES.to(packed.device)
    mag = fp4_values[mag_idx]
    values = torch.where(sign_bit.bool(), -mag, mag)
    e_unbiased = e8m0.to(torch.int32) - 127
    scale = torch.exp2(e_unbiased.to(torch.float32))
    scale_expanded = scale.unsqueeze(-1).expand(*outer_shape, G, group_size).reshape(*outer_shape, K)
    return values * scale_expanded


if __name__ == "__main__":
    torch.manual_seed(0)
    E, N, K = 4, 256, 4096
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16) * 0.3
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_fp32_ref = w_fp8.to(torch.float32)

    packed, e8m0 = quantize_to_mxfp4_w4a8(w_fp8, group_size=32)
    print(f"packed shape: {packed.shape}, e8m0 shape: {e8m0.shape}")
    print(f"weight bytes: FP8={w_fp8.numel()} → packed={packed.numel()} + e8m0={e8m0.numel()}  "
          f"({(packed.numel()+e8m0.numel())/w_fp8.numel()*100:.1f}% of original)")

    w_recovered = dequantize_mxfp4_to_fp32(packed, e8m0, group_size=32)

    # Element-level error (informational only)
    err = (w_recovered - w_fp32_ref).abs()
    print(f"Element error: max_abs={err.max():.4f}  mean_abs={err.mean():.4f}")
    # Filter near-zero refs for meaningful rel error
    mask = w_fp32_ref.abs() > 0.05
    rel_err = (err[mask] / w_fp32_ref.abs()[mask]).mean().item()
    print(f"Element rel error (|ref|>0.05): {rel_err*100:.2f}%")

    # MATMUL error — actual goal: produce similar matmul output to FP8 weight
    M = 512
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
    x_fp32 = x_bf16.to(torch.float32)
    for ei in range(E):
        out_fp8_ref = x_fp32 @ w_fp32_ref[ei].T  # (M, N)
        out_mxfp4 = x_fp32 @ w_recovered[ei].T
        out_err = (out_mxfp4 - out_fp8_ref).abs()
        rel = (out_err / (out_fp8_ref.abs() + 1e-6)).mean().item()
        print(f"  expert {ei}: matmul max_abs={out_err.max():.3f}  mean_rel={rel*100:.2f}%")
    print("OK")
