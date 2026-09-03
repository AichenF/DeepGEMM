"""MXFP4 quantization + SM90 fused-layout correctness gate.

Validates only the host-side pieces of the SM90 MXFP4 MegaMoE port -- the
quantizer, the E8M0 scale codec, the tile-major scale repack, and the fused
80-byte BK128 weight row. Deliberately does NOT touch the GEMM kernel, so this
runs on any GPU (or CPU) and gates the layout before kernel bring-up.

Run: python3 tests/test_mxfp4_layout_sm90.py
"""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# quantization_mxfp4 is pure torch, but importing it through the deep_gemm
# package pulls in the compiled _C extension. Load it directly when _C is not
# built yet, so this gate runs in any image that merely has torch.
try:
    from deep_gemm import quantization_mxfp4 as _qm  # noqa: E402
except ImportError:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "quantization_mxfp4",
        os.path.join(REPO_ROOT, "deep_gemm", "quantization_mxfp4.py"),
    )
    _qm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_qm)

FP4_VALUES = _qm.FP4_VALUES
UE8M0_BIAS = _qm.UE8M0_BIAS
dequantize_mxfp4_to_fp32 = _qm.dequantize_mxfp4_to_fp32
fp32_to_ue8m0_ceil = _qm.fp32_to_ue8m0_ceil
mxfp4_fuse_packed_with_scale_tile_major = _qm.mxfp4_fuse_packed_with_scale_tile_major
mxfp4_scale_to_tile_major = _qm.mxfp4_scale_to_tile_major
quantize_to_mxfp4 = _qm.quantize_to_mxfp4
ue8m0_to_fp32 = _qm.ue8m0_to_fp32

GROUP_SIZE = 32
BLOCK_N = 256
BLOCK_K = 128


def test_e8m0_codec(device: str) -> None:
    """E8M0 must be an exact power of two and round up, never down.

    Rounding down would let a group's max magnitude exceed FP4_MAX*scale and
    silently clip during quantization.
    """
    codes = torch.arange(0, 255, dtype=torch.uint8, device=device)
    values = ue8m0_to_fp32(codes)
    exponents = torch.log2(values)
    torch.testing.assert_close(exponents, exponents.round(), rtol=0, atol=0)
    torch.testing.assert_close(
        exponents, (codes.to(torch.float32) - UE8M0_BIAS), rtol=0, atol=0
    )

    x = torch.logspace(-20, 20, 4096, dtype=torch.float32, device=device)
    encoded = ue8m0_to_fp32(fp32_to_ue8m0_ceil(x))
    assert (encoded >= x).all(), "E8M0 encode must round up"
    assert (encoded < x * 2 + 1e-30).all(), "E8M0 encode must be the *smallest* such power"
    print("e8m0 codec: PASS")


def test_quant_roundtrip(device: str) -> None:
    """Dequantized values must land exactly on the FP4-times-scale lattice, and
    the scale must never clip the group max.

    Note what is deliberately NOT asserted: idempotency. Re-quantizing a
    dequantized MXFP4 tensor can halve the scale, and that is a property of the
    format, not a bug. ``ceil_po2`` at most doubles, so max/scale lands in
    (3, 6]; a group whose max/scale is just above 3 rounds to the FP4 level 3.0,
    making the next round's max/6 exactly scale/2. NVFP4 does not show this
    because its UE4M3 scale tracks max/6 closely enough that max/scale stays
    near 6. Single quantize -> dequantize -> GEMM, which is what the kernel
    does, is unaffected.
    """
    torch.manual_seed(0)
    w = torch.randn(4, BLOCK_N, 512, dtype=torch.float32, device=device) * 0.3
    packed, scale = quantize_to_mxfp4(w, group_size=GROUP_SIZE)
    assert packed.dtype == torch.uint8 and packed.shape == (4, BLOCK_N, 256)
    assert scale.dtype == torch.uint8 and scale.shape == (4, BLOCK_N, 16)

    deq = dequantize_mxfp4_to_fp32(packed, scale, group_size=GROUP_SIZE)
    assert deq.shape == w.shape and torch.isfinite(deq).all()

    scale_f = ue8m0_to_fp32(scale).repeat_interleave(GROUP_SIZE, dim=-1)
    ratio = (deq.abs() / scale_f).flatten()
    lattice = FP4_VALUES.to(device)
    off_lattice = (ratio.unsqueeze(-1) - lattice).abs().min(dim=-1).values
    assert off_lattice.max().item() < 1e-5, "dequant left the FP4 lattice"

    # The scale must cover the group max, or quantization silently clips.
    group_max = w.abs().view(4, BLOCK_N, 16, GROUP_SIZE).amax(dim=-1)
    assert (ue8m0_to_fp32(scale) * 6.0 >= group_max - 1e-6).all(), "scale clips the group max"

    cosine = torch.nn.functional.cosine_similarity(
        deq.flatten(), w.flatten(), dim=0
    ).item()
    assert cosine > 0.99, f"quantization cosine {cosine:.4f} too low"
    print(f"quant roundtrip: PASS (on-lattice, non-clipping, cosine={cosine:.4f})")


def test_dequant_matches_elementwise_reference(device: str) -> None:
    """Cross-check the vectorized dequant against a literal per-element unpack.

    This is the assertion that actually pins down the Marlin nibble order: byte
    b of each 4-byte chunk holds K[b] in its high nibble and K[b+4] in its low
    nibble. A transposed or off-by-one unpack still produces plausible-looking
    numbers and would pass every other check here.
    """
    torch.manual_seed(7)
    E, N, K = 2, 8, 128
    w = torch.randn(E, N, K, dtype=torch.float32, device=device) * 0.4
    packed, scale = quantize_to_mxfp4(w, group_size=GROUP_SIZE)
    deq = dequantize_mxfp4_to_fp32(packed, scale, group_size=GROUP_SIZE)

    flat_p = packed.reshape(-1, K // 2).cpu()
    nibbles = torch.zeros(flat_p.shape[0], K, dtype=torch.uint8)
    for row in range(flat_p.shape[0]):
        for chunk in range(K // 8):
            for b in range(4):
                byte = int(flat_p[row, chunk * 4 + b])
                nibbles[row, chunk * 8 + b] = (byte >> 4) & 0xF
                nibbles[row, chunk * 8 + b + 4] = byte & 0xF

    scale_f = ue8m0_to_fp32(scale).reshape(-1, K // GROUP_SIZE).cpu()
    magnitudes = FP4_VALUES[(nibbles & 0x7).long()]
    signs = torch.where((nibbles >> 3).bool(), -1.0, 1.0)
    expected = magnitudes * signs * scale_f.repeat_interleave(GROUP_SIZE, dim=-1)
    torch.testing.assert_close(deq.reshape(-1, K).cpu(), expected, rtol=0, atol=0)
    print("dequant vs elementwise reference: PASS")


def test_tile_major_is_pure_permutation(device: str) -> None:
    """The tile-major repack must move bytes without changing their multiset,
    and must be exactly invertible -- a wrong permute would corrupt scales in a
    way the kernel cannot detect."""
    torch.manual_seed(1)
    E, N, G = 3, BLOCK_N * 2, 512 // GROUP_SIZE
    scale = torch.randint(0, 255, (E, N, G), dtype=torch.uint8, device=device)
    tm = mxfp4_scale_to_tile_major(
        scale, block_n=BLOCK_N, block_k=BLOCK_K, group_size=GROUP_SIZE
    )
    groups_per_k_block = BLOCK_K // GROUP_SIZE
    assert tm.shape == (
        E, N // BLOCK_N, G // groups_per_k_block, BLOCK_N, groups_per_k_block,
    )
    torch.testing.assert_close(
        tm.flatten().sort().values, scale.flatten().sort().values, rtol=0, atol=0
    )
    restored = (
        tm.permute(0, 1, 3, 2, 4)
        .reshape(E, N, G)
    )
    torch.testing.assert_close(restored, scale, rtol=0, atol=0)
    print("tile-major repack: PASS")


def test_fused_row_layout(device: str) -> None:
    """Each BK128 fused row must be 64B packed FP4 + 4B E8M0 scale + 12B zero
    padding, recoverable byte-for-byte. The kernel indexes these offsets
    directly, so a layout drift here is a silent wrong-answer bug."""
    torch.manual_seed(2)
    E, N, K = 2, BLOCK_N, 512
    w = torch.randn(E, N, K, dtype=torch.float32, device=device) * 0.2
    packed, scale = quantize_to_mxfp4(w, group_size=GROUP_SIZE)
    tm = mxfp4_scale_to_tile_major(
        scale, block_n=BLOCK_N, block_k=BLOCK_K, group_size=GROUP_SIZE
    )
    fused = mxfp4_fuse_packed_with_scale_tile_major(packed, tm, block_k=BLOCK_K)

    k_blocks = K // BLOCK_K
    groups_per_k_block = BLOCK_K // GROUP_SIZE
    fused_row_bytes = BLOCK_K // 2 + 16
    assert fused.shape == (E, N, k_blocks * fused_row_bytes)

    rows = fused.view(E, N, k_blocks, fused_row_bytes)
    got_packed = rows[..., : BLOCK_K // 2].reshape(E, N, K // 2)
    torch.testing.assert_close(got_packed, packed, rtol=0, atol=0)

    got_scale = rows[
        ..., BLOCK_K // 2 : BLOCK_K // 2 + groups_per_k_block
    ].reshape(E, N, k_blocks * groups_per_k_block)
    torch.testing.assert_close(got_scale, scale, rtol=0, atol=0)

    padding = rows[..., BLOCK_K // 2 + groups_per_k_block :]
    assert padding.shape[-1] == 12, f"expected 12 padding bytes, got {padding.shape[-1]}"
    assert (padding == 0).all(), "fused row padding must be deterministic zero"

    # The fused layout must still dequantize to the same values.
    deq_direct = dequantize_mxfp4_to_fp32(packed, scale, group_size=GROUP_SIZE)
    deq_fused = dequantize_mxfp4_to_fp32(got_packed, got_scale, group_size=GROUP_SIZE)
    torch.testing.assert_close(deq_fused, deq_direct, rtol=0, atol=0)
    print("fused row layout: PASS")


def test_weight_transform_matches_nvfp4_structure(device: str) -> None:
    """The MXFP4 SM90 weight transform must produce the same fused-row stride
    and dtypes as the NVFP4 one, differing only in scale-byte count, so the
    shared K-major TMA descriptor path and sign braid stay valid."""
    try:
        import deep_gemm
        from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4
    except ImportError:
        print("mxfp4 weight transform vs nvfp4: SKIP (deep_gemm._C not built)")
        return

    torch.manual_seed(3)
    E, IH, H = 2, BLOCK_N, 512
    l1 = torch.randn(E, 2 * IH, H, dtype=torch.float32, device=device) * 0.2
    l2 = torch.randn(E, H, IH, dtype=torch.float32, device=device) * 0.2

    mx_l1 = quantize_to_mxfp4(l1, group_size=GROUP_SIZE)
    mx_l2 = quantize_to_mxfp4(l2, group_size=GROUP_SIZE)
    nv_l1 = quantize_to_nvfp4(l1, group_size=16)
    nv_l2 = quantize_to_nvfp4(l2, group_size=16)

    (mx_l1_out, mx_l1_sf), (mx_l2_out, mx_l2_sf) = (
        deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(mx_l1, mx_l2)
    )
    (nv_l1_out, _), (nv_l2_out, _) = (
        deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(nv_l1, nv_l2)
    )

    # Same fused storage footprint: 80-byte BK128 rows either way.
    assert mx_l1_out.shape == nv_l1_out.shape, (mx_l1_out.shape, nv_l1_out.shape)
    assert mx_l2_out.shape == nv_l2_out.shape
    assert mx_l1_out.dtype == torch.uint8 and mx_l2_out.dtype == torch.uint8
    assert mx_l1_out.shape[-1] % 80 == 0

    # MXFP4 carries half the scale elements of NVFP4 (group 32 vs 16).
    assert mx_l1_sf.numel() * 2 == nv_l1[1].numel(), (mx_l1_sf.numel(), nv_l1[1].numel())

    # Layout sentinel must be set so nvfp4_mega_moe cannot silently accept
    # MXFP4 weights (and vice versa).
    from deep_gemm.mega import (
        _SM90_MXFP4_H20_FUSED_LAYOUT,
        _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR,
        _SM90_NVFP4_H200_FUSED_LAYOUT_ATTR,
    )
    assert getattr(mx_l1_out, _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR, None) == _SM90_MXFP4_H20_FUSED_LAYOUT
    assert getattr(mx_l1_out, _SM90_NVFP4_H200_FUSED_LAYOUT_ATTR, None) is None
    print("mxfp4 weight transform vs nvfp4: PASS")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")
    test_e8m0_codec(device)
    test_quant_roundtrip(device)
    test_dequant_matches_elementwise_reference(device)
    test_tile_major_is_pure_permutation(device)
    test_fused_row_layout(device)
    test_weight_transform_matches_nvfp4_structure(device)
    print("ALL MXFP4 LAYOUT TESTS PASS")


if __name__ == "__main__":
    main()
