import argparse
import os

import pytest
import torch

import deep_gemm


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

TASK_SHAPES = [
    (512, 2048, 2048, 8),
    (4096, 2048, 2048, 8),
    (8192, 4096, 2048, 16),
    (4096, 7168, 2048, 32),
    (8192, 4096, 7168, 8),
    (128, 4096, 7168, 8),
]


def require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 GPU required")


def quantize_fp8_per_token(x):
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    q = (x / scale).to(torch.float8_e4m3fn)
    return q, scale, q.float() * scale


def quantize_nvfp4(w):
    e, n, k = w.shape
    global_scale = w.abs().amax(dim=(1, 2)).clamp_min(1e-12) / (6.0 * 448.0)
    grouped = w.view(e, n, k // 16, 16)
    block_scale_f32 = grouped.abs().amax(dim=-1) / 6.0
    block_scale = (
        block_scale_f32 / global_scale.view(e, 1, 1)
    ).to(torch.float8_e4m3fn)
    effective_scale = (
        block_scale.float() * global_scale.view(e, 1, 1)
    ).clamp_min(1e-12)

    table = FP4_VALUES.to(w.device)
    normalized = grouped / effective_scale.unsqueeze(-1)
    codes = (normalized.unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
    dequant = table[codes.long()] * effective_scale.unsqueeze(-1)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(e, n, k // 2)
    return packed.contiguous(), block_scale.contiguous(), global_scale.contiguous(), dequant.reshape(e, n, k)


def make_offsets(m, e, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.rand(e, generator=generator)
    counts = ((raw / raw.sum()) * m).floor().to(torch.int64).tolist()
    counts[-1] += m - sum(counts)
    values = [0]
    for count in counts:
        values.append(values[-1] + count)
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def run_case(shape, seed=1, explicit_offsets=None):
    require_sm90()
    m, n, k, e = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a_f32 = torch.randn((m, k), generator=generator, device="cuda")
    w_f32 = torch.randn((e, n, k), generator=generator, device="cuda")
    a, a_scale, a_dequant = quantize_fp8_per_token(a_f32)
    w_packed, block_scale, global_scale, w_dequant = quantize_nvfp4(w_f32)
    offsets = (torch.tensor(explicit_offsets, dtype=torch.int32, device="cuda")
               if explicit_offsets is not None else make_offsets(m, e, seed + 7))

    actual = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_nvfp4_gemm_nt_contiguous(
        a, a_scale, w_packed, block_scale, global_scale, actual, offsets
    )
    torch.cuda.synchronize()

    expected = torch.empty_like(actual)
    host_offsets = offsets.cpu().tolist()
    for expert in range(e):
        begin, end = host_offsets[expert:expert + 2]
        if begin < end:
            expected[begin:end] = (
                a_dequant[begin:end] @ w_dequant[expert].T
            ).to(torch.bfloat16)

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    allowed = 2.0 + 0.05 * expected_f32.abs()
    violations = diff > allowed
    n_bad = int(violations.sum().item())
    max_abs = float(diff.max().item())
    cosine = float(torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0).item())
    print(f"shape={shape} n_bad={n_bad} max_abs={max_abs:.6f} cosine={cosine:.8f}")
    assert not bool(torch.isnan(actual_f32).any() or torch.isinf(actual_f32).any())
    assert n_bad == 0


def test_small_unaligned_and_empty_expert():
    run_case((70, 128, 128, 4), seed=7, explicit_offsets=[0, 17, 17, 69, 70])


def test_multiblock_k():
    run_case((257, 256, 2048, 8), seed=3)


@pytest.mark.skipif(os.getenv("DG_W4A8_FULL") != "1",
                    reason="set DG_W4A8_FULL=1 for the six task shapes")
@pytest.mark.parametrize("shape", TASK_SHAPES)
def test_task_shapes(shape):
    run_case(shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    test_small_unaligned_and_empty_expert()
    test_multiblock_k()
    if args.full:
        for task_shape in TASK_SHAPES:
            run_case(task_shape)
