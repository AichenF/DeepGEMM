"""Emit a real transformed MXFP4 weight tile plus its expected FP8 decode.

Uses the actual host transform rather than a reimplementation: the point of the
test is whether the kernel reads what the transform writes, so re-deriving the
layout in the test would only check my arithmetic against itself.
"""
import importlib.util, os, sys, numpy as np, torch

# Defaults match the build container's mounts; override to run elsewhere.
ROOT = os.environ.get("DG_ROOT", "/src")
OUT = os.environ.get("DG_OUT", "/gate")
spec = importlib.util.spec_from_file_location(
    "qm", os.path.join(ROOT, "deep_gemm", "quantization_mxfp4.py"))
qm = importlib.util.module_from_spec(spec); spec.loader.exec_module(qm)
sys.modules["quantization_mxfp4"] = qm
# Load the braid straight from the installed package; adding ROOT to sys.path
# would shadow the built extension with the source tree.
from deep_gemm.mega import _braid_nvfp4_mode2_signs   # shared with NVFP4: it
                                                      # only permutes FP4 sign
                                                      # bits, group-size free

torch.manual_seed(7)
E, N, K = 1, 256, 128
w = torch.randn(E, N, K, dtype=torch.float32) * 0.3

packed, scale = qm.quantize_to_mxfp4(w, group_size=32)
tm = qm.mxfp4_scale_to_tile_major(scale, block_n=N, block_k=128, group_size=32)
fused = qm.mxfp4_fuse_packed_with_scale_tile_major(packed, tm, block_k=128)
fused = _braid_nvfp4_mode2_signs(fused)

# Reference FP8 bytes, in logical K order.
deq = qm.dequantize_mxfp4_to_fp32(packed, scale, group_size=32)
ref = deq.to(torch.float8_e4m3fn).view(torch.uint8)

fused.cpu().numpy().astype(np.uint8).tofile(os.path.join(OUT, "chain_packed.bin"))
ref.cpu().numpy().astype(np.uint8).tofile(os.path.join(OUT, "chain_ref.bin"))
print(f"wrote packed {tuple(fused.shape)} ref {tuple(ref.shape)}")
