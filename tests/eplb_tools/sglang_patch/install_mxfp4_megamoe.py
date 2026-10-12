"""Wire the SM90 MXFP4 MegaMoE path (with optional dynamic EPLB) into an SGLang
source tree that already carries the SM90 FP8 MegaMoE integration.

Usage: python install_mxfp4_megamoe.py <sglang_python_root>   (the dir containing sglang/)
Idempotent. Edits:
  * copies mega_moe_sm90_mxfp4.py next to mega_moe_sm90.py
  * mega_moe.py: should_use_mega_moe accepts MXFP4 experts on SM90;
    _run_mega_routed dispatches to run_sm90_mxfp4_mega_routed for them
  * quantization/fp8.py: on SM90 with the megamoe a2a backend, FP4 experts are
    built for the MXFP4 kernel instead of asserting
"""
import os
import re
import shutil
import sys

root = sys.argv[1]
srt = os.path.join(root, "sglang", "srt")
here = os.path.dirname(os.path.abspath(__file__))

dst = os.path.join(srt, "layers", "moe", "mega_moe_sm90_mxfp4.py")
shutil.copyfile(os.path.join(here, "mega_moe_sm90_mxfp4.py"), dst)
print("copied", dst)

# --- mega_moe.py -----------------------------------------------------------
p = os.path.join(srt, "layers", "moe", "mega_moe.py")
s = open(p).read()
if "run_sm90_mxfp4_mega_routed" not in s:
    s = s.replace(
        "from sglang.srt.layers.moe.mega_moe_sm90 import (\n    is_sm90_fp8_mega_moe_available,\n    run_sm90_mega_routed,\n)",
        "from sglang.srt.layers.moe.mega_moe_sm90 import (\n    is_sm90_fp8_mega_moe_available,\n    run_sm90_mega_routed,\n)\n"
        "from sglang.srt.layers.moe.mega_moe_sm90_mxfp4 import (\n    is_sm90_mxfp4_mega_moe_available,\n    run_sm90_mxfp4_mega_routed,\n)",
    )
    old = "    if _device_sm == 90:\n        if not is_sm90_fp8_mega_moe_available(moe.experts):\n            return False\n"
    assert old in s, "should_use_mega_moe anchor not found"
    s = s.replace(old, "    if _device_sm == 90:\n        if not (is_sm90_fp8_mega_moe_available(moe.experts)\n                or is_sm90_mxfp4_mega_moe_available(moe.experts)):\n            return False\n")
    open(p, "w").write(s)
    print("patched", p)
else:
    print("already patched", p)
# The MXFP4 dispatch must run before the FP8 symmetric buffer is created (that
# buffer is sized for the FP8 kernel and uses a newer DeepGEMM signature).
late = ("    if _device_sm == 90 and is_sm90_mxfp4_mega_moe_available(moe.experts):\n"
        "        return run_sm90_mxfp4_mega_routed(\n            moe, hidden_states, topk_ids_in, topk_weights_in, num_tokens, num_max_tokens_per_rank)\n")
early = ("    if _device_sm == 90 and is_sm90_mxfp4_mega_moe_available(moe.experts):\n"
         "        return run_sm90_mxfp4_mega_routed(\n            moe, hidden_states, topk_ids, topk_weights, num_tokens, num_max_tokens_per_rank)\n\n")
s = open(p).read()
if early not in s:
    s = s.replace(late, "")
    anchor = "    buf = _get_mega_moe_symm_buffer(\n        ep_group,\n"
    assert anchor in s, "symm buffer anchor not found"
    s = s.replace(anchor, early + anchor, 1)
    open(p, "w").write(s)
    print("moved MXFP4 dispatch before the FP8 buffer", p)

# --- fp8.py ------------------------------------------------------------------
p = os.path.join(srt, "layers", "quantization", "fp8.py")
s = open(p).read()
if "build_sm90_mxfp4_mega_moe_experts_weights" not in s:
    old = ("                if get_moe_a2a_backend().is_megamoe():\n"
           "                    from sglang.srt.layers.moe.mega_moe import (\n"
           "                        build_mega_moe_experts_weights,\n"
           "                    )\n\n"
           "                    build_mega_moe_experts_weights(layer)\n"
           "                    return\n")
    assert old in s, "fp8.py megamoe anchor not found"
    new = ("                if get_moe_a2a_backend().is_megamoe():\n"
           "                    if get_platform().is_sm90:\n"
           "                        from sglang.srt.layers.moe.mega_moe_sm90_mxfp4 import (\n"
           "                            build_sm90_mxfp4_mega_moe_experts_weights,\n"
           "                        )\n\n"
           "                        build_sm90_mxfp4_mega_moe_experts_weights(layer)\n"
           "                        return\n"
           "                    from sglang.srt.layers.moe.mega_moe import (\n"
           "                        build_mega_moe_experts_weights,\n"
           "                    )\n\n"
           "                    build_mega_moe_experts_weights(layer)\n"
           "                    return\n")
    s = s.replace(old, new)
    # The FP4 experts never reach the FP8 builder on SM90 any more, but keep the
    # assertion honest if they do.
    open(p, "w").write(s)
    print("patched", p)
else:
    print("already patched", p)
