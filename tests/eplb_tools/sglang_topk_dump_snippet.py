# Routing-trace capture for the MegaMoE dynamic EPLB benchmark.
# Appended to sglang/srt/layers/moe/topk.py by install_topk_hook.py, which wraps
# select_experts() to call _dump_topk_ids(layer_id, topk_output, num_tokens). Set MEGAMOE_TOPK_DUMP_DIR to enable; only
# prefill-sized batches (>= MEGAMOE_TOPK_DUMP_MIN_TOKENS, default 256) on
# TP rank 0 are written, one file per (layer, call).
import os as _os

_DUMP_DIR = _os.environ.get("MEGAMOE_TOPK_DUMP_DIR")
_DUMP_MIN_TOKENS = int(_os.environ.get("MEGAMOE_TOPK_DUMP_MIN_TOKENS", "256"))
_DUMP_COUNTS = {}


def _dump_topk_ids(layer_id, topk_output, num_tokens):
    if _DUMP_DIR is None or num_tokens < _DUMP_MIN_TOKENS:
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:
        pass
    ids = getattr(topk_output, "topk_ids", None)
    if ids is None:
        return
    import torch
    n = _DUMP_COUNTS.get(layer_id, 0)
    _DUMP_COUNTS[layer_id] = n + 1
    _os.makedirs(_DUMP_DIR, exist_ok=True)
    torch.save(ids[:num_tokens].detach().to("cpu", torch.int32),
               _os.path.join(_DUMP_DIR, f"layer{int(layer_id) if layer_id is not None else -1:03d}_call{n:04d}.pt"))
