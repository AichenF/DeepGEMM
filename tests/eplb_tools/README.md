# Routing traces for the dynamic EPLB benchmark

1. Install the dump hook into the serving SGLang (edits `topk.py` in place,
   wraps `select_experts`):
   `python install_topk_hook.py [path/to/sglang/srt/layers/moe/topk.py]`
2. Run real prompts through an engine with `MEGAMOE_TOPK_DUMP_DIR` set:
   `python capture_routing.py --model-path <ckpt> --tp 8 --dataset gsm8k --data gsm8k_test.parquet`
   (`--dataset sharegpt --data ShareGPT_V3_unfiltered_cleaned_split.json` for ShareGPT;
   `--extra` forwards engine flags, bare flags such as `--disable-radix-cache` are accepted).
3. Fold the per-layer dumps into one trace:
   `python build_trace.py <dump_dir> <out.pt> --num-experts E --num-topk K`
4. Replay it in the kernel benchmark:
   `python tests/test_mxfp4_mega_moe_sm90_eplb.py --trace <out.pt> --trace-num-layers 4 ...`

## Serving integration (SGLang, SM90 MXFP4 MegaMoE)

`sglang_patch/install_mxfp4_megamoe.py <sglang_python_root>` wires the SM90
MXFP4 MegaMoE kernel into an SGLang tree that already carries the SM90 FP8
MegaMoE integration (`--moe-a2a-backend megamoe`). It copies
`mega_moe_sm90_mxfp4.py` next to `mega_moe_sm90.py`, lets
`should_use_mega_moe` accept FP4 experts on SM90, dispatches them before the
FP8 symmetric buffer is created, and routes the FP4 expert weights to
`build_sm90_mxfp4_mega_moe_experts_weights` (OCP packed FP4 + per-32 E8M0
scales, float or byte encoded).

The two serving arms differ only in one environment variable read by the
server: `SGLANG_MEGAMOE_DYNAMIC_EPLB_REDUNDANT_SLOTS=0` is fixed placement on the same
kernel, `SGLANG_MEGAMOE_DYNAMIC_EPLB_REDUNDANT_SLOTS=S` enables the balancer with `S`
redundant slots per rank (`SGLANG_MEGAMOE_DYNAMIC_EPLB_OPEN_GATE_BLOCKS` is the
open gate). With the balancer on, the arena is allocated at weight-build time
and shares one physical-slot symmetric buffer across layers, so KV-cache
sizing sees it and nothing is allocated during CUDA graph capture.

`gsm8k_quick_eval.py <gsm8k_test.parquet> <port> [num] [concurrency]` is a
greedy zero-shot GSM8K accuracy probe against a running server, used to check
that both arms produce the same quality.
