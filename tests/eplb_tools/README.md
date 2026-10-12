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
