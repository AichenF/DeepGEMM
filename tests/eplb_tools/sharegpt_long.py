"""Long-prompt subset of ShareGPT for prefill-heavy serving runs.

Keeps conversations whose first human turn has at least MIN_CHARS characters,
truncates it to MAX_CHARS, and writes NUM records in the same ShareGPT format
so sglang.bench_serving serves it unchanged.

usage: sharegpt_long.py <sharegpt.json> <out.json> [num=512] [min_chars=8000] [max_chars=16000]
"""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
num = int(sys.argv[3]) if len(sys.argv) > 3 else 512
min_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
max_chars = int(sys.argv[5]) if len(sys.argv) > 5 else 16000
rows = []
for c in json.load(open(src)):
    conv = c.get("conversations") or []
    if len(conv) < 2 or conv[0].get("from") != "human" or len(conv[0]["value"]) < min_chars:
        continue
    rows.append({"conversations": [{"from": "human", "value": conv[0]["value"][:max_chars]},
                                   {"from": "gpt", "value": conv[1]["value"]}]})
    if len(rows) == num:
        break
json.dump(rows, open(dst, "w"))
lens = [len(r["conversations"][0]["value"]) for r in rows]
print(len(rows), "rows ->", dst, "first-turn chars mean", sum(lens) // max(len(lens), 1))
