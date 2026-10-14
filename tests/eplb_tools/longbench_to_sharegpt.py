"""LongBench (THUDM/LongBench, data.zip) as a ShareGPT-format prompt set for
prefill-heavy serving runs with sglang.bench_serving.

Each record is one LongBench sample: prompt = context + question, truncated to
MAX_CHARS characters (about MAX_CHARS / 4 tokens), answer = the first reference
answer. Samples are drawn round-robin from the English tasks so no single task
dominates the routing mix.

usage: longbench_to_sharegpt.py <data.zip> <out.json> [num=256] [max_chars=48000] [min_chars=8000]
"""
import io
import json
import sys
import zipfile

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
         "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
         "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

src, dst = sys.argv[1], sys.argv[2]
num = int(sys.argv[3]) if len(sys.argv) > 3 else 256
max_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 48000
min_chars = int(sys.argv[5]) if len(sys.argv) > 5 else 8000

per_task = {}
with zipfile.ZipFile(src) as z:
    names = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.endswith(".jsonl")}
    for task in TASKS:
        name = names.get(f"{task}.jsonl")
        if name is None:
            continue
        rows = []
        for line in io.TextIOWrapper(z.open(name), encoding="utf-8"):
            r = json.loads(line)
            prompt = (r.get("context") or "") + "\n\n" + (r.get("input") or "")
            if len(prompt) < min_chars:
                continue
            answers = r.get("answers") or [""]
            rows.append({"conversations": [{"from": "human", "value": prompt[-max_chars:]},
                                           {"from": "gpt", "value": str(answers[0])}]})
        per_task[task] = rows

out = []
i = 0
while len(out) < num and any(per_task.values()):
    task = TASKS[i % len(TASKS)]
    i += 1
    rows = per_task.get(task)
    if rows:
        out.append(rows.pop(0))
json.dump(out, open(dst, "w"))
lens = [len(r["conversations"][0]["value"]) for r in out]
print(len(out), "rows ->", dst, "from", sum(1 for t in per_task if per_task[t] is not None), "tasks,",
      "prompt chars mean", sum(lens) // max(len(lens), 1), "min", min(lens or [0]), "max", max(lens or [0]))
