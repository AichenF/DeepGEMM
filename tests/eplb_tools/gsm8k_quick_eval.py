"""Tiny GSM8K accuracy probe against a running SGLang server (OpenAI-compatible
completions). Usage: gsm8k_quick_eval.py <parquet> <port> [num] [concurrency]"""
import json, re, sys, concurrent.futures, urllib.request
import pandas as pd

path, port = sys.argv[1], int(sys.argv[2])
num = int(sys.argv[3]) if len(sys.argv) > 3 else 100
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 16
df = pd.read_parquet(path).head(num)

def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None

def ask(q):
    body = json.dumps({"model": "default", "prompt": f"Question: {q}\nAnswer: Let's think step by step.", "max_tokens": 384, "temperature": 0, "stop": ["\nQuestion:", "\n\n\n"]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["text"]

with concurrent.futures.ThreadPoolExecutor(conc) as ex:
    outs = list(ex.map(ask, df["question"].tolist()))
correct = 0
for out, ans in zip(outs, df["answer"].tolist()):
    gold = ans.split("####")[-1].strip().replace(",", "")
    pred = last_number(out)
    correct += int(pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-6) if pred and re.match(r"^-?\d+(\.\d+)?$", pred) else 0
json.dump([{"q": q, "out": o} for q, o in zip(df["question"].tolist(), outs)], open(f"gsm8k_outputs_{port}.json", "w"))
print(f"GSM8K quick eval: {correct}/{len(outs)} = {correct/len(outs):.3f}")
print("SAMPLE:", outs[0][:300].replace("\n", " | "))
