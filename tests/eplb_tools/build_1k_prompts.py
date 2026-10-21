"""Build about-1k-token prefill prompts from GSM8K (few-shot) and MMLU (few-shot).

Writes ShareGPT-style json files that ``sglang.bench_serving --dataset-name
sharegpt`` can serve (a list of ``{"conversations": [{"from": "human", "value":
prompt}, {"from": "gpt", "value": answer}]}``), for prefill-only serving runs
(``--sharegpt-output-len 1``) and for routing capture. Short raw questions
(about 60 tokens) only form 2-4k-token chunks under chunked prefill, where the
MoE layers see a few hundred tokens per rank and no balancer can win; the
few-shot prompts give the same datasets at about 1k tokens per request.

GSM8K: each prompt is ``shots`` worked examples sampled (seeded, per prompt)
from the train split, "Question: ...\\nAnswer: <solution> #### <number>", then
the target test question and "Answer:". The "gpt" value is the reference
solution. If no train split is given, the first ``shots`` test questions serve
as the shots for every prompt and are excluded from the targets.

MMLU: the standard few-shot format, "The following are multiple choice
questions (with answers) about <subject>." followed by ``shots`` examples of
the same subject (the 5 dev examples first, then validation examples when more
shots are requested), then the target question and "Answer:". Targets are
sampled uniformly over the subjects with a fixed seed. The "gpt" value is the
answer letter.

The number of shots is raised (up to ``--max-shots``) while the mean prompt
length is below 0.9 x ``--target-tokens``; prompts longer than
``--max-tokens`` are dropped and resampled. Lengths are measured with the
model's tokenizer.

Usage:
  python build_1k_prompts.py --tokenizer <model dir> --out-dir <datasets dir> \\
      --gsm8k-test gsm8k_test.parquet --gsm8k-train gsm8k_train.parquet \\
      --mmlu-test mmlu_test.parquet --mmlu-dev mmlu_dev.parquet \\
      --mmlu-validation mmlu_validation.parquet [--num-prompts 1000]
"""
import argparse
import collections
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

# Login nodes may cap the thread count; the files are small, so read them single-threaded.
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load(path):
    return pq.read_table(path, use_threads=False).to_pylist()


def stats(lengths):
    s = sorted(lengths)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"count": len(s), "mean": round(sum(s) / len(s), 1), "p10": pick(0.10), "p50": pick(0.50),
            "p90": pick(0.90), "min": s[0], "max": s[-1]}


def gsm8k_example(row, with_answer):
    text = f"Question: {row['question']}\nAnswer:"
    return text + f" {row['answer']}\n\n" if with_answer else text


def build_gsm8k(test, train, shots, num_prompts, seed):
    rng = random.Random(seed)
    if train:
        pool, targets = train, test
    else:
        pool, targets = test[:shots], test[shots:]
    targets = targets[:num_prompts]
    rows = []
    for row in targets:
        demos = rng.sample(pool, shots) if train else pool
        prompt = "".join(gsm8k_example(d, True) for d in demos) + gsm8k_example(row, False)
        rows.append({"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": row["answer"]}]})
    return rows


LETTERS = "ABCD"


def mmlu_example(row, with_answer):
    text = row["question"].strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(LETTERS, row["choices"])) + "Answer:"
    return text + f" {LETTERS[row['answer']]}\n\n" if with_answer else text


def mmlu_header(subject):
    return f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"


def build_mmlu(test, dev, validation, shots, num_prompts, seed, drop=()):
    rng = random.Random(seed)
    by_subject = collections.defaultdict(list)
    for row in test:
        by_subject[row["subject"]].append(row)
    dev_by = collections.defaultdict(list)
    for row in dev:
        dev_by[row["subject"]].append(row)
    val_by = collections.defaultdict(list)
    for row in validation:
        val_by[row["subject"]].append(row)
    subjects = sorted(by_subject)
    order = {s: rng.sample(by_subject[s], len(by_subject[s])) for s in subjects}
    # Round-robin over the subjects so the sample is uniform over them.
    picked, i, exhausted = [], 0, set()
    while len(picked) < num_prompts and len(exhausted) < len(subjects):
        s = subjects[i % len(subjects)]
        i += 1
        while order[s] and id(order[s][-1]) in drop:
            order[s].pop()
        if not order[s]:
            exhausted.add(s)
            continue
        picked.append(order[s].pop())
    rows = []
    for row in picked:
        s = row["subject"]
        demos = (dev_by[s] + val_by[s])[:shots]
        prompt = mmlu_header(s) + "".join(mmlu_example(d, True) for d in demos) + mmlu_example(row, False)
        rows.append({"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": LETTERS[row["answer"]]}],
                     "_id": id(row), "subject": s})
    return rows


def fit_shots(build, tok, shots, max_shots, target, max_tokens, name):
    """Raise the shot count while the mean is below 0.9 x target; return rows and lengths."""
    while True:
        rows = build(shots)
        lengths = [len(tok(r["conversations"][0]["value"])["input_ids"]) for r in rows]
        mean = sum(lengths) / len(lengths)
        print(f"[{name}] shots={shots}: {stats(lengths)}")
        if mean >= 0.9 * target or shots >= max_shots:
            return shots, rows, lengths
        shots += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gsm8k-test"); p.add_argument("--gsm8k-train")
    p.add_argument("--mmlu-test"); p.add_argument("--mmlu-dev"); p.add_argument("--mmlu-validation")
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-tokens", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--gsm8k-shots", type=int, default=8); p.add_argument("--gsm8k-max-shots", type=int, default=12)
    p.add_argument("--mmlu-shots", type=int, default=5); p.add_argument("--mmlu-max-shots", type=int, default=8)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    os.makedirs(a.out_dir, exist_ok=True)
    report = {}

    if a.gsm8k_test:
        test = load(a.gsm8k_test)
        train = load(a.gsm8k_train) if a.gsm8k_train else None
        if train is None:
            print(f"[gsm8k] no train split: the first {a.gsm8k_shots} test questions are the shots and are excluded from the targets")
        shots, rows, lengths = fit_shots(lambda k: build_gsm8k(test, train, k, a.num_prompts, a.seed), tok,
                                         a.gsm8k_shots, a.gsm8k_max_shots, a.target_tokens, a.max_tokens, "gsm8k")
        keep = [i for i, n in enumerate(lengths) if n <= a.max_tokens]
        rows, lengths = [rows[i] for i in keep], [lengths[i] for i in keep]
        out = os.path.join(a.out_dir, "gsm8k1k_sharegpt.json")
        json.dump(rows, open(out, "w"))
        report["gsm8k1k"] = {"file": out, "shots": shots, "shot_source": "train (sampled per prompt)" if train else "first test questions",
                             "tokens": stats(lengths)}
        print(f"[gsm8k] {len(rows)} prompts, {shots} shots -> {out}")

    if a.mmlu_test:
        test, dev = load(a.mmlu_test), load(a.mmlu_dev)
        validation = load(a.mmlu_validation) if a.mmlu_validation else []
        drop = set()
        shots, rows, lengths = fit_shots(lambda k: build_mmlu(test, dev, validation, k, a.num_prompts, a.seed), tok,
                                         a.mmlu_shots, a.mmlu_max_shots, a.target_tokens, a.max_tokens, "mmlu")
        # Drop prompts over the cap and resample at the chosen shot count until the set is full.
        for _ in range(8):
            over = [r["_id"] for r, n in zip(rows, lengths) if n > a.max_tokens]
            if not over:
                break
            drop.update(over)
            rows = build_mmlu(test, dev, validation, shots, a.num_prompts, a.seed, drop)
            lengths = [len(tok(r["conversations"][0]["value"])["input_ids"]) for r in rows]
        keep = [i for i, n in enumerate(lengths) if n <= a.max_tokens]
        rows, lengths = [rows[i] for i in keep], [lengths[i] for i in keep]
        subjects = collections.Counter(r["subject"] for r in rows)
        for r in rows:
            r.pop("_id"); r.pop("subject")
        out = os.path.join(a.out_dir, "mmlu1k_sharegpt.json")
        json.dump(rows, open(out, "w"))
        report["mmlu1k"] = {"file": out, "shots": shots, "dropped_over_max_tokens": len(drop), "subjects": len(subjects),
                            "per_subject_min_max": [min(subjects.values()), max(subjects.values())], "tokens": stats(lengths)}
        print(f"[mmlu] {len(rows)} prompts over {len(subjects)} subjects, {shots} shots, {len(drop)} dropped (> {a.max_tokens} tokens) -> {out}")

    json.dump(report, open(os.path.join(a.out_dir, "build_1k_prompts_stats.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
