"""Run real prompts through an SGLang engine so the topk dump hook records routing.

Usage (inside the serving container, hook installed, MEGAMOE_TOPK_DUMP_DIR set):
  python capture_routing.py --model-path <ckpt> --tp 8 --dataset gsm8k \
      --data /path/gsm8k_test.parquet --num-prompts 512 --batch 64 [--extra "--mem-fraction-static 0.8"]

With --max-new-tokens 1 only prefill matters: prompts are sent in batches so
each chunked-prefill step carries a few thousand tokens, which is what the
EPLB benchmark replays per rank. With --max-new-tokens N > 1 every batch also
runs N decode steps with --batch concurrent sequences, which records the
decode routing (set MEGAMOE_TOPK_DUMP_MIN_TOKENS=1 so small batches are kept).
"""
import argparse
import json
import shlex
import time


def load_prompts(dataset, path, n):
    if dataset == 'gsm8k':
        import pyarrow.parquet as pq
        t = pq.read_table(path).to_pydict()
        prompts = [f"Question: {q}\nAnswer:" for q in t['question']]
    elif dataset == 'sharegpt':
        rows = json.load(open(path))
        prompts = []
        for r in rows:
            conv = r.get('conversations', [])
            if conv and conv[0].get('from') == 'human' and len(conv[0]['value']) > 200:
                prompts.append(conv[0]['value'][:12000])
    else:
        raise ValueError(dataset)
    return prompts[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--tp', type=int, default=8)
    p.add_argument('--dataset', choices=['gsm8k', 'sharegpt'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--num-prompts', type=int, default=512)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--extra', default='')
    p.add_argument('--max-new-tokens', type=int, default=1)
    a = p.parse_args()

    import sglang as sgl
    engine_kwargs = dict(model_path=a.model_path, tp_size=a.tp, trust_remote_code=True,
                         disable_cuda_graph=True, log_level='warning')
    extra = shlex.split(a.extra)
    i = 0
    while i < len(extra):
        key = extra[i].lstrip('-').replace('-', '_')
        if i + 1 < len(extra) and not extra[i + 1].startswith('--'):
            engine_kwargs[key] = type_cast(extra[i + 1])
            i += 2
        else:
            engine_kwargs[key] = True  # bare flag such as --disable-radix-cache
            i += 1
    engine = sgl.Engine(**engine_kwargs)
    prompts = load_prompts(a.dataset, a.data, a.num_prompts)
    print(f'{len(prompts)} prompts from {a.dataset}', flush=True)
    t0 = time.time()
    for i in range(0, len(prompts), a.batch):
        engine.generate(prompts[i:i + a.batch], {'max_new_tokens': a.max_new_tokens, 'temperature': 0, 'ignore_eos': a.max_new_tokens > 1})
        print(f'  batch {i // a.batch}: {time.time() - t0:.1f}s', flush=True)
    engine.shutdown()


def type_cast(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    return v


if __name__ == '__main__':
    main()
