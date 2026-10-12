"""Install the routing dump hook into an installed sglang (idempotent).

Wraps the module-level ``select_experts`` in sglang/srt/layers/moe/topk.py,
which every TopK forward path calls with ``layer_id=`` and ``hidden_states=``,
so it works across sglang versions regardless of the TopK class layout.

Usage: python install_topk_hook.py [path/to/sglang/srt/layers/moe/topk.py]
"""
import importlib.util
import os
import re
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    spec = importlib.util.find_spec('sglang')
    path = os.path.join(os.path.dirname(spec.origin), 'srt', 'layers', 'moe', 'topk.py')
src = open(path).read()
if '_dump_topk_ids' in src:
    print(f'hook already present in {path}')
    sys.exit(0)
assert re.search(r'^def select_experts\(', src, re.M), 'select_experts not found'
snippet = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sglang_topk_dump_snippet.py')).read()
wrapper = '''

_select_experts_without_dump = select_experts


def select_experts(*args, **kwargs):
    out = _select_experts_without_dump(*args, **kwargs)
    hs = kwargs.get("hidden_states", args[0] if args else None)
    _dump_topk_ids(kwargs.get("layer_id", -1), out, hs.shape[0] if hs is not None else 0)
    return out
'''
open(path, 'w').write(src.rstrip('\n') + '\n\n' + snippet + wrapper)
print(f'hook installed in {path}')
