# Iteration 638: exact compact-W13 persistent-state resource gate

## Artifact

```text
/tmp/torch_ext_v4_tp/v4tp_7babc76b617baae01065_v178mspec/
v4tp_7babc76b617baae01065_v178mspec.so
```

The path is specified directly from the candidate identity printed during
Iteration 637; no mtime or directory-order heuristic is used.

## Result

| Tokens | SplitK | Registers | Stack | Static shared | Fixed local |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 / 4 | 56 | 32 B | 2,048 B | 0 B |
| 64 | 2 / 4 | 64 | 32 B | 2,048 B | 0 B |
| 32 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |
| 16 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |
| 8 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |

M128 retains the 56-register limit required by the selected nine-CTA/SM
launch.  The compact outlined call retains its 32-byte stack frame and no
single-launch specialization declares local allocation.  This is a static
resource result only; persistent mbarrier correctness and latency are not yet
claimed.
