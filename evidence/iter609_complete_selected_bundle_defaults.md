# Iteration 609 evidence

The production default now resolves the two previously selected fast paths
from the same bundle used by compact W13:

```python
SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL = os.environ.get(
    "V4_SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL",
    "1" if SINGLE_LAUNCH_COMPACT_W13_BUNDLE else "0",
) == "1"

SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS = os.environ.get(
    "V4_SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS",
    "1" if SINGLE_LAUNCH_COMPACT_W13_BUNDLE else "0",
) == "1"
```

Compatibility contract:

- `V4_SINGLE_LAUNCH_TP4=1` with no overrides: both resolve true.
- Ordinary multi-kernel import: both continue to resolve false.
- `V4_SINGLE_LAUNCH_COMPACT_W13_BUNDLE=0`: both resolve false.
- Either component can still be explicitly forced to `0` or `1`.

Static syntax gate completed successfully:

```
python3 -m py_compile v4_flash_tp_wgmma.py
exit 0
```
