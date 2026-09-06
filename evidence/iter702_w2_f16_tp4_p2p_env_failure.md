# Iteration 702: TP4 P2P checker environment failure

The intended M128 paired cold-L2 run on physical H20 GPUs 1,5,6,7 stopped
while constructing SGLang `CustomAllReduceV2`. Because this GPU subset needed
a fresh P2P cache entry, SGLang spawned `custom_all_reduce_utils.py`. The
benchmark runner's pinned imports are installed through the parent process's
`sys.path`; the child inherited only `PYTHONPATH`, selected an unrelated
editable SGLang checkout, and failed with `ModuleNotFoundError: orjson`.

No CUDA Graph was captured and no benchmark kernel/timing executed. The retry
keeps the same GPUs and parameters but exports the runner's overlay, pinned
SGLang source, dflash dependency site-packages, repository and Humming paths
through `PYTHONPATH` so the child process sees the same environment.
