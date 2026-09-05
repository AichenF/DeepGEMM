# Iteration 650a: NCU launcher failure before target execution

## Intended profile

Profile the selected TP4 M128 one-kernel compute body on physical H20 GPU 1
with application-managed excluded 256 MiB cold-L2 clearing and NCU
Launch/Occupancy/Scheduler/WarpState/SourceCounters sections.  The TP
collective was to be disabled only for local phase attribution.

## Result

NCU exited before starting Python or launching CUDA because the target command
named the script directly instead of prepending the interpreter:

```text
==ERROR== The target application is not an executable binary.
==ERROR== If the target application is a script, try prepending the full path to the interpreter binary.
```

No extension import, reference computation, L2 clear, CUDA kernel, correctness
check, profile pass, or latency measurement ran.  This result carries no
performance or correctness evidence.

## Decision

Retry the identical profile with `/usr/bin/python3` immediately before
`bench/profile_v4_flash_tp_single_compute.py`.  Keep all kernel flags,
physical GPU, route seed, NCU sections, and cold-L2 policy unchanged.
