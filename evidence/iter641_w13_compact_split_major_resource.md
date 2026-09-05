# Iteration 641: compact-W13 split-major resource gate

## Candidate

`V4_SINGLE_LAUNCH_W13_COMPACT_SPLIT_MAJOR_TASKS=1` changes only the physical
task enumeration inside the selected M128 compact W13 phase. It maps each
split-major ordinal back to the existing route-GEMM logical task index before
executing the unchanged task body.

## Build and resource protocol

- H20 physical GPU1; SM90a JIT.
- Python bytecode compilation for the kernel and paired A/B fixture.
- Exact fresh extension: `v4tp_de18849883f1094462fd_v178mspec`.
- Exact cubin inspected with `cuobjdump --dump-resource-usage`.
- No CUDA kernel launch and no performance timing in this gate.

## Result

| Tokens | SplitK | Registers | Stack | Static shared | Fixed local |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 / 4 | 56 | 32 B | 2,048 B | 0 B |
| 64 | 2 / 4 | 64 | 32 B | 2,048 B | 0 B |
| 32 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |
| 16 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |
| 8 | 2 / 4 | 61 | 32 B | 2,048 B | 0 B |

M128 retains the production 56-register ceiling, 32-byte caller frame, and
zero fixed local allocation required for 702 resident 128-thread CTAs, or
nine CTAs per each of 78 H20 SMs. This is a static resource result only.
