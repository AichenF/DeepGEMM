# Iteration 634: W2-only persistent-state cubin resource gate

## Candidate

`V4_SINGLE_LAUNCH_W2_PERSISTENT_STATE=1` reuses the existing persistent
route-GEMM state machine only in the inline W2 task loop.  The production
compact W13 outline, route construction, SwiGLU/requantization and embedded TP
collective are unchanged.  Cross-task weight prefetch is disabled.

## Artifact and protocol

- GPU: physical H20 GPU1, SM90a.
- Extension:
  `/tmp/torch_ext_v4_tp/v4tp_f72712dd95f6ccb3ff4e_v178mspec/v4tp_f72712dd95f6ccb3ff4e_v178mspec.so`.
- Inspection: `cuobjdump --dump-resource-usage` on every emitted TP4
  single-launch token/SplitK specialization.
- This gate compiled and inspected the cubin only.  It did not launch a CUDA
  kernel and makes no correctness or performance claim.

## Resource result

| Tokens | SplitK | Registers | Stack | Static shared | Fixed local |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 / 4 | 56 | 32 B | 2,048 B | 0 B |
| 64 | 2 / 4 | 64 | 32 B | 2,048 B | 0 B |
| 32 | 2 | 62 | 32 B | 2,048 B | 0 B |
| 32 | 4 | 61 | 32 B | 2,048 B | 0 B |
| 16 | 2 | 62 | 32 B | 2,048 B | 0 B |
| 16 | 4 | 61 | 32 B | 2,048 B | 0 B |
| 8 | 2 | 62 | 32 B | 2,048 B | 0 B |
| 8 | 4 | 61 | 32 B | 2,048 B | 0 B |

M128 retains the production 56-register bound required for 702x128 (nine
CTAs per each of 78 SMs).  There is no fixed local allocation in any emitted
entry.  The resource gate therefore passes; runtime occupancy, barrier-state
reuse and exact output remain to be proven by the next compute-only gate.
