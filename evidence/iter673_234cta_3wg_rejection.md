# Iteration 673: reject 234 CTA x three independent WGs

## Motivation

The earlier 156 CTA x four-WG topology reduced the number of physical grid
barrier participants, but also reduced resident WGMMA workers from nine to
eight per H20 SM and compiled at 64 registers.  That result could not isolate
whether packed independent WGs themselves were harmful.

This experiment adds an M-independent `234 CTA x 384 threads` specialization:
three physical CTAs per SM, each containing three private 128-thread
route-GEMM workers.  It therefore preserves exactly nine WGMMA workers per SM
and the same 702 logical worker ordinals as the selected `702 x 128` path,
while reducing physical grid-barrier participants from 702 to 234.

The implementation generalized the already-correct 156x4 machinery only:
private route-task shared slabs, named per-WG CTA barriers, packed route
construction, and the existing W13/requant/W2 bodies.  Task math, data layout,
communication policy, and cold-L2 protocol were unchanged.

## Static and correctness gates

- H20 occupancy query admitted exactly three CTAs/SM.
- TP4 M128 cubin: `REG 56`, `STACK 64`, `LOCAL 0`, `SHARED 2048`.
- The selected `702x128` M128 entry is `REG 56`, `STACK 32`, `LOCAL 0`.
- Local M8 and M128 compute output matched the reference bitwise.
- Packed barrier wrap produced `[0, 0, 0, 0]` at both endpoint shapes.
- Corrected TP4 runs passed final-output and allreduce checks on every rank.

Local logs:

- `bench/results/iter673_234cta_3wg_m8_correctness_20260906.log`
- `bench/results/iter673_234cta_3wg_m128_jit_correctness_20260906.log`
- `bench/results/iter673b_234cta_3wg_selected_flags_m128_correctness_20260906.log`

## Configuration correction

The first distributed candidate run inherited `assume_valid=false` and
`release_grid_arrival=false` when the compact bundle was disabled.  It is
retained as a failed harness attempt and is not the topology verdict:
`bench/results/iter673_234cta_3wg_on_a_tp4_m8_m128_cold_20260906.log`.

The valid topology comparison explicitly restored
`V4_SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS=1` and
`V4_SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL=1`.  Its emitted benchmark metadata
confirms both values.

## TP4 cold-L2 endpoint result

Both runs used random routing, CUDA Graph replay, two outer batches, 20
samples per implementation per batch, maximum latency across ranks, and an
excluded separate 256 MiB L2 clear immediately before every replay.
The same-source multi path is the per-process drift control.

| M | topology | multi (ms) | single (ms) | single / multi |
| ---: | --- | ---: | ---: | ---: |
| 8 | selected 702x128 | 0.071408 | 0.075600 | 1.058705 |
| 8 | corrected 234x384 | 0.071456 | 0.086864 | 1.215629 |
| 128 | selected 702x128 | 0.307984 | 0.351168 | 1.140215 |
| 128 | corrected 234x384 | 0.308096 | 0.407376 | 1.322237 |

After normalizing each single-launch latency by its same-process multi
control, the packed topology is:

- M8: `1.148223x`, or **14.82% slower** than selected.
- M128: `1.159638x`, or **15.96% slower** than selected.
- Endpoint geometric-mean normalized ratio: `1.153916x`, or **15.39% slower**.

Both candidate batch medians agree in direction.  The loss is far larger than
noise, so another bracket is unnecessary.

Valid raw logs:

- `bench/results/iter673_234cta_3wg_off_a_tp4_m8_m128_cold_20260906.log`
- `bench/results/iter673b_234cta_3wg_selected_flags_on_a_tp4_m8_m128_cold_20260906.log`

## Decision

Reject and remove the 234x3 code.  Matching the selected number of resident
WGMMA workers and register count does not make independent WGs inside a
384-thread CTA behave like fresh 128-thread CTAs.  The extra stack frame,
shared CTA scheduling/interference, and packed synchronization dominate the
reduction in grid-barrier participants.

This also closes the simple packed-CTA-count family: 78x8, 156x4, and 234x3
all lose materially.  Any future use of the Hopper MegaMoE 384-thread shape
must transfer its asymmetric loader/math role specialization and dynamic
register redistribution, not merely pack identical route-GEMM workers.

Production source and benchmark were restored byte-for-byte to SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`
and `e39ab6a4b9727f16688668be39372825ebc781ce7320a08e149f2087aa1bba97`.
