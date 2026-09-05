# Iteration 599 evidence

M128 TP4 cold-L2 CUDA-graph bracket for the compact-ABI W13 outline. The cache clear is a separate excluded 256 MiB operation immediately before every implementation replay. Candidate/control ratios remove most window drift.

| Window | Variant | Candidate median ms | Same-window multi median ms | Candidate / control |
|---|---:|---:|---:|---:|
| OFF_A | inline, 8 CTA/SM | 0.3434560001 | 0.2985279858 | 1.1504985007 |
| ON_A | compact outline, 9 CTA/SM | 0.3412960023 | 0.2983359993 | 1.1439987233 |
| ON_B | compact outline, 9 CTA/SM | 0.3422560096 | 0.2992639989 | 1.1436591467 |
| OFF_B | inline, 8 CTA/SM | 0.3441760093 | 0.2993279994 | 1.1498289837 |

Mean normalized ON/OFF = 1.1438289349966213 / 1.150163742213174 = 0.99449226, or 0.5508% improvement. This is a screening result, not yet the selection result.

Correctness was identical in every window: cosine_min 0.999995608996884, rel_l2_max 0.0029634544238714383, finite on all ranks, embedded all-reduce passed, 768 routed rows padded to 1944.

