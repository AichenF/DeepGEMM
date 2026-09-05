# Iteration 605 evidence

Long TP4 all-M cold-L2 selection gate. Each process contains 300 candidate and 300 same-source multi samples per M; every replay has a separate excluded 256 MiB cache clear. Process ordering brackets two ON windows with two OFF windows.

| M | OFF_A | ON_A | ON_B | OFF_B | normalized ON improvement |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.068407 | 1.067446 | 1.067101 | 1.070207 | 0.19% |
| 16 | 1.087346 | 1.088285 | 1.088157 | 1.089736 | 0.03% |
| 32 | 1.134473 | 1.123479 | 1.124766 | 1.132070 | 0.81% |
| 64 | 1.127874 | 1.126498 | 1.124263 | 1.126939 | 0.18% |
| 128 | 1.132513 | 1.126415 | 1.125455 | 1.132464 | 0.58% |

Values are candidate/control latency ratios. Aggregate mean ratio improves 0.3575%, while direct candidate geometric-mean window averages improve 0.332%. Large-M absolute latencies drift with the shared system, so the same-window normalized ratios are authoritative.

