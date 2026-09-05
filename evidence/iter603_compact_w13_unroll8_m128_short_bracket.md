# Iteration 603 evidence

TP4 M128 compact-W13 four-way versus eight-way K-loop bracket, each normalized against an exact Humming MXFP4 plus SGLang CARv2 graph in the same process. Every replay receives its own separate excluded 256 MiB L2 clear.

| Window | W13 split2 unroll | Candidate median ms | Humming median ms | Candidate / Humming |
|---|---:|---:|---:|---:|
| OFF_A | 4 | 0.3483359963 | 0.3750559986 | 0.9287572992 |
| ON_A | 8 | 0.3535839915 | 0.3744639903 | 0.9442403026 |
| ON_B | 8 | 0.3532160074 | 0.3744319975 | 0.9433382021 |
| OFF_B | 4 | 0.3468319923 | 0.3747199923 | 0.9255764288 |

Mean normalized ON/OFF is approximately 1.017927: eight-way unrolling regresses the compact single kernel by 1.79%. The external baseline is stable within 0.17%, and the direct custom medians agree with rejection.

