# Iteration 653b: production M128 residual-wave placement

## Protocol and recovery

The remote tee log from Iteration 653a was read after the client polling
session disappeared.  It contains a complete accepted record from the
existing default-off `V4_SINGLE_LAUNCH_TRACE_SMID=1` diagnostic:

- physical H20 GPU 1, 78 SM;
- selected TP4 production one-kernel M128 specialization, split-K2;
- random routes and seed 20260902;
- caller-provided FP8-E4M3 X/group-128 scale and MXFP4 weights;
- TP communication disabled only for local placement attribution;
- separate excluded 256 MiB L2 clear immediately before the checked launch;
- no latency measurement or performance claim.

The trace is written only after W2's phase-3 barrier, into dead W13 partial
storage.  It does not change W13/W2 task assignment.

## Correctness and placement

The recovered record is accepted: complete routed `down` is bitwise equal to
the independent same-source multi-kernel local reference (`cosine=1.0`,
`rel_l2=0.0`, finite), with 1,992 padded rows.  Packed phase words are
`[2048,2048,2048,2048]`.

All 702 resident CTAs are present.  Every one of the 78 physical SMs owns
exactly nine CTAs, so the selected M128 launch residency and trace are
complete.

## Residual-wave distribution

M128 has 249 padded M-blocks:

| phase | tasks | resident CTAs | complete waves | residual tasks | residual CTA fraction |
|---|---:|---:|---:|---:|---:|
| W13, eight N128 tiles x split-K2 | 3,984 | 702 | 5 | 474 | 67.52% |
| W2, 32 N128 tiles | 7,968 | 702 | 11 | 246 | 35.04% |

The current grid-stride loop assigns the residual task to CTA IDs below the
residual count.  Mapping those exact CTA IDs through the measured placement
gives:

| phase | residual CTAs per SM histogram | measured min/max | mathematical minimum possible max |
|---|---|---:|---:|
| W13 | 8 SM x 5, 56 SM x 6, 14 SM x 7 | 5 / 7 | `ceil(474/78)=7` |
| W2 | 2 SM x 2, 62 SM x 3, 14 SM x 4 | 2 / 4 | `ceil(246/78)=4` |

For both phases, the measured static mapping already attains the smallest
possible maximum residual-task count per SM.  It is not perfectly even, but
no reassignment can lower the SM-level critical count below seven W13 or four
W2 residual tasks.  The same SM IDs 64--77 own the measured maximum in both
phases.

## Decision

Reject the Iteration-652 residual-wave ticket proposal before implementation.
A global ticket could react to transient CTA timing variation, but it cannot
reduce the phase's maximum per-SM residual work; it would add 702 contended
atomics and a CTA ticket broadcast to an already count-optimal tail.  The
prior all-wave runtime remap was slower, so there is no credible double-digit
or even stable low-single-digit expectation here.

This closes the last direct scheduling adaptation from the multi-kernel path.
The remaining 10% objective requires changing the amount/concurrency of
GEMM and intermediate work—most plausibly a new coarse W13-to-W2 dataflow—not
copying the multi-kernel launch scheduler.
