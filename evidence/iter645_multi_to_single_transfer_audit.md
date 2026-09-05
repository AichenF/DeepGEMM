# Iteration 645: multi-kernel to one-kernel transfer audit

This is a source, Git-topology, cubin, and historical-evidence audit.  It does
not launch a CUDA kernel and makes no new latency claim.

## Git topology

`tpmoe_multikernel_baseline` is an exact ancestor of
`tpmegamoe_single_launch`:

```text
multi tip / merge base: 071abc27ac106803c76ad02bb9fb5586426f6922
multi-only commits:     none
one-kernel audited tip: 67e245c
```

There is therefore no literal multi-only commit to cherry-pick.  The relevant
question is whether a code-generation or scheduling property of the
standalone path can be reproduced inside the one business launch.

## Shared compute body

Both standalone GEMMs and the one-kernel phases instantiate the same
`route_gemm_task` template in `v4_flash_tp_wgmma.py`:

- standalone `route_gemm` is a thin global wrapper that calls one task using
  `blockIdx.x`;
- selected M128 one-kernel W13 calls
  `single_launch_w13_gemm_phase_compact`, whose grid-stride loop calls that
  same task body;
- one-kernel W2 calls the same task body directly from its grid-stride loop.

Consequently the selected interleaved MXFP4 weight/scale layout, Mode-2
decode/braid, TMA staging, S2R prefetch, WGMMA sequence, epilogues, and
assume-valid task guard deletion are already shared.  Copying the standalone
body would duplicate code, not transfer an optimization.

## Exact cubin comparison

Audited production extension:

```text
/tmp/torch_ext_v4_tp/v4tp_db6bdbe1b363bb873f88_v178mspec/
  v4tp_db6bdbe1b363bb873f88_v178mspec.so
```

The extracted SM90a cubin was
`/tmp/multi_one_audit_644/cuda.sm_90a.cubin`.

| Function / phase | text bytes | registers | stack bytes | shared bytes | local bytes |
|---|---:|---:|---:|---:|---:|
| standalone W13 TP4, K4096 N1024 split-K2 | 23,552 | 47 | 0 | 2,048 | 0 |
| one-kernel compact W13 device callee | 23,888 | caller contract | caller contract | caller contract | caller contract |
| standalone W2 TP4, K512 N4096 | 21,504 | 61 | 0 | 2,048 | 0 |
| one-kernel M128 split-K2 entry | 71,680 | 56 | 32 | 2,048 | 0 |

The compact W13 callee is only 336 bytes, approximately 21 SM90 SASS
instructions, larger than the standalone W13 kernel.  Those instructions are
the grid-stride loop/task-ordinal plumbing, not another GEMM algorithm.

For a second check, the rejected whole-W2-phase outline cubin from Iterations
375--378 was inspected at
`/tmp/w2_outline_audit_644/cuda.sm_90a.cubin`.  Its M128 W2 device callee is
20,944 bytes versus the standalone W2 kernel's 21,504 bytes.  Thus W2 also
does not contain an unported standalone math body.

## What produces the multi-kernel advantage

For the Iteration-612 M128 route population, 1,992 padded rows make 249
M-blocks.  Standalone launch geometry creates one fresh CTA for every task:

```text
W13: 249 * 8 N-tiles * split-K2 = 3,984 CTAs
W2:  249 * 32 N-tiles           = 7,968 CTAs
```

The selected one-kernel geometry must keep its whole cooperative/barrier grid
resident: 702 CTAs x 128 threads.  Each resident CTA therefore executes five
or six W13 tasks and eleven or twelve W2 tasks.  The standalone path obtains,
without software machinery:

1. hardware work distribution of one task per newly scheduled CTA;
2. a phase-specific compilation contract (W13 47 registers, W2 61
   registers, both stack-free);
3. kernel-completion global visibility and phase reset.

The one-kernel path instead has one fixed 56-register/32-byte-stack launch
contract, persistent grid-stride task loops, and software whole-grid
publication.  These properties cannot be copied as ordinary source snippets:
launching all 3,984 or 7,968 CTAs while retaining an in-kernel whole-grid
barrier would deadlock once nonresident CTAs were included.

## Transfer matrix

| Multi-kernel property | One-kernel status | Action |
|---|---|---|
| MXFP4/TMA/WGMMA GEMM body | already shared | no cherry-pick |
| W13/W2 epilogues and activation quantization task body | already shared | no cherry-pick |
| task validity guard deletion | already transferred | keep selected |
| CustomAllReduceV2 transport policy | one-shot multicast push for small M and P2P two-shot for large M already embedded | no missing transport snippet |
| one-task-per-fresh-CTA hardware scheduling | incompatible with resident one-launch grid barrier | requires a different global scheduling architecture |
| per-phase 47/61-register, stack-free contracts | incompatible with one fixed entry resource contract | cannot direct cherry-pick |
| kernel-exit phase fence/reset | replaced by software grid barriers | cannot direct cherry-pick |
| W2 whole-phase outline | already tested; noise-sized gain and replay instability | reject |

## Only bounded untested adaptation

A compact, shared-record ABI for the existing whole-W2-phase outline could
mirror the selected compact W13 ABI and may keep the caller frame at 32 bytes.
It is not a missing multi-kernel optimization: the prior high-argument W2
outline already generated a callee as small as the standalone body, improved
M128 by only 0.12% in its cold TP4 bracket, and then failed the strict replay
correctness envelope (`rel_l2` 0.00653--0.00761).  It therefore has low
expected value and must remain a default-off, correctness-first experiment if
attempted.

## Decision

There is no production-worthy direct cherry-pick from the multi-kernel branch.
The compute body and communication policy are already present.  The remaining
same-source multi advantage is structural: fresh CTA scheduling, independent
phase resource contracts, and kernel-completion synchronization.  Preserve
the selected one-kernel default and do not replace it with the previously
rejected W2 outline merely because its source resembles the standalone
wrapper.
