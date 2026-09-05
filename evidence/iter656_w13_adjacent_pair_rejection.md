# Iteration 656: sequential adjacent-N W13 pair rejection

## Design

The experiment retained the selected TP4 M128 compact W13 phase, one
128-thread WGMMA warpgroup, one accumulator set and the existing
nonpersistent route-GEMM task.  Only the static task permutation changed.
For each grid-stride pair, a CTA executed two adjacent N128 tiles from the
same mblock and K split before moving to its next pair.  For 1,992 padded rows
this kept 3,984 W13 tasks on 702 CTAs with the same six-task critical path.

The intended benefit was L1 reuse of the just-read 8 x 2,048-byte activation
slice without the extra registers, shared memory, wider CTA or cohort
barriers of earlier simultaneous dual-tile experiments.

## Gates

The opt-in extension was
`/tmp/torch_ext_v4_tp/v4tp_5847ec4147e82b6695e0_v178mspec`.
Exact M128 resources remained:

```text
split-K2: REG56 STACK32 SHARED2048 LOCAL0
split-K4: REG56 STACK32 SHARED2048 LOCAL0
```

The local random-route M128/split-K2 test after an excluded 256 MiB L2 clear
passed bitwise against the independent same-source multi route output:

```text
cosine=1, rel_l2=0, finite=true, padded_rows=1992
packed_generation_words_after=[0,0,0,0]
```

Two preliminary attempts failed before CUDA execution because the repository
root was absent from `PYTHONPATH` and then because the container-default Torch
lacked `float8_e8m0fnu`.  The valid run used the established Miniforge Torch
2.11 environment; those import failures make no kernel claim.

## Cold-L2 phase result

Physical H20 GPU0, seed 20260902, random M128 routes, device phase stamps and
eight independent processes ordered OFF/ON/ON/OFF/OFF/ON/ON/OFF.  Every
measured one-kernel launch was immediately preceded by a separate excluded
256 MiB L2 clear.

| W13 phase | min us | median us | max us | mean us |
|---|---:|---:|---:|---:|
| selected OFF | 211.936 | 213.600 | 214.464 | 213.400 |
| adjacent pairs ON | 227.040 | 227.424 | 229.920 | 227.952 |

The candidate is 6.47% slower by median and 6.82% slower by mean.  All four
candidate samples are slower than all four controls, while route,
activation/requant and W2 remain in their expected ranges.

## Verdict

Reject without a four-rank run.  Same-CTA activation locality does not repay
destroying the selected cross-CTA contiguous cold-weight/TMA wave order.
The experiment was removed from source after measurement, leaving production
behavior unchanged.

Raw records:

- `bench/results/iter656_w13_adjacent_pairs_jit_20260906.log`
- `bench/results/iter656_w13_adjacent_pairs_resources_20260906.log`
- `bench/results/iter656c_w13_adjacent_pairs_m128_correctness_20260906.log`
- `bench/results/iter656_w13_adjacent_pairs_local_phase_abba_20260906.log`
