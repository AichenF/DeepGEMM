# Iteration 714c: unique RDC build and static equivalence

Date: 2026-09-06

The Iteration-714b source copy compiles, device-links with host PIC, and links
into a loadable-shaped CPython extension without error.  The resulting library
SHA256 is
`682d5fe5d18860810fd1e4dfb68926437b3c014f20780337ccaa94eb6a5f1815`.

The unique TP4 M128 split-K2 entry reports:

```text
REG 195, STACK 112, SHARED 1616, LOCAL 0
```

The linked `single_launch_w2_gemm_phase<true>` still contains 32 QGMMA, 16
DEPBAR and 32 warpgroup-arrive instructions.  Its complete extracted SASS is
byte-identical to Iteration 713b, SHA256
`adf4eb6fd60c7954ca7c2bec966d6dc005c160d0f78d66fbc27804daca7f418e`.
Thus unique host/kernel naming removes dynamic symbol ambiguity without
changing the candidate's device code or resource contract.

Build log:
`bench/results/iter714c_unique_rdc_build_20260906.log`.
Resource dump:
`bench/results/iter714c_unique_rdc_resources_20260906.log`.
No CUDA business kernel launched during this build/static audit.
