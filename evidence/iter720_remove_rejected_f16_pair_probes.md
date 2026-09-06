# Iteration 720: remove rejected FP16-pair probes

Iterations 718b and 719b reject both the paired FP16 inline issue and the
late-arrive refinement at the static gate.  The latter produces SASS that is
byte-identical to the former, and both retain 32 dependency waits for 32 W2
QGMMAs.  Neither path launched a CUDA business kernel.

This cleanup removes only their default-off environment switches, validation,
template parameters, inline PTX branch, JIT discriminators/defines and
benchmark metadata.  The independently measured
`V4_SINGLE_LAUNCH_W2_F16_WGMMA_ACCUM` option remains available; selected
default math, task ownership, communication and benchmark behavior are
unchanged.

Post-cleanup hashes exactly recover the pre-Iteration-718 production files:

- `v4_flash_tp_wgmma.py`:
  `30e6c402b4dc220d2a4eb03099d1f1cd0f82e59b0af38747b59906d84c3fdbdf`
- `bench/v4_flash_tp_single_vs_multi_graph.py`:
  `16bb9e09d6d23247ea2291f08992f48c1cd394309829913732c6bff9c55901bb`

Python bytecode compilation passes.  No JIT or CUDA execution is needed for
an exact source restoration.
