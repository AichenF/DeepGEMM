# SM90 NVFP4 Layout Policy and Wrapper Cleanup

## Contract

`block_n` is a deployment-time weight-layout policy. The framework selects one
layout while loading the model, prepackages one copy of the NVFP4 weights, and
uses that layout for every request served by the instance. BN256 selects the
fused kernel and BN128 selects split L1/L2. Request-time layout switching and
dual weight copies are out of scope.

## Selected Approach

Use an isolated correctness-matrix driver, then apply behavior-preserving JIT
and host-config cleanup, and only then measure layout crossovers. Alternatives
were rejected: keeping both layouts costs model-weight memory, while selecting
inside the request wrapper cannot work without a matching physical layout.

## Correctness Guard

Run each shape in a fresh process and JIT cache:

- Flash (`I=2048`): cover expected-token boundaries 3 and 8.
- Pro (`I=3072`): cover the expected-token boundary 8.
- Middle (`I=2560`): cover the generic swapAB boundary 8.
- Forced BN128: cover expected 32, 64, 96, 128, and 192.
- Run both no-global-scale and per-expert-global-scale modes.

## Behavior-Preserving Cleanup

- Remove the eleven trailing generated template arguments that exactly repeat
  defaults in the fused and split kernel declarations.
- Emit parameter-name comments beside generated phase booleans.
- Add an NVFP4-specific config builder that constructs the final block, thread,
  pipeline-stage, and shared-memory configuration once. Leave the generic FP8
  SM90 config helper unchanged.
- Preserve forced-layout behavior. BN128 low/mid-expected policies are not dead
  merely because the default deployment chooser currently selects BN128 above
  expected 192.

## Layout Sweep

For `I={2048,2304,2560,2816,3072}` and
`expected={128,160,192,224,256}`, compare forced BN256 and BN128 with ABBA order
and multiple routing seeds. Use the same H20 eight-rank harness, fixed capacity,
L2 flush, and robust per-run statistic used by the existing range-policy work.
Record raw runs and summaries outside the repository.

Choose the smallest policy justified by the data: a shared cutoff if it is
stable, otherwise a three-class table for Flash-like, middle, and Pro-like
shapes. Do not infer a linear formula from `intermediate_hidden`.

## Measured Result

The completed 54-point sweep used expected work
128/160/176/184/192/200/208/216/224/256 for all five I values, plus a denser
four-seed Pro crossover sweep. Flash and all three middle-I profiles crossed at
expected 192: BN256 won every seed at 184, BN128 won every seed at 200, and 192
was the near-tie point. Pro crossed slightly earlier: BN256 still won 3/4 seeds
at 190, while BN128 won at 192 and 4/4 seeds at 194.

The selected deployment table is therefore:

- `I < 3072`: BN256 when expected work is at most 192, otherwise BN128.
- `I >= 3072`: BN256 when expected work is at most 190, otherwise BN128.

The comparison is implemented with routed-token integers. There is no linear
I scaling and no request-time layout switch. Expected 256 showed a roughly 1%
non-monotonic result for Flash and Pro; a four-experts-per-wave experiment did
not remove it, so no narrow point exception was added.

Raw runs and generated summaries are under
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/layout_sweep/`.

## Acceptance

- `git diff --check` and `./develop.sh` pass.
- The policy correctness matrix passes from fresh JIT caches.
- Cleanup-only cubins retain resource usage and executable SASS, excluding
  symbol or source-line diagnostic changes.
- The final layout policy is documented as deployment-time and supported by
  the recorded crossover measurements.
