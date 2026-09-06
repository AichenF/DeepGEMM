# Iteration 709t — compiler operand fences do not constrain ptxas scheduling

## Exact build/result

- Source SHA256:
  `f8890202e7cdedc8c372b0515a1ac6a8dfc202e65f931ac4d4c648ba998cd1ae`.
- Extension: `v4tp_833ae1a52d617e1453dd_v178mspec`.
- Exact M128/split-K2 entry remains `REG56 STACK32 SHARED2048 LOCAL0` with
  the candidate's 19,456-byte dynamic shared allocation.
- Exact SASS SHA256:
  `43a59d14ee50530f69c484ef44d5c05df745c3bf40d5361e1064c49f087837f4`.
- Static instructions remain 64 QGMMAs and 64 dependency barriers across the
  two emitted W2 paths.

The critical region is unchanged: first QGMMA `0x6670`, immediate wait
`0x6680`, delayed second-group `LDS.64` at `0x6690`, second QGMMA at `0x66b0`.
CuTe's empty inline-assembly operand constraint influences NVCC register
liveness but does not prevent ptxas from scheduling the load after the first
machine-level wait.

## Decision

**REJECT before CUDA launch/timing.**  Stop relying on separate C++/PTX asm
statements to express the pair.  The next static probe will emit both QGMMAs
inside one inline-PTX block and mark all eight FP32 accumulator operands
early-clobber, explicitly forbidding accumulator/source register overlap.

Raw artifacts:

- `bench/results/iter709t_w2_pair_operand_fence_jit.log`
- `bench/results/iter709t_w2_pair_operand_fence_resources.log`
- `bench/results/iter709t_w2_pair_operand_fence_m128_split2.sass`
