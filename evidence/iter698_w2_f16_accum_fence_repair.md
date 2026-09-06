# Iteration 698: packed-FP16 WGMMA fence repair

The three FP16-only accumulator dependency fences now call CuTe's native
`uint32_t&` overload.  All FP32 paths still call DeepGEMM's existing
`float&` wrapper, and WGMMA issue, commit and wait placement is unchanged.

Repaired source SHA-256:
`fc6cc41d86d6e77c0f4d5d54efc2dace24e457115d94e32e4de0f678387da79f`.
Python syntax validation passes.  This source-only iteration contains no CUDA
execution or performance claim; it enables retrying the exact Iteration-697
M128 gate.
