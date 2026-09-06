# Iteration 708: exact restore after W13 FP16 rejection

All W13-FP16-specific source introduced in Iteration 706 was removed,
including its environment/JIT plumbing and template arguments. The generic
F16 specialization is again statically limited to flat W2 tasks.

The restored source is byte-for-byte SHA-256
`fc6cc41d86d6e77c0f4d5d54efc2dace24e457115d94e32e4de0f678387da79f`,
the W2-only source already compiled and fully measured in Iterations 699–705.
Python syntax validation passes. Selected default FP32 behavior is unchanged;
W2 FP16 remains available only through its explicit default-off flag.
