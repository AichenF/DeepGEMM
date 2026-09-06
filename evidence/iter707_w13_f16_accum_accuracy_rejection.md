# Iteration 707: reject W13 packed-FP16 accumulation

The W13-only candidate compiled as
`v4tp_e8d69416044af98025ab_v178mspec` and ran the complete local M128
route/W13/SwiGLU-requant/W2 path on physical H20 GPU1 after a separate
excluded 256 MiB L2 clear. TP communication was disabled only for arithmetic
isolation.

Against the unaffected same-source FP32 multi path, complete `down` reports
cosine `0.9999705045196186`, relative L2 `0.007681265485016974`, finite
values, 1,992 padded rows and correct packed-barrier wrap. Relative L2 exceeds
the formal graph gate of 0.005, so no performance timing is admitted.

The exact M128/split-K2 top-level kernel remains `REG56 STACK32 SHARED2048
LOCAL0 CONSTANT[0]1361`; there is no occupancy-tier improvement to offset the
accuracy loss. The probe is rejected and will be removed, restoring the
W2-only source SHA
`fc6cc41d86d6e77c0f4d5d54efc2dace24e457115d94e32e4de0f678387da79f`.
