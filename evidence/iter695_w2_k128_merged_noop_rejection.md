# Iteration 695: W2 K128 merged-group no-op rejection

Five OFF and five ON M128 processes were interleaved on physical H20 GPU1.
Each process used the same random routes/seed, prequantized FP8-E4M3 input,
FP32 group-128 scales, MXFP4 weights, and a separate excluded 256 MiB clear
before its phase-stamped launch.  All outputs were bitwise equal to the
independent same-source multi reference.

W2 OFF/ON medians are both 106.240 us.  Means are 106.227/106.598 us, making
the candidate 0.35% slower.  Complete route+W13+requant+W2 medians are
326.528/327.520 us (+0.304%), and means are 326.746/327.590 us (+0.259%).

The extracted M128/split-K2 SASS for OFF and ON has the same SHA-256,
`bfbf7d2677e3dad303e09716cbc1568a5686ad44028b5e84927a78017acdae90`.
ptxas already produces the same machine schedule, so the source-level merge
is a no-op.  The candidate is rejected before TP4 timing and production is
restored byte-for-byte.
