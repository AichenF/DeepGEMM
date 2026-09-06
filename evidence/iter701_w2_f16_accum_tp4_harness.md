# Iteration 701: paired TP4 graph harness admission

`bench/compare_v4_flash_tp_w2_store.py` now recognizes
`V4_SINGLE_LAUNCH_W2_F16_WGMMA_ACCUM` as a compare flag and as an explicitly
tolerance-qualified numerical experiment. The same process loads FP32 OFF
first, then the identical source with only FP16 ON; weights, FP8 activation,
routes and the SGLang CustomAllReduceV2 communicator are shared.

Both complete one-kernel MegaMoE+collective CUDA Graphs are captured before
timing. Every replay gets its own separate 256 MiB L2 clear, excluded from
the CUDA events, and order alternates A/B then B/A per sample. Timing cannot
start unless every rank is finite with cosine >= 0.99999 and relative L2 <=
0.005; exact equality remains visible in output.

This is a harness-only iteration with no CUDA or latency result.
