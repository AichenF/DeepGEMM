# Iteration 712c — restore the buildable one-launch source

Remove the default-off function-level `__maxnreg__` probe rejected by NVCC in
Iteration 712b.  This is an exact source restoration, not a performance
change: kernel and paired-benchmark SHA256 values return to
`30e6c402b4dc220d2a4eb03099d1f1cd0f82e59b0af38747b59906d84c3fdbdf` and
`16bb9e09d6d23247ea2291f08992f48c1cd394309829913732c6bff9c55901bb`.

Both files pass Python bytecode compilation and compare byte-identical to the
last buildable Iteration-711i source pair.  No JIT, CUDA launch, correctness
or latency claim is made.

