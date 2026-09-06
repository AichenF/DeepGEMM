# Iteration 693: W2 K128 merged-group launcher failure

The default-off candidate changes only the flat TP4 single-launch W2 route
GEMM: four K32 RS-WGMMA operations per K128 tile share one commit/wait group.
The same-source standalone/multi W2 template remains unchanged.

`python -m py_compile` passed.  Candidate source SHA-256 is
`85c7005b63ef5c2264b13ddc6cda1dbbeb969cc04bb5ff0eb989d2b610362bf6`, and
JIT emitted `v4tp_58c93f6e47c09fccaf68_v178mspec`.

The first compute-profile invocation did not reach CUDA.  It bypassed
`v4_bench_env_runner.py`, and SGLang import failed because that Python
environment did not contain `orjson`.  Consequently this record contains no
resource, correctness, cold-L2, or performance claim.  Retry through the
established environment runner.
