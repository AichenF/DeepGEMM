# Iteration 643: TP4 timing harness invocation failure

The intended five-batch M128 paired cold-L2 run did not start.  `torchrun`
interpreted the benchmark option `--m 128` as one of its own ambiguous
`--max-*`, `--monitor-*`, `--module`, or `--master-*` options because the
training-script argument separator was missing.

No Python benchmark body and no CUDA kernel executed.  The retry must retain
the same GPUs, inputs, graph protocol, sample counts, and candidate and add
only `--` before the benchmark arguments.
