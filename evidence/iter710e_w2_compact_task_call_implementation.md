# Iteration 710e — compact per-task W2 call implementation

## Hypothesis

The fused entry and the whole-phase W2 outline both compile one dependency
wait per QGMMA, while the standalone W2 task compiles one wait per two
QGMMAs.  The remaining structural difference is the persistent grid-stride
task loop.  Outline exactly one W2 task, but avoid the rejected thirteen-
pointer ABI by passing a CTA-shared argument record plus the scalar task id.

## Change

- Add default-off `V4_SINGLE_LAUNCH_W2_COMPACT_TASK_CALL` to the JIT identity
  and compiler definitions.
- Reuse the selected M128 compact-W13 shared argument record after the W13 and
  activation phases are globally complete; no extra shared allocation is
  introduced.
- Populate that record with W2 descriptors/pointers once per CTA, synchronize
  the CTA, and call a shape-specialized `__noinline__` W2 task body with only
  `(record_pointer, task_id)`.
- Preserve the existing task order, wave rotation and post-task CTA barrier.
  M=8/16/32/64 specializations retain their current inline W2 path.

## Isolation and gate

The experiment is restricted to the selected TP4 M128 schedule-0,
WOUT128/two-stage, compact-W13, dynamic-route-smem, bound-9 configuration and
rejects alternative W2 pipelines.  It is compile-only until the extracted
callee restores the standalone static signature: 32 QGMMAs, about 16
dependency waits, REG56/no local spill, and the nine-CTA launch tier.  Only
then may correctness or cold-L2 timing run.

Python bytecode compilation passes.  Staged source SHA256:
`93236bc78b9d0585f922cb6078cfeabb9986252a15ce8e9dd9362be63c4925cc`.

