# Iteration 724f — reject terminal no-return W2 after exact SASS

## Build and resource result

Fresh JIT succeeds as extension
`v4tp_4e349730b677b396241b_v178mspec`.  TP4 M128 split-K2 and split-K4 both
report:

```text
REG64 STACK0 SHARED2048 LOCAL0
```

This is a real ABI/resource improvement over the returning compact/whole-W2
calls, which carried STACK48.  The generated M128 split-K2 control flow also
contains the terminal W2 call and its two possible `EXIT` paths, with no W2
callee `RET`.

## Exact W2 gate

The W2 terminal callee begins at SASS PC `0x97f0`, exits early at `0x9800`
for CTAs with no work, loops back at `0xe790`, and ends in `EXIT` at
`0xe7a0`.  Its isolated 329,208-byte interval has SHA256:

```text
8ff5b16dc2b29e42e301adacf8db1052901d0bb2e67bd2b77c4b7c196ef389e2
```

Direct counts are:

```text
QGMMA                    32
WARPGROUP.DEPBAR         32
CALL.REL                  0
RET.REL                   0
EXIT                      2
STL / LDL                 0 / 0
```

The no-return ABI therefore removes stack/return state but does **not**
recover standalone W2's 32-QGMMA / 16-wait schedule.  It fails the declared
static gate decisively.

## Decision

**Reject before CUDA business-kernel launch.**  The candidate intentionally
omits the final grid barrier, k6 combine, TP collective, and replay cleanup,
so executing it would be invalid.  Restore the production source exactly and
do not migrate those semantics into a tail that has no W2 scheduling benefit.

This result closes ordinary returning calls, compact calls, unbounded calls,
bound-eight calls, and now terminal no-return calls as ways to recover the
standalone W2 code generation inside the monolithic code object.  A future
attempt must change the W2 algorithm/dataflow itself rather than its C++ call
boundary.

## Artifacts

- `evidence/iter724c_w2_terminal_noreturn_static_probe.md`
- `bench/results/iter724d_w2_terminal_noreturn_jit_20260906.log`
- `bench/results/iter724e_w2_terminal_noreturn_resources_20260906.log`
- `bench/results/iter724e_w2_terminal_noreturn_m128_split2_exact.sass`
- `bench/results/iter724e_w2_terminal_noreturn_m128_split2_w2_exact.sass`
