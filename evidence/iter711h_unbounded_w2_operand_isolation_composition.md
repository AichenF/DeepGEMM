# Iteration 711h — natural-register W2 operand-isolation composition

## Hypothesis

The fused SASS allocates a W2 QGMMA's RS source registers on top of its
accumulator destination, forcing an immediate dependency wait.  The existing
predecode + paired issue + one-value shared spill + compiler operand-fence
chain was designed to keep both groups' sources disjoint, but every prior
build was capped at 56 registers.  Recompile that exact chain in the natural
M128 register regime before rejecting the source-lifetime approach.

## Change

Permit `V4_SINGLE_LAUNCH_W2_PREDECODE_S2R=1` when either the selected M128
bound9 path or the new default-off unbounded M128 path is active.  No device
body, arithmetic, task mapping, barrier, collective, public ABI, or default
behavior changes in this iteration.

The static candidate will enable:

```text
V4_SINGLE_LAUNCH_M128_UNBOUNDED=1
V4_SINGLE_LAUNCH_W2_PREDECODE_S2R=1
V4_SINGLE_LAUNCH_W2_PAIR_WGMMA_GROUPS=1
V4_SINGLE_LAUNCH_W2_PAIR_SPILL_ONE=1
V4_SINGLE_LAUNCH_W2_PAIR_OPERAND_FENCE=1
```

Python bytecode compilation passes.  Staged source SHA256:
`30e6c402b4dc220d2a4eb03099d1f1cd0f82e59b0af38747b59906d84c3fdbdf`.

No JIT or CUDA launch is claimed.  Require the spill LDS to precede both
QGMMAs, disjoint source/destination bases, about 16 waits, no local spill and
runtime-admissible residency before correctness or timing.

