# Iteration 710f — compact per-task W2 call static rejection

## Scope

Compile-only audit of Iteration 710e on H20 GPU1.  No business kernel was
launched, so this iteration makes no numerical or latency claim.

Configuration:

```text
CUDA_VISIBLE_DEVICES=1
TORCH_CUDA_ARCH_LIST=9.0a
V4_SINGLE_LAUNCH_TP4=1
V4_SINGLE_LAUNCH_W2_COMPACT_TASK_CALL=1
```

The JIT extension is `v4tp_5db5ea89ab57f9aab05f_v178mspec`.

## Resource gate

The TP4 M128/split-K2 entry remains `REG56 STACK48 SHARED2048 LOCAL0`, so the
candidate retains the selected nine-CTA register tier and introduces no local
spill or additional static shared memory.  The local compact W2 callee is
20,672 bytes.

## Exact callee SASS

The split-K2 compact W2 callee was sliced from its local function label to the
next function declaration.  It contains exactly:

```text
QGMMA: 32
WARPGROUP.DEPBAR: 32
```

Exact sliced SASS SHA256:
`63e5ce2faec911affabc2dd779271f5d38478db1728c677d5d97e1f6c56d085f`.

## Decision

**Reject before correctness and timing.**  Reducing the device-call ABI to a
shared-record pointer plus task id does not reproduce standalone W2's 2:1
QGMMA/wait schedule.  Therefore neither the high-argument ABI nor the outer
grid-stride loop alone explains serialization.  The next structural audit
must align the standalone kernel's descriptor/parameter form or move its
already-paired WGMMA issue body into a fused-callable specialization.

