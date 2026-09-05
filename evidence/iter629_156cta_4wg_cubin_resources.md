# Iteration 629: 156 CTA × 4 WG cubin resource gate

## Artifact

`/tmp/torch_ext_v4_tp/v4tp_63a318a5fa109f26b014_v178mspec/v4tp_63a318a5fa109f26b014_v178mspec.so` (10,914,952 bytes)

## `cuobjdump` result

For every emitted `tp4_megamoe_single_launch_kernel` specialization covering Tokens 8/16/32/64/128 and SplitK 2/4:

```text
REG:64 STACK:48 SHARED:2048 LOCAL:0 CONSTANT[0]:1425
```

## Residency accounting

- Threads per packed CTA: 512.
- Registers per CTA: 512 × 64 = 32,768.
- Two CTAs: 65,536 registers, exactly the H20 SM register-file budget; three cannot reside.
- Dynamic route-task scratch: 4 × 18,432 = 73,728 bytes/CTA.
- Static shared memory: 2,048 bytes/CTA.
- Total shared memory: 75,776 bytes/CTA, or 151,552 bytes for two CTAs, below the SM90 opt-in shared-memory capacity.
- Fixed local allocation: zero. The reported 48-byte stack frame is retained as a performance risk to observe, but does not invalidate this occupancy gate.

## Conclusion

The cubin is resource-feasible for exactly two CTAs/SM. This is only a static gate; the launch path's hard occupancy check plus bitwise output comparison must pass before timing is meaningful.
