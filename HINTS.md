# SM90 NVFP4 Unified Small-M AKO Constraints

- Keep `megamoe_nvfp4` and every existing experimental branch untouched. All
  implementation and benchmark changes belong to `megamoe_nvfp4_dev`.
- Flash, Pro, and MiMo must use one production small-M fused kernel body, one
  Mode2 Braided BN256/BK128 weight ABI, one decoder implementation, and one
  cross-rank synchronization protocol.
- Model shapes and routing topology are JIT shape parameters, not selector
  fingerprints. Do not select on model names, exact hidden sizes, exact expert
  counts, or isolated token-count equality points.
- Select tuning configurations from expected routed load, derived tile/wave
  counts, and the device SM count. Every protocol-relevant field must be the
  same on all ranks.
- Compile-time tuning may vary block M, stages, experts per wave, swap-AB,
  decoder schedule, and active dispatch warps. Do not add hot-loop runtime
  branches to share the kernel body.
- Use cold-L2 comparisons. For small M, collect 50 measurements and compare
  medians. Use fixed routing seeds and ABBA ordering for retained changes.
- A retained change must pass exact NVFP4 correctness for Flash, Pro, and MiMo
  with both absent and per-expert global scales, including skewed routing.
- Preserve large-M behavior until a separately measured migration is ready.
- Do not retain experimental environment-variable switches or disabled code.
- Do not push this branch unless the user explicitly requests it.

