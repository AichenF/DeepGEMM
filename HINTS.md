# SM90 NVFP4 Unified Small-M AKO Constraints

- Keep `megamoe_nvfp4` and every existing experimental branch untouched. All
  implementation and benchmark changes belong to `megamoe_nvfp4_dev`.
- Flash, Pro, and MiMo must use one production small-M fused kernel body, one
  Mode2 Braided BN256/BK128 weight ABI, one decoder implementation, and one
  cross-rank synchronization protocol.
- Model shapes and routing topology are JIT shape parameters, not selector
  fingerprints. Do not select on model names, exact hidden sizes, exact expert
  counts, or isolated token-count equality points.
- Select tuning configurations from expected routed load and derived
  tile/wave counts. The physical SM count controls only the local launch grid
  and grid synchronization; it is not a tuning fingerprint. Every
  protocol-relevant field must be the same on all ranks.
- Keep the selector layered like the SM90 FP8 implementation: heuristic input,
  normalized load, schedule tuning, config materialization, and legality. Do
  not mix exact request-M cases into any layer.
- Compile-time tuning may vary block M, stages, experts per wave, swap-AB,
  decoder schedule, and active dispatch warps. Do not add hot-loop runtime
  branches to share the kernel body.
- Use cold-L2 comparisons. For small M, collect 50 measurements and compare
  medians. Use fixed routing seeds and ABBA ordering for retained changes.
- A retained change must pass exact NVFP4 correctness for Flash, Pro, and MiMo
  with both absent and per-expert global scales, including skewed routing.
- Use one Mode2 Braided BK128 weight ABI for both BN128 and BN256. Migrating
  BN128 may invalidate standard-sign prepacked weights; callers must repack
  cached BN128 weights instead of selecting a legacy decoder at runtime.
- Keep the BN128 split communication, scheduler, and selector policy intact
  while moving its L1/L2 dequantization onto the shared Mode2 implementation.
- Reject a retained BN128 migration if its stable cold-L2 latency regresses by
  more than 3% against the current standard-sign split baseline.
- Do not retain experimental environment-variable switches or disabled code.
- Do not push this branch unless the user explicitly requests it.
