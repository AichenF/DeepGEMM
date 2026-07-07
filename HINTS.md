# SM90 FP8 H200 Retuning Constraints

- Optimize only the existing SM90 FP8 MegaMoE implementation. NVFP4 is out of
  scope for this work.
- All new parameter decisions must be measured on NVIDIA H200. Historical H20
  tuning results are not valid evidence for choosing an H200 parameter.
- Do not import, restore, or implement the PR323 fused single-kernel path. Keep
  the current split L1/L2 kernel architecture.
- Tune the Flash shape `(H=4096, I=2048, E=256, top-k=6)` and the Pro shape
  `(H=7168, I=3072, E=384, top-k=6)`.
- Required M points are `8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  8192`.
- The current tuning target supersedes the earlier PR323 gate. Tune Pro M512
  and M1024 against the same-node DeepEP high-throughput grouped-FP8 baseline:
  M512 must be below 1060.8 us and M1024 below 1455.8 us (at least 10% faster
  than 1178.7 us and 1617.5 us respectively).
- Final H200 result: wide-FFN Pro M512 and M1024 use L1 N-major scheduling
  with the retained BN512/BK128 packed-BF16 parent. Confirmed cold-L2
  median-20 latencies are 985.632 us and 1367.848 us, respectively, so both
  exceed the 10%-faster target. Wide-FFN BN512 range rules require the
  independently validated packed-BF16 numerical policy; unsupported cases
  such as M768 must fall back to the generic BN256/FP32 schedule.
- Points with `M < 128` must not show a confirmed regression from the current
  `megamoe_sm90_opt` FP8 baseline.
- Use identical routes, capacity, rank topology, and timing boundaries for
  comparisons. Treat a nominal gap below 1% as unconfirmed until repeated.
- H200-specific selector choices must not alter H20, H100, or generic SM90
  behavior.
- Temporary experiment controls are allowed, but the final production selector
  must not depend on a new environment variable.
- Do not add phase-specific BF16 accumulation. Previously validated global
  BF16 accumulation for Pro candidates remains allowed, subject to the final
  exact-shape numerical gate.
- Refactor the H200 selector so schedule selection and numerical-format
  selection are separate.  Global packed-BF16x2 scaled accumulation should be
  default-on for every validated H200 Flash and Pro point at
  `M={8,16,32,64,128,256,512,1024,2048,4096,8192}`.  Add explicit FP32
  exceptions only after a repeated same-configuration BF16 comparison regresses
  by more than 0.5% or fails the unchanged numerical gate.
- Extend the BF16 path to cover the configurations needed by that matrix,
  including Pro M256 L1 BK256 and the low-M swap-AB path.  Do not silently
  fall back to FP32 when the H200 numerical policy selects BF16.
- The BF16 refactor itself must not broaden E5M2 combine coverage.  A later
  targeted H200 retune may add a point only after direct PR323 confirmation
  and the unchanged multi-seed numerical gate; Flash M1024 satisfies this
  separate requirement with N-major1 plus E5M2 combine.
- Do not create a commit unless the user explicitly asks for one.  Do not push
  any commit or branch.
- Refactor H200 schedule selection without model names or exact-M rows. Match
  paired ranges of `hidden` and `intermediate_hidden`, then select a continuous
  load bucket using `expected_tokens_per_expert = M * top_k / local_experts`.
  `top_k`, total experts, and local experts must not directly choose a shape
  schedule; local experts may only constrain experts-per-wave legality.
- Let L1 and L2 choose `block_n`, `block_k`, experts per wave, stages, and SM
  count independently. Keep common fields only where the split kernel truly
  requires them. Derive dispatch/non-epilogue/epilogue thread counts and shared
  memory from the selected tiles instead of storing coupled/dead selector
  fields.
- The range refactor must first reproduce the resolved configuration of all 22
  existing Flash/Pro points. Unmatched or illegal ranges fall back to the
  generic SM90 selector. Keep the current numerical-format policy independent
  and do not broaden BF16 or E5M2 coverage as part of the schedule refactor.
- Keep the H200 DeepEP + grouped-FP8 baseline as a standalone benchmark-only
  path in this workspace.  It must execute only DeepEP dispatch, grouped-FP8
  L1, Triton SwiGLU plus FP8 quantization, grouped-FP8 L2, and DeepEP combine;
  it must not call or warm up any PR323 or local fused MegaMoE kernel.
- Benchmark that baseline for Flash, Pro, and MiMo-V2.5-Pro at
  `M={8,16,32,64,128,256,512,1024,2048,4096,8192}` with an explicit L2 flush.
- Add a separate DeepEP low-latency benchmark-only path for Flash, Pro, and
  MiMo-V2.5-Pro at `M={8,16,32,64,128}`.  It must use
  `Buffer(low_latency_mode=True)`, low-latency dispatch/combine, the current
  workspace's masked grouped-FP8 GEMMs, and a masked Triton SwiGLU plus FP8
  quantization kernel.  It must not call, initialize, or warm up any PR323 or
  local fused MegaMoE kernel.  Use the same seed-101, eight-rank, three-by-20,
  max-rank, explicit-cold-L2 protocol as the high-throughput baseline, and
  report comparisons against both that baseline and current MegaMoE.
