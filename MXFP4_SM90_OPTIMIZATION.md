# SM90 MXFP4 MegaMoE — optimization record

Target: the fused MXFP4 MegaMoE kernel on this branch
(`deep_gemm/include/deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused*.{cuh,inl}`),
measured against the two NVFP4 reference branches:

* `origin/megamoe_nvfp4_dev_m` — the fused H200 NVFP4 kernel this branch was
  ported from. Its files are **byte-identical** to the `nvfp4` arm carried on
  this branch, so it can be benchmarked *in the same process* as MXFP4, on the
  same weights and the same routing. That is the strongest comparison available
  and it is what every `nvfp4` column below is.
* `origin/megamoe_nvfp4_dev` — the "all-M" branch, an auto-selecting portfolio
  (`kernel_family="auto"`) whose small-M arms are register-source
  specializations of the same kernel. It has no SM90 FP8 MegaMoE and no MXFP4,
  so it can only be compared across processes; see the A/B/B/A protocol below.

Nothing here changes the numerics: weight format, scale semantics, activation
dtype, output dtype, shapes and tile boundaries are untouched. Every change is
either a different way of computing the same bits, a different schedule, or a
bug fix.

---

## 1. Measurement contract

| | |
|---|---|
| Shape | MiMo-V2.5-Pro MoE: hidden 6144, intermediate 2048, 384 experts, top-k 8, EP8 (48 experts/rank) — the only shape all three implementations accept |
| Hardware | 8x H20-3e (`viking-prod-583`) and 8x H200 (`viking-prod-321`), computelab |
| Harness | `tests/bench_mega_moe_formats_sm90.py` — all arms built from the **same** BF16 weights and the **same** routing inside one process, interleaved and repeated |
| Timing | `bench_kineto` CUDA kernel time, median over `--reps` repetitions of 20 timed calls, 8 GB L2 flush between calls |
| M | source tokens **per rank**, before top-k expansion |

Two properties of this harness matter for reading any number below:

* **Within one process, arms are directly comparable.** They see identical
  weights, identical routing and the same drift.
* **Across processes they are not.** Re-running the identical configuration
  moves the median by up to ~5 % on these nodes, and a fresh process also draws
  a different router (the `recv=` column). Every claim in this document is
  therefore either a within-process delta, or an A/B/B/A alternation.

---

## 2. Where the branch started

`origin/megamoe_mxfp4_dev_m` as shipped, versus its own in-process NVFP4 arm
(= `megamoe_nvfp4_dev_m`), 8x H200:

| M | mxfp4 (us) | nvfp4 dev_m (us) | mxfp4 vs dev_m |
|---:|---:|---:|---:|
| 16 | 515.3 | 405.9 | +26.9 % |
| 32 | 596.0 | 415.6 | +43.4 % |
| 64 | 953.4 | 466.7 | +104.3 % |
| 128 | 820.3 | 553.9 | +48.1 % |
| 256 | 633.2 | 510.9 | +23.9 % |

The MXFP4 port is a *minimal* delta from NVFP4 — the kernel bodies differ only
in the dequant call sites and the removal of NVFP4's shared LUT — so the whole
gap had to be in the dequant.

---

## 3. Root cause, at SASS level

`make_scaled_lut()` folded the E8M0 exponent into the magnitude table with
byte-wise saturating SIMD intrinsics (`__vaddus4`, `__vsubus4`, `__vminu4`,
`__vcmpgeu4`). **Hopper has no native byte-wise SIMD**, so ptxas emulates all of
them. Isolated probes (`nvcc -cubin` + `cuobjdump -sass`, `sm_90a`):

| probe | non-NOP SASS |
|---|---:|
| MXFP4 `make_scaled_lut` alone | **78** (39 LOP3 + 8 PRMT + 3 VIADD + …) |
| NVFP4 shared-LUT load **plus** full word decode | 34 |
| MXFP4 word decode with the table already in registers | 18 |

It ran once per 32-element scale group — four times per BK128 weight row per
thread. Over a whole row:

| full BK128 row decode | non-NOP SASS |
|---|---:|
| register-folded scale (as shipped) | **420** |
| shared-memory table (this branch now) | **179** |

So the format that should have been *cheaper* to dequantize than NVFP4 — one
scale per 32 elements instead of per 16 — was paying 2.4x more.

---

## 4. What the public / in-repo implementations do

| source | technique | verdict here |
|---|---|---|
| `megamoe_nvfp4_dev` (in repo) | **register-source (RS) WGMMA A operand** for the swapAB tiles: dequantize straight into the MMA's A registers, no decoded-B tile in shared memory | **adopted** — see §5.3 |
| `megamoe_nvfp4_dev` | `--ptxas-options=--register-usage-level=5` on its RS arms | **rejected**: no effect on either kernel here (168 regs / 0 spill / identical QGMMA+DEPBAR counts either way). The 168 comes from the kernel's own `warpgroup_reg_alloc<208>`, not from ptxas's heuristic |
| `megamoe_nvfp4_dev` | per-(target, model, M-range) bucket tables | adopted in spirit, not in form: the tier boundaries here are computed from the shape and the SM count (§5.4) rather than enumerated |
| Marlin-style nibble packing (already in this repo) | pre-permute nibbles so the PRMT selector is the packed word masked with `0x77777777` | already present; it is why the decode is 6 ops/word |
| `YijiaZhao/hopper_megamoe_basedondeepgemm` (public, same DeepGEMM lineage) | decode **unscaled** — every E2M1 magnitude is exactly representable in E4M3, so the table is two compile-time constants — and apply the E8M0 coefficient during accumulator promotion, which is exact because the WGMMA's K (32) equals MXFP4's group | not adopted: it needs one WGMMA group and one promotion per 32-K slice, which multiplies accumulator traffic; the table lookup here is already down to ~4 instructions per group. Recorded because it is the natural next step if the promotion cost ever becomes free |
| same | scaled-LUT-by-arithmetic over a restricted exponent window (`exponent * 0x08080808 + base`), with a x256 bias so no saturation logic is needed | not adopted, same reason — 2 IMAD vs 1 clamp + 1 LDS is not where the time is anymore |
| same | 2 K128 blocks per pipeline stage at the BM8 tier | not tried; noted as the remaining idea for the smallest tiles |
| LiquidGEMM (SC'25) | defer dequantization to the epilogue | not applicable: MXFP4's scale varies along K inside the reduction, so it cannot be hoisted past the K-sum |
| vLLM PR #53709 | on Hopper, **re-encode MXFP4 experts to block-FP8 at load time** (lossless: E2M1 x 2^k is exactly representable in E4M3 once the block scale absorbs the exponent) and run the existing block-FP8 MoE — no dequant in the kernel at all, at 2x the weight bytes | deliberately not taken: the `fp8` column in §9 is exactly that trade on this shape, and the FP4 path is 8-19 % ahead of it on H200. It is the right answer only if the kernel is dequant-bound *and* HBM is free, which is the regime this work removed |
| sglang PR #38884 | SM90 mixed-input CUTLASS grouped GEMM feeding MXFP4 nibbles to WGMMA directly, activations in BF16 | different operating point (W4A16, prefill); this kernel is W4A8 and already runs FP8 tensor cores |
| FlashInfer | NVFP4/MXFP4 (de)quantization kernels and scale-layout helpers | mostly Blackwell-targeted (native block-scaled FP4 MMA); nothing SM90-specific to port |

---

## 5. Changes made

### 5.1 32-entry shared-memory scaled-LUT window

`make_scaled_lut()` stays as the definition of the numerics, but the mainloop
reads its output from a table built once per CTA. Sweeping all 256 E8M0 codes
shows only **20 distinct tables**, at codes 118..137 (k = -9..+10): below that
every magnitude underflows E4M3's normal range and flushes to zero, above it
every magnitude saturates at 448. A **32-entry clamped window (256 B)**
therefore reproduces `make_scaled_lut` bit-exactly for every code — a quarter of
the 1 KB table the NVFP4 path loads, and half as many lookups per row, because
MXFP4 groups 32 elements where NVFP4 groups 16.

Verified bit-exact by a standalone gate: all 256 codes agree, and all
256 x 65536 packed-word patterns decode identically. The same check is now part
of `tests/test_mxfp4_mega_moe_sm90_correctness.py`, which passes 14/14 over
M = 1..256 x two global-scale modes plus 3/3 over M = 257/512/1024, every case at
cosine 0.9988-0.9990 — the same values the branch reported before this work.

### 5.2 Decoded-B ring sized correctly, pre-barrier region padded to the combine floor

The decoded-B tile was allocated `kNumStages` deep. Decode and consume are
serialized inside one K-block iteration (`warpgroup_wait<0>` before the loop
closes) and the two warpgroups touch disjoint N halves of the same BN256 tile,
so **one slot is enough** — the ring was holding `(kNumStages-1) x 32 KB` of
tiles nobody could read, and that is what capped the pipeline at 3-4 stages.

Freeing it exposed a second constraint: the combine phase reuses the whole
pre-GEMM region as its own ring and needs at least
`3 x warps x hidden x 2 / chunks` bytes of it. That floor is now explicit, so
the GEMM region is padded up to it instead of tripping a static assert.

### 5.3 Register-source (RS) WGMMA A operand

Ported from `megamoe_nvfp4_dev`. Lane pairs own one weight row of the half tile;
each K32 slice is two packed words plus one `__shfl_xor_sync` to assemble both
lanes' four A registers. Nothing lands in shared memory, so `kNumDecodedBStages`
drops to **0** and the pipeline-depth ceiling rises from 3-4 (original) to 7 (SS)
to **8-10** (RS).

MXFP4 needs **one** table lookup per K32 slice where NVFP4 needs two, because
its scale group is exactly the WGMMA's K.

Within the RS path, half 1's decode is issued under half 0's in-flight MMA. This
is deliberately *intra*-stage: prefetching the next K-block instead was measured
and is worse (§6).

### 5.4 Schedule selection

* Pipeline depth per tier comes from the sweep, and shared memory is always
  requested at full capacity. The old per-tier `smem_size` constants were the
  exact layout size of one fixed stage count; once depth became tunable they
  became a trap that faults with an illegal address rather than failing to
  compile. **This was a real bug** and is what the M=64 / M=128 crashes during
  tuning turned out to be.
* The swapAB tile is selected while a local expert's tokens still fit one tile —
  `num_tokens * num_ranks * num_topk / num_experts <= BLOCK_M` — scaled by SM
  count, because a transposed tile issues a WGMMA of N<=24 where the straight
  tile issues one of 128 and therefore needs ~5x the MMA instructions for the
  same work. Measured: BM24 wins by 6.5 % at M=128 and loses by 57 % at M=256 on
  H20's 78 SMs; on H200's 132 it already loses by 23 % at M=128 and wins by
  4.6 % at M=64. The formula reproduces both crossovers and moves with the shape
  instead of being pinned to a model name.
* RS is disabled for the BM8 tiers on parts wider than the reference SM count,
  where the tier is short enough that the shuffle latency is not hidden.
* An over-tight host assertion (`swap_ab == (num_tokens <= 64)`) was replaced by
  the constraint the kernel actually has (`swap_ab => BLOCK_M <= 24`), which is
  what had made the larger-batch swapAB experiments unreachable.

### 5.5 Contiguous weight tiles

The fused weight tensor was `(E, N, k_blocks * 80)`: N-major, so the
`BLOCK_N` rows a pipeline stage wants sat `k_blocks * 80` bytes apart and the
stage load was `BLOCK_N` strided 80-byte TMA requests. Section 7.2 measures what
that costs — 63 % of the device's streaming bandwidth.

`mxfp4_fused_to_tile_contiguous` regroups the same rows into
`(E, n_tiles * k_blocks, 128 * 80)`, one contiguous block per
`(expert, 128-row n-tile, k-block)`, and the loader issues `BLOCK_N / 128`
`cp.async.bulk` copies instead of a 2D TMA. Properties that made this cheap:

* It is a **pure byte permutation** done offline in the weight transform, so
  every quantized value, scale byte and sign-braid position is untouched. The
  correctness sweep returns the same cosine to four decimals.
* The tile row count is a constant 128 rather than `BLOCK_N`, so the offline
  layout stays valid for every `BLOCK_N` the selector can pick.
* A bulk copy has no descriptor, so the two weight `CUtensorMap`s and their
  prefetches go away; the kernel takes two base pointers instead.

Once the tile is bulk-copied, the 12 B of padding in each 80 B row has no
reason to exist either — it was there only because a TMA descriptor needs a
row that is a multiple of 16. Dropping it naively gives a 68-byte row, which
misaligns the 128-bit loads the Mode2 decoders use, and a 64-byte row plus a
separate scale plane is 16 words, which 4-way bank-conflicts them. Storing the
tile **16-byte-chunk-major** avoids both: chunk `c` of row `r` at
`c * 128 * 16 + r * 16`, then the E8M0 bytes at `128 * 64 + r * 4`. Every chunk
stays 16-byte aligned, the bank pattern is arithmetically identical to the
80-byte row (`4r mod 32` instead of `20r mod 32`, both 8 distinct banks over a
phase), and the row costs its natural 68 bytes. `decode_tile_row` is the one
place that knows the layout; all three decoders go through it.

`EVICT_NORMAL` is passed explicitly rather than taking `tma_load_1d`'s streaming
default, because above the swapAB tiers one expert spans several m-blocks and
those CTAs share these tiles through L2.

### 5.6 The transposed tile is a token tile, so pick it by tokens per expert

Under swapAB the tile's `BLOCK_M` *is* the WGMMA's token dimension, so a tile
wider than one local expert's token count pads the MMA with slots that carry
nothing. Routing puts `num_tokens * num_ranks * num_topk / num_experts` slots on
an expert, and the selector already had `max_tokens_for_block_m` to invert that.
Taking the narrowest tile that still covers it replaces three hand-set `BLOCK_M`
tiers with one rule, and reproduces every measured optimum:

| M | slots/expert | rule | measured (EP1, 48 experts) |
|---:|---:|---|---|
| 32 | 5.3 | BM8 | BM8 **330.5** vs BM16 337.3 vs BM24 343.2 |
| 64 | 10.7 | BM16 | BM16 **355.2** vs BM24 361.4 |
| 128 | 21.3 | past the swapAB bound | BM64 **419.0** vs BM24 443.0 vs BM16 640.8 |

One step too narrow is much worse than one too wide, because the expert no
longer fits one m-block and its weights are re-read: BM16 at M=128 costs 45 %,
BM8 at M=64 costs 47 %. The rule never does that -- it only ever rounds up.

With the rule in the selector, the default arm lands on the measured optimum
(same process, EP1/48 experts):

| part | M | default | BM8 | BM16 | BM24 | default before |
|---|---:|---:|---:|---:|---:|---:|
| H200, 132 SM | 32 | **330.8** | 330.9 | 335.8 | 343.2 | 337.6 |
| H200, 132 SM | 64 | **355.3** | 522.9 | 355.7 | 361.4 | 361.1 |
| H20-3e, 78 SM | 32 | **503.3** | 503.3 | 536.2 | 542.9 | — |
| H20-3e, 78 SM | 64 | 581.9 | 813.1 | 581.4 | **572.5** | — |

One exception, and it is left in deliberately. At M=64 on 78 SMs, BM24 is 1.6 %
faster than the BM16 the rule picks, where on 132 SMs BM16 is 1.7 % faster than
BM24. That is the same SM-count dependence the swapAB bound above already
encodes — a wider tile amortises per-tile overhead where the part is issue-bound,
a narrower one wastes less WGMMA width where it is load-bound. Scaling the rule
for it would mean fitting a factor to one 1.6 % data point, so the rule stays as
measured: it wins 6.1 % at M=32 on 78 SMs and 2.0 % on 132, and gives up 1.6 % at
one point on one part.

The RS gate needed rescoping with it. It had disabled the register-source
operand for `BLOCK_M < 16` on parts wider than the reference SM count, which the
new rule would have applied at M=32; measured there, the same BM8 tile is 4.3 %
*faster* with RS (330.5) than without (345.5). The gate is a statement about how
little work a tier has, not about `BLOCK_M`, so it is now scoped to the batches
it was measured on.

### 5.7 Test and harness fixes found on the way

* `tests/compile_all_megamoe_kernels.sh` could not instantiate the MXFP4 kernel
  at all — its argument list predated the `kNumSMs` and shape template
  parameters, so the gate had been silently reporting a compile failure as the
  status quo. Fixed, and extended to cover the four tiers that actually ship.
* `tests/bench_mega_moe_formats_sm90.py` grew in-process schedule-variant arms
  (`--mxfp4-scheds BM,STAGES,EPW[,SINGLE_DISPATCH[,RS]]`). Comparing schedules
  across processes is meaningless here — a fresh process draws a different
  router — and this is what made the tuning numbers trustworthy.
* `srun` inherits stdin and swallowed the rest of any script fed to it over
  `bash -s`, which silently truncated batch runs after the first configuration.

---

## 6. Numerics

The correctness gate's reference uses the **same** FP8 input activations the
kernel gets (`x_ref` is `x_fp8 * x_sf` in fp32) and the **exact** MXFP4
dequantized weights, in fp32 throughout. So the residual it reports is purely
the kernel's own arithmetic.

Measured on the real weight distribution (randn x 0.05, 4 x 2048 x 6144):

| path | exactly representable in E4M3 | mean rel err |
|---|---:|---:|
| **MXFP4 weight -> E4M3** | **100.0000 %** | **0** |
| NVFP4 weight -> E4M3 | 74.23 % | 1.59e-2 |
| W4A8 intermediate, per-128 E4M3 round trip | — | 2.24e-2 |

E2M1's eight magnitudes are all exactly representable in E4M3 and E8M0 is a pure
exponent shift, so **MXFP4's dequant-to-FP8 path is lossless for the weights** —
the FP8 WGMMA sees exactly the MXFP4 values, saturation/flush at the range ends
being the only exception and one the shipped `make_scaled_lut` already defines.
NVFP4 on the same path must round, because UE4M3 carries a mantissa.

The 0.9989 / 0.9990 cosine the gate reports is therefore the **W4A8 intermediate
requantization** (L1 output is stored as FP8 e4m3 with per-128 scales before L2),
which is a property of the format contract and identical for the fp8, nvfp4 and
mxfp4 arms. It is unchanged before and after this work, at every M.

The LUT window itself is bit-exact against the function it replaces: all 256
E8M0 codes and all 256 x 65536 packed-word patterns agree.

---

## 7. What the kernel is bound by

Per rank per call: `FLOPs = 2 * routes * 3*H*I`, weight bytes =
`touched_experts * 3*H*I * 80/128`, plus `routes * (3H + 2I)` of activations and
`routes * (H/32 + I/8)` of activation scales. Peaks: H200 1979 TF/s dense FP8 and
4.8 TB/s; H20-3e 296 TF/s and 4.8 TB/s.

| dev | M | AI (F/B) | ridge | bound | achieved TF/s | roofline | % of roofline | % of FP8 peak | % of HBM peak |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|---:|
| H200 | 8 | 5.3 | 412 | mem | 12.5 | 25.3 | 49.5 % | 0.6 % | 49.5 % |
| H200 | 32 | 18.6 | 412 | mem | 50.0 | 89.4 | **55.9 %** | 2.5 % | 55.9 % |
| H200 | 64 | 32.7 | 412 | mem | 82.9 | 156.8 | 52.8 % | 4.2 % | 52.8 % |
| H200 | 128 | 67.3 | 412 | mem | 163.9 | 322.9 | 50.8 % | 8.3 % | 50.8 % |
| H200 | 256 | 134.0 | 412 | mem | 309.6 | 643.4 | 48.1 % | 15.6 % | 48.1 % |
| H200 | 2048 | 829 | 412 | cmp | 522.9 | 1979 | 26.4 % | 26.4 % | 13.1 % |
| H20-3e | 32 | 17.1 | 61.7 | mem | 32.3 | 82.1 | 39.4 % | 10.9 % | 39.4 % |
| H20-3e | 64 | 36.5 | 61.7 | mem | 68.1 | 175.3 | 38.8 % | 23.0 % | 38.8 % |
| H20-3e | 128 | 67.5 | 61.7 | **cmp** | 95.7 | 296.0 | 32.3 % | 32.3 % | 29.5 % |
| H20-3e | 256 | 132.5 | 61.7 | **cmp** | 161.1 | 296.0 | **54.4 %** | 54.4 % | 25.3 % |
| H20-3e | 2048 | 832 | 61.7 | cmp | 187.1 | 296.0 | 63.2 % | 63.2 % | 7.6 % |

Three regimes, and they are qualitatively different:

1. **Decode (M <= 256) is weight-stream and fixed-overhead bound, not compute
   bound.** AI is 3-134 F/B against H200's ridge of 412, so the tensor cores see
   0.2-15.6 % of peak. A linear fit gives **H200: 401 us + 52 ns/token** and
   **H20-3e: 550 us + 203 ns/token** — even at M=256, 78 % (H200) and 57 % (H20)
   of the time does not depend on how many tokens there are. The 48-expert weight
   stream alone is **236 us at HBM peak**, and the kernel runs that stream at
   **48-56 % of peak** on H200. The remaining ~165 us of the H200 intercept is
   dispatch/combine/NVLink barriers, launch, and dequant that is not hidden.
2. **H20-3e crosses to compute-bound near M ~ 120** (its ridge is only 61.7 F/B
   because its FP8 rate is 6.7x lower at the same bandwidth) and reaches **54 %
   of dense FP8 peak at M=256**.
3. **Prefill (M >= 1024) is compute-bound on both, and there FP4 is the wrong
   trade.** Hopper has no FP4 MMA: the format buys HBM bytes, which stop
   mattering once compute-bound, and pays ALU for the dequant. Measured at
   M=2048: MXFP4 reaches 26.4 % of FP8 peak on H200 where the FP8 MegaMoE reaches
   33.7 %, and 63.2 % vs 77.6 % on H20-3e — a 20-25 % deficit that is the dequant
   tax, not a scheduling problem.

### 7.1 Splitting the runtime three ways

`kNumRanks` became a template parameter, so the same kernel runs at EP1 — one
rank, no all-to-all, no cross-rank barriers. Running EP1 with 48 experts against
EP8 with 384 gives each GPU the same weight volume and the same routed-token
count, so the difference is exactly what the collective costs. H200, MXFP4:

| M | EP1 / 48 experts | EP8 / 384 experts | collective |
|---:|---:|---:|---:|
| 8 | 287.8 us (34 exp) | 368.3 us (37 exp) | +80.5 |
| 32 | 385.1 us (47 exp) | 415.6 us (47 exp) | +30.5 |
| 64 | 407.1 us (47 exp) | 450.9 us (48 exp) | +43.8 |
| 128 | 442.5 us (47 exp) | 474.4 us (48 exp) | +31.9 |
| 256 | 455.2 us (47 exp) | 511.1 us (48 exp) | +55.9 |

A linear fit over EP1's M=32..256, where the touched-expert count is already
saturated, gives **375 us + 0.313 us/token**. The 375 us intercept is the weight
stream plus everything that does not scale with tokens. With the weight stream
measured independently at 3.52 TB/s (the expert-count ablation), H200 at M=32
decomposes as:

| part | time | share |
|---|---:|---:|
| weight stream, 1.109 GB at 3.52 TB/s | 316 us | 76 % |
| local fixed cost (launch, grid sync, epilogues, combine) | ~59 us | 14 % |
| cross-rank dispatch + combine + NVLink barriers | ~40 us | 10 % |
| **total** | **415.6 us** | |

The collective is format-independent: it costs FP8, NVFP4 and MXFP4 the same, so
no weight-format work can remove it. That puts a hard ceiling on any
roofline percentage quoted against weight bytes alone.

### 7.2 The load path, measured in isolation

The roofline above assumes the weight bytes move at HBM peak. They did not, and
the reason was not the byte count. Two standalone measurements on the same
H200:

*What the device can actually stream* (torch, 512 MiB buffers): 4.217 TB/s for a
copy, **4.154 TB/s read-only**, against a 4.8 TB/s theoretical peak — so ~87 %
is the practical ceiling, not 100 %.

*What each weight-tile addressing scheme sustains* — a standalone 132-CTA
pipeline that issues the loads and does nothing else
(`tests/bench_sm90_weight_load.cu`, nvcc-only), 1.3-1.6
GB working set, 6 stages:

| addressing | TB/s | bytes/iter |
|---|---:|---:|
| 2D TMA, 80 B rows, `k_blocks * 80` stride (what the kernel did) | **2.769** | 1557 MB |
| 2D TMA, 64 B rows + separate scale row | 3.406 | 1324 MB |
| 1D bulk copy, contiguous 17408 B tile | 4.413 | 1324 MB |
| **1D bulk copy, contiguous 20480 B tile (same bytes as today)** | **4.487** | 1557 MB |
| 1D bulk copy, contiguous 16384 B tile | 4.269 | 1246 MB |

A pipeline stage wanted `BLOCK_N` rows of one k-block, and those rows sat
`k_blocks * 80` bytes apart, so the stage was 256 strided 80-byte requests. That
pattern tops out at **63 % of what the same device streams contiguously**.
Grouping the tile so its rows are adjacent moves the *identical* bytes 1.62x
faster, and needs no change to the shared-memory image the decoder reads.

This also settles the reverted experiment in section 8: its layout was right and
its load instruction was right — the 25-40 % it lost came from elsewhere, not
from the bulk copy.

### 7.3 Where the kernel stands after contiguous tiles

Same H200, same routing, `fp8` as the in-process control (its code is
untouched, and across the two EP1 runs it reproduces to 0.2 %):

| | EP1 M=32 | EP8 M=32 |
|---|---:|---:|
| before | 385.1 us | 415.6 us |
| after | **345.5 us** | **390.3 us** |
| weight bytes (47 experts) | 1.109 GB | 1.109 GB |
| at 4.8 TB/s theoretical | 231 us -> **66.9 %** | 231 us -> **59.2 %** |
| at 4.154 TB/s achievable | 267 us -> **77.2 %** | 267 us -> **68.4 %** |

Backing the weight stream out of the EP1 intercept puts it at **4.01 TB/s, 97 %
of what this device streams**. The load path is closed; what is left above the
roofline is not the GEMM:

Measured directly rather than inferred, by varying the expert count at fixed
M=32 on EP1 — the tile count, and so the weight bytes, scale with experts while
the token-dependent work does not:

| experts touched | time |
|---:|---:|
| 47 | 337.4 us |
| 35 | 265.6 us |
| 23 | 194.9 us |

Both gaps give the same slope to 1.5 % (5.98 / 5.89 us per expert) and the same
intercept from either end, so:

| part | M=32 EP1 | |
|---|---:|---|
| weight stream, 20.06 MB/expert at **3.38 TB/s** | 279 us | |
| fixed cost, independent of the weights | **58 us** | |

The load path on its own sustains **4.4 TB/s** (section 7.2, variant G, and the
same on H20-3e). The kernel only gets 3.38, so **the consumer — decode plus
WGMMA — is throttling the loader by 24 %**; the memory system is not the limit.

Running the real decoder standalone over a resident tile puts the decode at
**490 ns per stage per SM** against the load's 522, which looks like there is no
slack at all. There is: making the decode 10 % cheaper moves the kernel 0.7 %
(section 8), so the decode is in fact mostly hidden. The ~161 ns per stage that
is not hidden is the WGMMA and the synchronisation around it -- consistent with
the narrower transposed tile (section 5.6) being what actually helped.
That is the opposite of what the earlier "97 % of achievable" reading suggested,
which measured the stream in isolation rather than in the kernel.

`ncu` is not installed on any node image these run on, so the phase split came
from `clock64()` timestamps compiled into the kernel behind a macro and read
back with `printf` — diagnostic only, not kept. At M=32 on 78 SMs, EP1, two
independent runs:

| phase | cycles | share |
|---|---:|---:|
| entry -> GEMM end (prologue, dispatch, mainloop) | 990 833 / 992 747 | **91.8 / 91.9 %** |
| combine + workspace cleanup | 88 889 / 87 202 | **8.2 / 8.1 %** |

1 079 722 cycles at ~1.98 GHz is ~545 us against ~503 us of measured wall time,
so the timestamps track the real kernel. **The combine tail is worth at most
8 %**, and it moves ~3 MB, which at 4 TB/s is under a microsecond — so what it
costs is the device-wide barrier in front of it and the latency of its own
chunked loop, not bandwidth. Removing it outright would not reach the target,
which is why the remaining work is not a single change.

The throttle has a model, and it checks out. Per stage per SM at M=32: the load
wants 522 ns (17408 B at the 4.4 TB/s the load path sustains), and the consumer
takes 683 ns (the expert-ablation slope, 5.94 us per expert over 8.7 stages).
If the consumer gates the loader, the stream should run at

    4.4 TB/s * 522 / 683 = 3.36 TB/s

against **3.38 TB/s measured** — 0.6 % apart. So the arithmetic that closes the
target is explicit: **cut 161 ns, 24 %, off the consumer** and the stream reaches
4.4 TB/s, the weight time 279 -> 214 us, and the kernel ~272 us, which is 72 % of
the 196 us roofline.

That 161 ns is now pinned down. A harness that runs the real consumer — the RS
decode feeding the swapAB WGMMA, both halves, as `decode_rs_half` and
`issue_half` do it — reproduces 544 ns of it (the rest is the epilogue, the
barriers and the A/SF loads it omits). Within that:

| consumer variant | ns/stage/SM |
|---|---:|
| as shipped | 545.1 |
| **the same with a free scale lookup** | **484.5** |
| one lookup per row instead of four | 488.6 |
| decode alone, no WGMMA (earlier harness) | 490 |

Three things follow. The **WGMMA adds only ~54 ns** on top of the decode, so the
transposed tile's narrow N is not what costs. The **scale lookup is 60 ns**, and
collapsing four per row into one recovers 56 — but that is only correct when all
four groups in a K-block share an E8M0 exponent, which they generally do not,
and a warp-uniform guard would fire too rarely to pay. And with a **free** lookup
the consumer still costs 484 ns against a 522 ns load.

That last number is the answer to the whole question. Feeding 4.4 TB/s needs the
dequant to produce 32768 values per 522 ns, about 32 per cycle per SM; it
produces about 34. **MXFP4 on Hopper sits right at the edge of being
dequant-bound** — the format has no FP4 MMA here, so every weight byte is
unpacked in the ALU before it can be multiplied, and that unpacking is as
expensive as fetching it. Even a free lookup leaves the kernel at 64 % of
roofline. Reaching 70-80 % is not a scheduling or layout problem; it needs
either fewer dequantised values per byte of weight, or hardware that multiplies
FP4 directly.

It also resolves why the decode reorder disappointed: 10 % off the decode
*alone* is nothing once the WGMMA is there to overlap with, which is exactly
what both the harness (543.8 vs 544.0) and the kernel (0.7 %) show.

Two levers, both quantified:

| if | M=32 EP1 | vs 196 us theoretical roofline |
|---|---:|---:|
| today | 337 us | 58 % |
| stream reaches the load path's 4.4 TB/s | 272 us | **72 %** |
| ...and the fixed cost halves | 243 us | **81 %** |

Dropping the 12 B of per-row padding on top (chunk-major tiles, section 5.5)
buys much less than its 15 % of bytes suggests — **-3.6 / -1.2 / -2.2 / +0.1 /
+0.1 %** at EP1 M=8..256. Two reasons, both visible in section 7.2's table: at
M>=128 the tier is no longer load-bound, and a smaller bulk copy sustains less
bandwidth (variant D's 16 KB tile runs at 4.27 TB/s where variant E's 20 KB tile
runs at 4.49). It is kept because it is never a regression and it takes 15 % off
the weight footprint itself, which is the point of a 4-bit format.

What is left above the roofline is not the GEMM and not the weight layout.

---

## 8. Measured and rejected

Recorded so they are not retried:

| idea | result |
|---|---|
| Contiguous weight tiles via one 1D bulk copy **with the scales split out** (64 B rows) | **-25 to -40 %** at H20 M=32-64, while moving 15 % fewer bytes. Section 7.2 later showed the load itself was 1.6x faster, so the loss was on the consumer side: a 64 B shared-memory row stride is 16 words, which 4-way bank-conflicts the decoder's 128-bit loads where 80 B (20 words) is conflict-free. Contiguity alone, keeping the 80 B row, is what shipped |
| Larger weight bulk copies (one 17408 B copy per stage instead of two 8704 B ones, via k-major tile order) | **No gain, and the reorder would have been wasted work.** The standalone pipeline sustains 4.53 TB/s on today's two-copy pattern against 4.34 for one contiguous copy. Measured before implementing |
| Deeper pipelines now that a stage is 17408 B, not 20480 B (10 stages fit where 6-8 did) | **No effect.** Standalone, 4/6/8/10 stages give 4.36/4.48/4.37/4.40 TB/s -- the load path saturates at 4. End-to-end, M=32 gives 336.6/335.5/339.1 us at 6/8/10 and M=64 gives 362.0/362.9 at 8/10. The consumer is not waiting on buffering |
| Decode both packed words before either lane-pair shuffle, so the second is not stuck behind the first shuffle's latency | **10 % off the decode in isolation** (472 -> 432 ns per stage per SM, standalone harness) and **0.7 % end-to-end** at M=32, nil at M=64. The decode's issue cost is largely hidden behind the load and the WGMMA already, so shrinking it buys almost nothing. Reverted -- but the measurement is the useful part: of the ~161 ns per stage that is *not* hidden, the decode is not what it is made of |
| Second dispatch warp (`single_active_dispatch_warp=0`), and experts per wave 48 vs 24 vs 16 | **Nothing**, to 0.12 %: 503.4 / 503.5 / 503.2 / 503.8 us at M=32 on 78 SMs, one process. These were the last two knobs sitting in the ~58 us weight-independent intercept, so that intercept is not reachable by configuration either |
| Prefetch: decode K+1 under K's in-flight WGMMAs (two decoded slots, `warpgroup_wait<1>`) | **+5.9 / +7.1 / +10.8 %** at MiMo/H20 M=64 over three runs. Its prefetch waits on stage K+1's TMA barrier, and that wait sits between the WGMMA issue and `arrive_empty(K)`, so it delays the loader by exactly what it saves on the decode |
| `--ptxas-options=--register-usage-level=5` | no effect: 168 regs, 0 spill, identical QGMMA/DEPBAR, on both the SS and RS kernels |
| Pipeline depth alone (SS path) | ~1 %, and it flipped sign between runs — inside the drift band, so the shipped depths come from the RS sweep where the effect is outside it |
| BM24 instead of BM16 at M=32, BM16/EPW24/stage variants at M=64 | every alternative was 2-6 % worse than the shipped tier; the selector is at a local optimum for the knobs it exposes |

---

## 9. Results

### 9.1 vs `megamoe_nvfp4_dev_m`, 8x H200, one process, identical routing

`nvfp4` here *is* `megamoe_nvfp4_dev_m` — the files are byte-identical — so this
is a same-process, same-weights, same-router comparison with no cross-run drift
in it at all.

| M | fp8 (us) | mxfp4 (us) | nvfp4 dev_m (us) | mxfp4 vs dev_m | mxfp4 vs fp8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 177.1 | 172.5 | 185.0 | **-6.8 %** | -2.6 % |
| 8 | 436.7 | 368.3 | 368.6 | -0.1 % | -15.7 % |
| 16 | 490.5 | 417.1 | 416.8 | +0.1 % | -15.0 % |
| 32 | 515.5 | 415.6 | 422.7 | **-1.7 %** | -19.4 % |
| 64 | 542.6 | 450.9 | 465.3 | **-3.1 %** | -16.9 % |
| 128 | 526.3 | 474.4 | 480.2 | **-1.2 %** | -9.9 % |
| 256 | 553.1 | 511.1 | 500.2 | +2.2 % | -7.6 % |

Starting point for the same comparison was +23.9 % to +104.3 % (§2).

### 9.2 vs `megamoe_nvfp4_dev` (`kernel_family="auto"`), 8x H20-3e, A/B/B/A

The two implementations live in different worktrees and cannot share a process,
so whole processes are alternated A/B/B/A and each arm's two passes are averaged.
The node drifts a lot at small M — dev's own two passes differ by 21 % at M=1 and
10 % at M=8 — so only the larger-M rows are tight.

| M | mxfp4 pass 1 / 2 (us) | dev pass 1 / 2 (us) | mean mxfp4 | mean dev | delta |
|---:|---|---|---:|---:|---:|
| 1 | 161.6 / 163.2 | 171.8 / 141.5 | 162.4 | 156.7 | +3.7 % |
| 8 | 556.4 / 444.6 | 476.1 / 431.6 | 500.5 | 453.9 | +10.3 % |
| 16 | 522.1 / 507.7 | 522.3 / 521.6 | 514.9 | 522.0 | **-1.4 %** |
| 32 | 581.7 / 620.3 | 568.8 / 575.8 | 601.0 | 572.3 | +5.0 % |
| 64 | 637.5 / 640.8 | 611.7 / 596.5 | 639.2 | 604.1 | +5.8 % |
| 128 | 804.1 / 811.5 | 805.7 / 780.9 | 807.8 | 793.3 | +1.8 % |
| 256 | 981.8 / 970.2 | 980.3 / 976.6 | 976.0 | 978.5 | **-0.3 %** |

Before this work the same comparison was +10 % to +23 % across the board. What
remains is concentrated at M=32-64, where `megamoe_nvfp4_dev` additionally
rotates its RS schedule across K-blocks; the intra-stage half of that rotation is
adopted here (§5.3), the cross-stage half was measured and rejected (§6).

### 9.3 Before / after on the same hardware

8x H20-3e, MXFP4 arm only, against the FP8 MegaMoE measured in the same process
(the FP8 column is the drift anchor — it is the same kernel in both rows):

| M | branch as shipped (us) | optimized (us) | speedup | fp8 anchor, shipped run / optimized run |
|---:|---:|---:|---:|---|
| 1 | 205.5 | 139.8 | **1.47x** | 144.9 / 157.7 |
| 8 | 693.4 | 461.1 | **1.50x** | 475.8 / 492.4 |
| 16 | 817.4 | 502.8 | **1.63x** | 543.6 / 556.5 |
| 32 | 969.2 | 602.3 | **1.61x** | 596.7 / 601.2 |
| 64 | 1022.0 | 614.5 | **1.66x** | 677.3 / 653.6 |
| 128 | 1047.0 | 815.3 | **1.28x** | 960.7 / 978.4 |
| 256 | 1060.0 | 970.5 | **1.09x** | 986.3 / 987.7 |

The FP8 MegaMoE is the same kernel in both runs and moves by only 1-4 % between
them, which is what bounds the drift these speedups are measured against.


### 9.4 Contiguous weight tiles, H200

`fp8` runs in the same process and its code is untouched, so the ratio to it
cancels the cross-process drift. `nvfp4` still uses the strided descriptor path
and is the in-process control for the layout change itself.

EP8 / 384 experts, 8x H200:

| M | mxfp4/fp8 before | mxfp4/fp8 after | gain | mxfp4/nvfp4 before | mxfp4/nvfp4 after |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.843 | 0.726 | **-13.9 %** | 0.999 | **0.916** |
| 16 | 0.850 | 0.717 | **-15.6 %** | 1.001 | **0.860** |
| 32 | 0.806 | 0.725 | **-10.1 %** | 0.983 | **0.897** |
| 64 | 0.831 | 0.741 | **-10.8 %** | 0.969 | **0.881** |
| 128 | 0.901 | 0.820 | **-9.0 %** | 0.988 | 0.974 |
| 256 | 0.924 | 0.871 | **-5.7 %** | 1.022 | **0.945** |

EP1 / 48 experts, 1x H200 — no collective, and the fp8 anchor reproduces to
0.2 % between the two runs, so these are absolute:

| M | before (us) | after (us) | gain | fp8 anchor before / after |
|---:|---:|---:|---:|---|
| 8 | 287.8 | **253.5** | -11.9 % | 356.5 / 357.1 |
| 32 | 385.1 | **345.5** | -10.3 % | 484.6 / 484.7 |
| 64 | 407.1 | **374.8** | -7.9 % | 505.4 / 506.2 |
| 128 | 442.5 | **422.3** | -4.6 % | 496.0 / 497.0 |
| 256 | 455.2 | **433.7** | -4.7 % | 509.5 / 509.4 |

8x H20-3e is **flat** (-0.2 to -2.5 %, inside that node's drift), and that is the
expected result: H20 runs the same weight stream at 1.94 TB/s, 40 % of peak, so
it is bound by its 78 SMs and not by the load. Making the load faster cannot
help a kernel that is not waiting on it.

Correctness after the change: 18/18 over M=1..1024 in both global-scale modes,
cosine 0.9986-0.9990 — unchanged, as a pure byte permutation must be.

### 9.5 Unpadded chunk-major tiles, H200

Measured on top of section 9.4, same method. EP1 / 48 experts, where the fp8
anchor reproduces to 0.3 % across all three trees:

| M | baseline | contiguous | + unpadded | vs contiguous | fp8 anchor |
|---:|---:|---:|---:|---:|---|
| 8 | 287.8 | 253.5 | **244.5** | -3.6 % | 356.5 / 357.1 / 356.5 |
| 32 | 385.1 | 345.5 | **341.3** | -1.2 % | 484.6 / 484.7 / 484.8 |
| 64 | 407.1 | 374.8 | **366.6** | -2.2 % | 505.4 / 506.2 / 505.8 |
| 128 | 442.5 | 422.3 | 422.8 | +0.1 % | 496.0 / 497.0 / 496.3 |
| 256 | 455.2 | 433.7 | 434.2 | +0.1 % | 509.5 / 509.4 / 508.9 |

EP8 / 384 experts, mxfp4/nvfp4 in one process: 0.859 / 0.848 / 0.911 / 0.855 /
0.968 / 0.928 at M=8..256, against 0.916 / 0.860 / 0.897 / 0.881 / 0.974 / 0.945
for contiguous-only — better at five of six points.

Correctness: 18/18 over M=1..1024 in both global-scale modes, cosine
0.9986-0.9990.

### 9.6 EP4 and EP8 on the shipped stack

The first run of everything together — rank templating, contiguous tiles, the
unpadded chunk-major layout and the tile rule — at the rank counts that ship.
8x H20-3e, `fp8` in the same process as the anchor:

| M | EP4 / 192 experts | | EP8 / 384 experts | |
|---:|---:|---:|---:|---:|
| | fp8 | mxfp4 | fp8 | mxfp4 |
| 8 | 498.4 | **456.0** (-8.5 %) | 455.1 | **444.5** (-2.3 %) |
| 16 | 524.5 | **482.9** (-7.9 %) | 583.9 | **517.0** (-11.5 %) |
| 32 | 549.3 | **530.3** (-3.5 %) | 581.7 | 604.0 (+3.8 %) |
| 64 | 636.9 | 643.4 (+1.0 %) | 668.5 | **658.3** (-1.5 %) |
| 128 | 933.3 | **812.1** (-13.0 %) | 957.2 | **833.4** (-12.9 %) |
| 256 | 951.5 | **931.4** (-2.1 %) | 1003.0 | **980.5** (-2.3 %) |

EP4 correctness: 10/10 over M=8..256 at 192 experts, which is 48 per rank — the
same per-rank load the EP1 measurements were taken at.

The +3.8 % at EP8 M=32 looked like a regression from the tile rule and is not:
the in-process comparison in section 5.6 has the rule's BM8 **6.1 % ahead** of
BM16 at that exact point on the same part. Between those two EP8 runs the fp8
anchor itself moved 13 % and the router drew a different token count (253 vs
239 received), which is the drift these cross-process numbers carry and why the
selector conclusions all come from in-process arms.

### 9.7 Large M — the tier this work does not touch

Above `swap_ab_max_tokens` the selector falls to BM64, and above M=256 to the
BM128/BN128 split-M tier. Both are still **one fused megakernel**; this branch
has no split L1/L2 kernel family at all, which is what `megamoe_nvfp4_dev`
switches to for M >= 257.

8x H200, one process:

| M | fp8 (us) | mxfp4 (us) | nvfp4 dev_m (us) | mxfp4 vs dev_m | mxfp4 vs fp8 |
|---:|---:|---:|---:|---:|---:|
| 512 | 888.9 | 813.4 | 808.6 | +0.6 % | **-8.5 %** |
| 1024 | 1197.4 | 1534.0 | 1545.0 | **-0.7 %** | +28.1 % |
| 2048 | 1872.7 | 2387.0 | 2391.0 | **-0.2 %** | +27.5 % |

8x H20-3e (nvfp4 cannot run, 132 SMs hardcoded):

| M | fp8 (us) | mxfp4 (us) | mxfp4 vs fp8 |
|---:|---:|---:|---:|
| 512 | 1874.1 | 2209.0 | +17.9 % |
| 1024 | 2858.0 | 4337.0 | +51.8 % |
| 2048 | 5406.0 | 6640.0 | +22.8 % |

MXFP4 tracks `dev_m` to within 0.7 % here, as expected — none of §5 applies to
these tiers. The real statement is the FP8 column: **past M ~ 512 the fused FP4
path is the wrong kernel**, for the reason in §7 (no FP4 MMA on Hopper, so the
dequant is pure overhead once compute-bound). Closing that needs either a split
family or a large-M tile, not more dequant work.

A regression was found here and fixed: threading an explicit decoded-slot index
through `decode_b_stage` (§5.2) had pinned the BM128 split-M tier to slot 0,
losing the alternation its paired warpgroups rely on. It failed at M=512 with
cosine 0.8966 and passes at 0.9989-0.9990 after the fix — the same value every
other tier reports. **This tier had never been exercised by the small-M sweeps.**

### 9.8 Attribution, in-process where possible

| change | measurement | effect |
|---|---|---|
| shared-memory LUT window (§5.1) | H20, full sweep, fp8-anchored, two runs | **-27 to -33 %** at M=8..64 |
| decoded-B ring + combine floor (§5.2) | H20, in-process stage sweep | perf-neutral at fixed depth; removes 64 KB of dead shared memory and lifts the depth ceiling |
| RS operand (§5.3) | H20, in-process arms, identical routing | -6.7 % (M=8), -4.2 % (M=32), -2.1 % (M=64), -6.5 % (M=128, where it also beats the BM64 tier) |
| intra-stage half overlap (§5.3) | H20, fp8-anchored | M=64 went from +4.2 % to -6.0 % against fp8 |
| cross-K-block decode prefetch | H20, in-process arms | **rejected**: -2.4 % at M=64 but +5.5..6.7 % at M=32 and +13.3 % at M=8. Same mechanism as the SS rotation in §6 — the prefetch's stage-barrier wait lands in front of `arrive_empty`, and only at BM24 is the decode long enough to pay for it. A single-tier 2 % at the edge of the drift band is not worth a fourth schedule flag |
| tile/stage selection (§5.4) | H200, in-process | fixed a +22.9 % regression at M=128 that the shape-only rule had introduced |

---

---

## 10. Reproducing

The weight-load measurement in section 7.2 needs no allocation at all — it is
nvcc-only and runs on any sm_90a device, including the raplab dev boxes:

```bash
nvcc -O3 -arch=sm_90a -o /tmp/wl tests/bench_sm90_weight_load.cu \
     -lcuda -L/usr/local/cuda/lib64/stubs && /tmp/wl
# -DKSTAGES=N varies pipeline depth; nothing changes above four.
```

Note the raplab boxes cannot run the kernel itself: the only local torch is
2.5, which has no `symm_mem.rendezvous`, so the symmetric buffer the MegaMoE
needs cannot be built there. Everything else runs from a computelab allocation. The scripts and every raw log
live outside the repo, on shared scratch:
`/home/scratch.jinyanc_wwfo/github/DeepGEMM-aichen/work/mxfp4_opt_20260917/` —
`env.sh` (the CUDA 13 tree the JIT needs), `run.sh` (srun wrapper), `sweep.sh`,
`abba.sh` (cross-branch alternation), `switch.sh` (toggle shipped/optimized) and
`results/`. The raplab dev boxes have no SLURM; reach computelab with
`ssh -J us-proxy computelab-sc-01 'bash -s' < script.sh`.

```bash
# 8-GPU allocation (H200; use --partition=h20-3e@... --gres=gpu:8 for H20-3e)
sbatch work/mxfp4_opt_20260917/holder.sbatch

# correctness: dequant bit-exactness + reference cosine, every tier
bash work/mxfp4_opt_20260917/run.sh <jobid> <repo> tag corr \
  "python3 tests/test_mxfp4_mega_moe_sm90_correctness.py \
     --batches 1 8 16 32 64 128 256 --num-processes 8 --num-max-tokens-per-rank 256"

# three-way, one process (H200: all three arms; H20 drops nvfp4, which
# hardcodes 132 SMs)
bash work/mxfp4_opt_20260917/run.sh <jobid> <repo> tag bench \
  "python3 tests/bench_mega_moe_formats_sm90.py --arms fp8 mxfp4 nvfp4 \
     --baseline nvfp4 --batches 1 8 16 32 64 128 256 \
     --num-tests 20 --reps 5 --num-max-tokens-per-rank 256"

# schedule A/B inside one process, e.g. RS on/off at BM24
... --mxfp4-scheds 24,8,48,1,1 24,3,48,1,0

# cross-branch against megamoe_nvfp4_dev, alternating processes
bash work/mxfp4_opt_20260917/abba.sh <jobid> label 8 32 64 128 256

# EP4 (192 experts keeps 48 per rank, matching the EP1 measurements)
python3 tests/test_mxfp4_mega_moe_sm90_correctness.py \
        --num-processes 4 --num-experts 192 --batches 8 32 64 128 256
python3 tests/bench_mega_moe_formats_sm90.py \
        --num-processes 4 --num-experts 192 --arms fp8 mxfp4 --batches 8 32 64 128 256

# single-rank, no collective: the cleanest read on the weight stream, and the
# only configuration a profiler can attach to without cross-rank skew
python3 tests/bench_mega_moe_formats_sm90.py \
        --num-processes 1 --num-experts 48 --arms mxfp4 --batches 32

# GPU-free gates
bash tests/compile_all_megamoe_kernels.sh
```
