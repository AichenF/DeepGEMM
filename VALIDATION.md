# Validation record — `megamoe_nvfp4_fix_opt`

Branch = `megamoe_nvfp4` @ `ba7ee09` + 4 commits:

| commit | content |
|---|---|
| `bd4cee1` | Fix generic-to-async proxy race in all NVFP4 dequant paths (per-writer `fence.proxy.async` + writer-set `bar.sync` before WGMMA issue / mbarrier arrive; 4 site kinds, +30 lines) |
| `e5944f9` | Single-active-dispatch-warp 3-stage swapAB pipeline (hidden>=6144) + N24 swapAB bucket for I>=3072 |
| `d67b701` | Port the same race fix into the 4 experimental candidate bodies (not on the production call path) |
| `6a070f1` | Fence the config-unreachable math-side dequant branch in split L1/L2 bodies (same race class, defensive) |

Companion branch `megamoe_nvfp4_baseline` = pristine `ba7ee09` (tree-identical to upstream), kept as the A/B reference carrying the known race.

## The bug being fixed

Nondeterministic silent corruption in the SM90 NVFP4 MegaMoE kernels: the in-smem FP4->FP8
weight dequant writes through the generic proxy and WGMMA reads the same buffer through the
async proxy with no `fence.proxy.async` between them; the loader-dequant paths additionally
signal completion with a single-thread mbarrier arrive that does not order the other 63/127
writers. Symptom: ~1-7%/call per-token cosine drops to 0.77-0.90 with no error raised, at
large tokens-per-expert (m_e). Present in ALL upstream branches carrying in-smem dequant
(`megamoe_nvfp4`, `megamoe_nvfp4_fuse`, `megamoe_w4a8` — the latter contains the original
incorrect assumption in a comment: "callers enter a warpgroup-synchronous WGMMA fence",
which orders accumulator registers only, not cross-proxy smem visibility). Unfixed upstream
as of 2026-07-13.

## Kernel-level correctness evidence (all raw logs archived)

Probe protocol: unmodified `tests/test_nvfp4_mega_moe_sm90_correctness.py` driven through a
monkeypatch wrapper that only forces the BN layout selector (`diag_force_bn.py`), 56 reps x
8 GPUs = 448 timing trials per (BN, M) config; each rep runs both global-scale modes.

| tree / hardware | config | result |
|---|---|---|
| baseline `ba7ee09`, H200 NVL | BN256 M=2048/4096, 128 reps | **5 FAILs** (races, incl. at m_e=1024 — below the historically assumed threshold) |
| baseline `ba7ee09`, H200 SXM | BN256 M=2048/4096, 448 each | **3 + 16 FAILs** |
| baseline `ba7ee09`, H200 SXM | BN128 M=2048/4096, 448 each | **5 + 17 FAILs** |
| old `7add353`, H200 NVL | BN256 M=2048/4096, 128 reps | 0 FAILs (control) |
| fix `bd4cee1`, H200 NVL | BN256+BN128 x M=2048/4096, 448 each (1792) | **0 FAILs** |
| `e5944f9`, H200 SXM | BN256+BN128 x M=2048/4096, 448 each (1792) | **0 FAILs** |
| head `d67b701`, H200 SXM | BN256+BN128 x M=2048/4096, 448 each (1792) | **0 FAILs** |
| **final head `6a070f1`, H200 SXM** | BN256 M=4/8/16/24/32/48 (exercises swapAB buckets 8/16/24, the 3-stage single-dispatch-warp pipeline and the N24 path), 448 each | **0 FAILs** |
| **final head `6a070f1`, H200 SXM** | BN128 M=8/32, 448 each | **0 FAILs** |
| standard gate (20 cases) | x4 rounds on every tree above, both machines | all PASS |

`6a070f1` differs from `d67b701` only in a branch unreachable under every host plan
(split phase always selects loader dequant), so the d67b701 large-M evidence carries over;
gates + small-M probes were nevertheless re-run at the final head.

## End-to-end evidence (GSM8K, DeepSeek-R1-0528-FP4-v2, 8x H200 SXM, SGLang)

Fixed protocol: 8-shot, 1316 questions, parallel 1316, BN256, tp8 dp8 dp-attention,
4096-token/rank cap, 2 rounds (round 1 JIT warm-up, round 2 timed). Harness sha256
verified unchanged before/after the whole campaign. Accuracies independently recomputed
from raw per-question jsonl; zero garbage/repetition outputs in all rounds.

| tree | accuracy | invalid | latency | throughput |
|---|---|---|---|---|
| fix `bd4cee1` | 0.963 (1267/1316) | 0.000 | 124.3 s | 1145 tok/s |
| optimized `e5944f9` | 0.957 (1259/1316) | 0.000 | 123.7 s | 1151 tok/s |

Reference band 0.957-0.961; the 0.963 vs 0.957 delta is statistically insignificant
(McNemar p=0.15). Historical anchor of the buggy baseline: 122.1 s / 1159 tok/s /
acc 0.961 (different node). The baseline's clean e2e depends on the 4096-token cap
keeping m_e at 1024 — shown above to already be inside the trigger domain on some
hardware, so it is not a safe production configuration.

## Performance vs the (buggy) baseline, BN256

- Decode band (m_e<=8): **3-6% faster** on DSR1/V4-Pro/MiMo geometries (fence cost masked
  by memory latency + 3-stage pipeline gain); V4-Flash ~2-4% slower (already 3-stage,
  pays only the fence).
- Large-batch prefill (m_e>=128): 2-6.4% slower — the correctness cost; ablation shows it
  is the fence itself (dropping the WG barrier returns only ~1% on DSR1, nothing on Pro).
- Known cheapest recovery path (not landed): nibble-group decode layout for I<=2048
  (-27% decode instructions, also unlocks the swapAB<=16 + N24 combination).

## Known caveats (disclosed, non-blocking)

- Probe inputs are seed-deterministic across GPUs; the 448 trials per config are timing
  trials, not distinct datasets (appropriate for a race probe).
- Adversarial (hot-expert) routing was never exercised; uniform-routing thresholds may
  overestimate the safe batch size — an argument for the fix, not against it.
- BN128 remains banned for e2e serving pending a dedicated e2e revalidation (kernel-level
  it is now clean: 448x2 large-M + 448x2 small-M, zero fails).
- The E2/E3 5-9% decode-band gain is quantified from a single-sample 1-rank sweep
  (8-rank small-M timing noise is +/-36-62%); direction confirmed on two machines.
